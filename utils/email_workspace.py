"""Utilities for building async email workspace context panels."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

STALE_SUMMARY_THRESHOLD = 10


@dataclass
class ParticipantResolution:
    """Resolved primary external contact and participant map for a thread."""

    primary_contact: str | None
    last_non_user_sender: str | None
    participants: dict[str, dict[str, Any]]


def _normalize_email(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    return value if "@" in value else None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _is_internal_participant(email: str, owner_email: str | None, internal_domains: set[str], internal_emails: set[str]) -> bool:
    if owner_email and email == owner_email:
        return True
    if email in internal_emails:
        return True
    domain = email.split("@", maxsplit=1)[1] if "@" in email else ""
    return domain in internal_domains


def _extract_sender_from_message_text(messages: list[dict]) -> str | None:
    """Regex fallback only when structured metadata isn't available."""
    for msg in reversed(messages):
        content = msg.get("content") or ""
        match = re.search(r"\[Email from\s+([^\]]+)\]", content)
        if match:
            return match.group(1).strip().lower()
    return None


def resolve_primary_external_contact(agent, thread_id: str, messages: list[dict]) -> ParticipantResolution:
    """Resolve external contact using structured participant metadata with regex fallback."""
    owner_email = _normalize_email(os.environ.get("OWNER_EMAIL"))
    internal_emails = {_normalize_email(v) for v in _as_list(os.environ.get("INTERNAL_EMAILS", "").split(","))}
    internal_emails.discard(None)

    internal_domains = set()
    if owner_email:
        internal_domains.add(owner_email.split("@", maxsplit=1)[1])
    extra_domains = {d.strip().lower() for d in os.environ.get("INTERNAL_EMAIL_DOMAINS", "").split(",") if d.strip()}
    internal_domains.update(extra_domains)

    participants: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "email": None,
        "internal": False,
        "from_count": 0,
        "to_count": 0,
        "cc_count": 0,
        "last_seen": None,
    })
    last_non_user_sender: str | None = None

    memory = getattr(agent, "memory", None)
    events = []
    if memory:
        try:
            events = memory.find_events(channel="email", conversation_id=thread_id, limit=200)
        except Exception as exc:
            logger.debug("Failed to load email events for %s: %s", thread_id, exc)

    events = sorted(events, key=lambda e: e.timestamp or datetime.min)
    for event in events:
        meta = event.metadata or {}
        ts = event.timestamp.isoformat() if event.timestamp else None

        from_email = _normalize_email(meta.get("from") or meta.get("sender"))
        if from_email:
            p = participants[from_email]
            p["email"] = from_email
            p["from_count"] += 1
            p["internal"] = _is_internal_participant(from_email, owner_email, internal_domains, internal_emails)
            p["last_seen"] = ts
            if not p["internal"] and event.direction == "inbound":
                last_non_user_sender = from_email

        for to_email in [_normalize_email(v) for v in _as_list(meta.get("to"))]:
            if not to_email:
                continue
            p = participants[to_email]
            p["email"] = to_email
            p["to_count"] += 1
            p["internal"] = _is_internal_participant(to_email, owner_email, internal_domains, internal_emails)
            p["last_seen"] = ts

        for cc_email in [_normalize_email(v) for v in _as_list(meta.get("cc"))]:
            if not cc_email:
                continue
            p = participants[cc_email]
            p["email"] = cc_email
            p["cc_count"] += 1
            p["internal"] = _is_internal_participant(cc_email, owner_email, internal_domains, internal_emails)
            p["last_seen"] = ts

    external = [p for p in participants.values() if p["email"] and not p["internal"]]
    external.sort(key=lambda p: (p["from_count"], p["to_count"] + p["cc_count"], p["last_seen"] or ""), reverse=True)

    primary_contact = last_non_user_sender or (external[0]["email"] if external else None)

    # Fallbacks
    if not primary_contact and thread_id.startswith("external:"):
        primary_contact = _normalize_email(thread_id.split("external:", maxsplit=1)[1])
    if not primary_contact:
        primary_contact = _extract_sender_from_message_text(messages)

    return ParticipantResolution(
        primary_contact=primary_contact,
        last_non_user_sender=last_non_user_sender,
        participants={k: v for k, v in participants.items() if v["email"]},
    )


def _find_related_orgs(memory, person_id: str) -> list[dict]:
    orgs: list[dict] = []
    for edge in memory.get_edges(person_id):
        other_id = edge.to_entity_id if edge.from_entity_id == person_id else edge.from_entity_id
        entity = memory.get_entity(other_id)
        if entity and entity.type == "org":
            orgs.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "relationship": edge.relationship,
                    "confidence": edge.confidence,
                    "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
                }
            )
    return orgs


async def _fetch_attio_data(sender: str | None) -> tuple[dict | None, str, list[str]]:
    api_key = (os.environ.get("ATTIO_API_KEY") or "").strip()
    if not api_key:
        return None, "disabled", []
    if not sender or "@" not in sender:
        return None, "missing_contact", []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    errors: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            people_resp = await client.post(
                "https://api.attio.com/v2/objects/people/records/query",
                headers=headers,
                json={"filter": {"email_addresses": {"contains": sender}}, "limit": 1},
            )
            people_resp.raise_for_status()
            people_data = people_resp.json().get("data", [])
            if not people_data:
                return {"person": None, "organization": None}, "not_found", []

            person = people_data[0]
            company_refs = person.get("values", {}).get("company", [])
            org = None
            if company_refs:
                record_id = company_refs[0].get("target_record_id")
                if record_id:
                    org_resp = await client.get(
                        f"https://api.attio.com/v2/objects/companies/records/{record_id}",
                        headers=headers,
                    )
                    org_resp.raise_for_status()
                    org = org_resp.json().get("data")

            return {"person": person, "organization": org}, "ready", []
    except Exception as exc:
        msg = f"Attio lookup failed: {exc}"
        logger.debug(msg)
        errors.append(msg)
        return None, "error", errors


def _build_recent_interactions(memory, thread_id: str, limit: int = 5) -> list[dict]:
    try:
        events = memory.find_events(channel="email", conversation_id=thread_id, limit=limit)
    except Exception:
        return []

    interactions = []
    for e in sorted(events, key=lambda x: x.timestamp or datetime.min, reverse=True)[:limit]:
        interactions.append(
            {
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "direction": e.direction,
                "subject": (e.metadata or {}).get("subject"),
                "snippet": (e.content or "")[:240],
            }
        )
    return interactions


def _summary_snippet(text: str | None, size: int = 180) -> str | None:
    if not text:
        return None
    return text[:size] + ("..." if len(text) > size else "")


class WorkspacePanelCache:
    """In-memory cache with versioning + stale-while-revalidate refresh semantics."""

    def __init__(self):
        self._full_cache: dict[str, dict] = {}
        self._preview_cache: dict[str, dict] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def _compute_version(self, agent, thread_id: str, participant_email: str | None, person: dict | None, organizations: list[dict], summary_node) -> str:
        messages = agent.get_thread(thread_id)
        thread_size = len(messages)
        thread_tail = (messages[-1].get("content")[:120] if messages else "")
        person_updated = person.get("updated_at") if person else None
        org_updated = ",".join(sorted([o.get("updated_at") or "" for o in organizations]))
        summary_version = ""
        if summary_node:
            summary_version = f"{summary_node.summary_updated_at}|{summary_node.events_since_update}"
        attio_version = os.environ.get("ATTIO_NOTES_VERSION", "")
        return f"{thread_id}|{thread_size}|{thread_tail}|{participant_email}|{person_updated}|{org_updated}|{summary_version}|{attio_version}"

    def get_preview(self, thread_id: str) -> dict | None:
        return self._preview_cache.get(thread_id)

    def get_full(self, thread_id: str) -> dict | None:
        return self._full_cache.get(thread_id)

    def status(self, thread_id: str) -> str:
        task = self._tasks.get(thread_id)
        if task and not task.done():
            return "loading"
        if thread_id in self._full_cache:
            return "ready"
        if thread_id in self._preview_cache:
            return "preview"
        return "idle"

    def ensure_background_refresh(self, agent, thread_id: str):
        existing_task = self._tasks.get(thread_id)
        if existing_task and not existing_task.done():
            return

        async def _runner():
            payload = await self._build_panel_payload(agent, thread_id)
            self._full_cache[thread_id] = payload

            person = payload.get("person") or {}
            orgs = payload.get("organizations") or []
            self._preview_cache[thread_id] = {
                "name": person.get("name"),
                "company": orgs[0]["name"] if orgs else None,
                "summary_snippet": _summary_snippet(payload.get("summary")),
                "generated_at": payload.get("generated_at"),
            }

        self._tasks[thread_id] = asyncio.create_task(_runner())

    async def _build_panel_payload(self, agent, thread_id: str) -> dict:
        generated_at = datetime.now(timezone.utc).isoformat()
        errors: list[str] = []
        messages = agent.get_thread(thread_id)
        resolution = resolve_primary_external_contact(agent, thread_id, messages)
        memory = getattr(agent, "memory", None)

        thread_scope = {
            "thread_id": thread_id,
            "message_count": len(messages),
            "counterpart": resolution.primary_contact,
            "last_non_user_sender": resolution.last_non_user_sender,
            "participants": list(resolution.participants.values()),
        }

        person_payload = None
        organizations: list[dict] = []
        summary = "No summary yet."
        summary_status = "missing"
        summary_node = None

        if memory and resolution.primary_contact:
            entities = memory.find_entities(query=resolution.primary_contact, type="person", limit=1)
            person_entity = entities[0] if entities else None
            if not person_entity and "@" in resolution.primary_contact:
                local = resolution.primary_contact.split("@", maxsplit=1)[0].replace(".", " ")
                entities = memory.find_entities(query=local, type="person", limit=1)
                person_entity = entities[0] if entities else None

            if person_entity:
                person_payload = {
                    "id": person_entity.id,
                    "name": person_entity.name,
                    "aliases": person_entity.aliases,
                    "description": person_entity.description,
                    "event_count": person_entity.event_count,
                    "last_seen": person_entity.last_seen.isoformat() if person_entity.last_seen else None,
                    "updated_at": person_entity.updated_at.isoformat() if person_entity.updated_at else None,
                }
                organizations = _find_related_orgs(memory, person_entity.id)

                summary_node = memory.get_summary(f"entity:{person_entity.id}")
                if summary_node and summary_node.summary:
                    summary = summary_node.summary
                    if summary_node.events_since_update >= STALE_SUMMARY_THRESHOLD:
                        summary_status = "refreshing"
                        async def _refresh_summary():
                            try:
                                await memory.summaries.refresh_node(summary_node)
                            except Exception as exc:
                                logger.debug("Summary refresh failed for %s: %s", person_entity.id, exc)
                        asyncio.create_task(_refresh_summary())
                    else:
                        summary_status = "ready"
                else:
                    summary_status = "missing"

        recent_interactions = _build_recent_interactions(memory, thread_id, limit=5) if memory else []
        attio, attio_status, attio_errors = await _fetch_attio_data(resolution.primary_contact)
        errors.extend(attio_errors)

        version = self._compute_version(agent, thread_id, resolution.primary_contact, person_payload, organizations, summary_node)
        previous = self._full_cache.get(thread_id)
        status = "ready"
        if previous and previous.get("_version") == version:
            return previous

        return {
            "status": status,
            "generated_at": generated_at,
            "person": person_payload,
            "organizations": organizations,
            "summary": summary,
            "summary_status": summary_status,
            "recent_interactions": recent_interactions,
            "attio": attio,
            "attio_status": attio_status,
            "sections": {
                "thread_scope": thread_scope,
                "person_scope": {
                    "person_id": person_payload.get("id") if person_payload else None,
                    "organization_count": len(organizations),
                    "has_attio": attio_status == "ready",
                },
            },
            "errors": errors,
            "_version": version,
        }
