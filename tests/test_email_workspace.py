from datetime import datetime, timezone
import pytest

from utils.email_workspace import WorkspacePanelCache, resolve_primary_external_contact


class _FakeSummaryNode:
    def __init__(self, summary="Known contact", events_since_update=0):
        self.summary = summary
        self.events_since_update = events_since_update
        self.summary_updated_at = datetime.now(timezone.utc)


class _FakeEntity:
    def __init__(self, entity_id: str, name: str, entity_type: str = "person"):
        now = datetime.now(timezone.utc)
        self.id = entity_id
        self.name = name
        self.type = entity_type
        self.aliases = []
        self.description = ""
        self.event_count = 2
        self.last_seen = now
        self.updated_at = now


class _FakeEdge:
    def __init__(self, from_entity_id, to_entity_id, relationship="works_at", confidence=0.8):
        self.from_entity_id = from_entity_id
        self.to_entity_id = to_entity_id
        self.relationship = relationship
        self.confidence = confidence


class _FakeEvent:
    def __init__(self, direction, metadata, content="hello"):
        self.direction = direction
        self.metadata = metadata
        self.timestamp = datetime.now(timezone.utc)
        self.content = content


class _FakeSummaries:
    def __init__(self):
        self.refreshed = False

    async def refresh_node(self, node):
        self.refreshed = True


class _FakeMemory:
    def __init__(self, stale=False):
        self.person = _FakeEntity("p1", "Person Example", "person")
        self.org = _FakeEntity("o1", "Example Org", "org")
        self.summary_node = _FakeSummaryNode(events_since_update=20 if stale else 0)
        self.summaries = _FakeSummaries()

    def find_events(self, channel=None, conversation_id=None, limit=10):
        return [
            _FakeEvent("inbound", {"from": "person@example.com", "to": ["owner@company.com"], "cc": []}, content="first"),
            _FakeEvent("outbound", {"from": "owner@company.com", "to": ["person@example.com"], "cc": []}, content="reply"),
        ]

    def find_entities(self, query=None, type=None, limit=10):
        if type == "person":
            return [self.person]
        return []

    def get_edges(self, entity_id, direction="both"):
        if entity_id == "p1":
            return [_FakeEdge("p1", "o1")]
        return []

    def get_entity(self, entity_id):
        if entity_id == "p1":
            return self.person
        if entity_id == "o1":
            return self.org
        return None

    def get_summary(self, key):
        if key == "entity:p1":
            return self.summary_node
        return None


class _FakeAgent:
    def __init__(self, memory):
        self.memory = memory
        self._threads = {
            "external:person@example.com": [{"role": "user", "content": "message"}],
        }

    def get_thread(self, thread_id):
        return self._threads.get(thread_id, [])


def test_resolve_primary_external_contact_uses_structured_metadata(monkeypatch):
    monkeypatch.setenv("OWNER_EMAIL", "owner@company.com")
    memory = _FakeMemory()
    agent = _FakeAgent(memory)

    result = resolve_primary_external_contact(agent, "external:person@example.com", agent.get_thread("external:person@example.com"))

    assert result.primary_contact == "person@example.com"
    assert result.last_non_user_sender == "person@example.com"
    assert any(not p["internal"] for p in result.participants.values())


@pytest.mark.asyncio
async def test_stale_summary_returns_refreshing_status(monkeypatch):
    monkeypatch.setenv("OWNER_EMAIL", "owner@company.com")
    cache = WorkspacePanelCache()
    agent = _FakeAgent(_FakeMemory(stale=True))

    payload = await cache._build_panel_payload(agent, "external:person@example.com")

    assert payload["summary"] == "Known contact"
    assert payload["summary_status"] == "refreshing"
    assert payload["sections"]["thread_scope"]["counterpart"] == "person@example.com"


@pytest.mark.asyncio
async def test_preview_and_status_available_after_background_refresh(monkeypatch):
    monkeypatch.setenv("OWNER_EMAIL", "owner@company.com")
    cache = WorkspacePanelCache()
    agent = _FakeAgent(_FakeMemory())

    assert cache.status("external:person@example.com") == "idle"
    cache.ensure_background_refresh(agent, "external:person@example.com")

    # Let background task finish
    import asyncio
    await asyncio.sleep(0.02)

    assert cache.status("external:person@example.com") == "ready"
    preview = cache.get_preview("external:person@example.com")
    assert preview is not None
    assert preview["name"] == "Person Example"
