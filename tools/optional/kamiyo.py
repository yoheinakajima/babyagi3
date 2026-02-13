"""KAMIYO tools for BabyAGI3 optional tool loader.

Drop this file into `tools/optional/` in a BabyAGI3 checkout and add the loader
entry shown in this package README.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, UTC
import json
import urllib.error
import urllib.request
from typing import Any

try:
    from tools import tool, tool_error
except ImportError:  # Local fallback for standalone testing
    def tool(fn=None, **_kwargs):
        def _decorate(func):
            return func

        if fn is not None:
            return _decorate(fn)
        return _decorate

    def tool_error(error: str, fix: str = None, **extras) -> dict:
        payload = {"error": error}
        if fix:
            payload["fix"] = fix
        payload.update(extras)
        return payload


_MOCK_ESCROWS: dict[str, dict[str, Any]] = {}
_MOCK_SETTLEMENTS_BY_KEY: dict[str, dict[str, Any]] = {}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_mock_mode() -> bool:
    return os.getenv("KAMIYO_MODE", "mock").lower() == "mock"


def _base_url() -> str:
    return os.getenv("KAMIYO_BASE_URL", "http://localhost:8787").rstrip("/")


def _api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("KAMIYO_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _field_exists(data: Any, dotted_path: str) -> bool:
    current = data
    for key in dotted_path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
            continue
        return False
    return True


def _refund_pct_for_score(score: int) -> int:
    if score >= 80:
        return 0
    if score >= 65:
        return 35
    if score >= 50:
        return 75
    return 100


def _http_call(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{_base_url()}{path}"
    try:
        body_bytes = None
        if payload is not None:
            body_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=body_bytes,
            headers=_api_headers(),
            method=method.upper(),
        )

        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                return {
                    "ok": False,
                    "error": "Invalid backend response payload",
                    "code": "invalid_response_type",
                    "fix": "Ensure backend returns a JSON object.",
                }
            data.setdefault("ok", True)
            return data
    except urllib.error.HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        try:
            raw = exc.read().decode("utf-8")
        except Exception:
            raw = ""

        details: Any = None
        if raw:
            try:
                details = json.loads(raw)
            except Exception:
                details = raw[:2000]

        fix = "Check backend auth, payload shape, and endpoint availability."
        if status == 401:
            fix = "Set KAMIYO_API_KEY to the Bearer token for the KAMIYO bridge (or disable auth on the bridge)."
        elif status == 404:
            fix = "Verify KAMIYO_BASE_URL points at a server that implements /babyagi/v1/* endpoints."

        return {
            "ok": False,
            "error": f"Backend rejected request ({status})",
            "code": "backend_http_error",
            "status_code": status,
            "details": details,
            "fix": fix,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Backend request failed: {exc}",
            "code": "backend_unreachable",
            "fix": "Set KAMIYO_BASE_URL correctly or switch to KAMIYO_MODE=mock.",
        }


@tool(env=["KAMIYO_ENABLED"])
def kamiyo_create_escrow_call(
    provider_id: str,
    amount: float,
    currency: str = "USDC",
    transaction_id: str = "",
    timelock_seconds: int = 3600,
    idempotency_key: str = "",
    metadata: dict = None,
) -> dict:
    """Create a KAMIYO escrow before calling a paid provider.

    Args:
        provider_id: Provider identifier.
        amount: Amount to lock.
        currency: Currency code.
        transaction_id: External transaction reference.
        timelock_seconds: Escrow expiry window.
        idempotency_key: Client-generated key for dedupe.
        metadata: Optional metadata passed to backend.
    """
    if amount <= 0:
        return {
            "ok": False,
            **tool_error(
                "amount must be greater than zero",
                code="invalid_amount",
                fix="Pass a positive amount.",
            ),
        }

    if _is_mock_mode():
        escrow_id = f"escrow_{uuid.uuid4().hex[:12]}"
        expires_at = (_utc_now() + timedelta(seconds=max(timelock_seconds, 60))).isoformat()
        _MOCK_ESCROWS[escrow_id] = {
            "escrow_id": escrow_id,
            "provider_id": provider_id,
            "amount": amount,
            "currency": currency,
            "transaction_id": transaction_id,
            "expires_at": expires_at,
            "status": "active",
        }
        return {
            "ok": True,
            "escrow_id": escrow_id,
            "status": "active",
            "expires_at": expires_at,
            "provider_id": provider_id,
            "amount": amount,
            "currency": currency,
            "trace_id": f"mock_{uuid.uuid4().hex[:10]}",
        }

    return _http_call(
        "POST",
        "/babyagi/v1/escrows",
        {
            "provider_id": provider_id,
            "amount": amount,
            "currency": currency,
            "transaction_id": transaction_id,
            "timelock_seconds": timelock_seconds,
            "idempotency_key": idempotency_key,
            "metadata": metadata or {},
        },
    )


@tool(env=["KAMIYO_ENABLED"])
def kamiyo_execute_paid_call(
    escrow_id: str,
    url: str,
    method: str = "GET",
    headers: dict = None,
    body: dict = None,
    timeout_ms: int = 10000,
) -> dict:
    """Execute a paid provider call under escrow.

    Args:
        escrow_id: Escrow identifier from `kamiyo_create_escrow_call`.
        url: Provider URL to call.
        method: HTTP method.
        headers: Request headers.
        body: Request body.
        timeout_ms: Timeout budget in milliseconds.
    """
    if _is_mock_mode():
        escrow = _MOCK_ESCROWS.get(escrow_id)
        if not escrow:
            return {
                "ok": False,
                **tool_error(
                    "escrow_id not found",
                    code="escrow_not_found",
                    fix="Call kamiyo_create_escrow_call first.",
                ),
            }

        is_bad = "bad" in url.lower() or "fail" in url.lower()
        response_payload = (
            {"data": {"result": "ok", "records": 12}}
            if not is_bad
            else {"data": {"result": "partial", "records": 1}}
        )
        latency_ms = 420 if not is_bad else max(timeout_ms - 1000, 1200)

        escrow["last_call"] = {
            "url": url,
            "method": method,
            "latency_ms": latency_ms,
            "response": response_payload,
            "http_status": 200,
        }

        return {
            "ok": True,
            "escrow_id": escrow_id,
            "http_status": 200,
            "latency_ms": latency_ms,
            "response": response_payload,
            "provider_receipt_id": f"receipt_{uuid.uuid4().hex[:10]}",
            "trace_id": f"mock_{uuid.uuid4().hex[:10]}",
        }

    return _http_call(
        "POST",
        "/babyagi/v1/execute",
        {
            "escrow_id": escrow_id,
            "url": url,
            "method": method,
            "headers": headers or {},
            "body": body or {},
            "timeout_ms": timeout_ms,
        },
    )


@tool(env=["KAMIYO_ENABLED"])
def kamiyo_assess_quality(
    escrow_id: str,
    response: dict,
    expected_fields: list[str] = None,
    max_latency_ms: int = None,
    min_quality_score: int = 70,
) -> dict:
    """Assess response quality and return a deterministic quality score.

    Args:
        escrow_id: Escrow identifier.
        response: Provider response payload.
        expected_fields: List of required dotted paths.
        max_latency_ms: Optional latency SLA.
        min_quality_score: Minimum score required to auto-release.
    """
    if _is_mock_mode():
        escrow = _MOCK_ESCROWS.get(escrow_id)
        if not escrow:
            return {
                "ok": False,
                **tool_error(
                    "escrow_id not found",
                    code="escrow_not_found",
                    fix="Call kamiyo_create_escrow_call first.",
                ),
            }

        violations: list[str] = []
        payload = response if isinstance(response, dict) else {"raw": response}
        expected_fields = expected_fields or []

        for path in expected_fields:
            if not _field_exists(payload, path):
                violations.append(f"missing_field:{path}")

        last_call = escrow.get("last_call", {})
        latency_ms = int(last_call.get("latency_ms", 0))
        if max_latency_ms and latency_ms > max_latency_ms:
            violations.append(f"latency_exceeded:{latency_ms}>{max_latency_ms}")

        quality_score = max(0, 100 - (len(violations) * 30))
        passed = quality_score >= min_quality_score

        return {
            "ok": True,
            "escrow_id": escrow_id,
            "quality_score": quality_score,
            "passed": passed,
            "violations": violations,
            "refund_recommendation_pct": _refund_pct_for_score(quality_score),
            "trace_id": f"mock_{uuid.uuid4().hex[:10]}",
        }

    return _http_call(
        "POST",
        "/babyagi/v1/quality/assess",
        {
            "escrow_id": escrow_id,
            "response": response,
            "expected_fields": expected_fields or [],
            "max_latency_ms": max_latency_ms,
            "min_quality_score": min_quality_score,
        },
    )


@tool(env=["KAMIYO_ENABLED"])
def kamiyo_settle_or_dispute(
    escrow_id: str,
    quality_score: int,
    evidence: dict = None,
    auto_dispute_threshold: int = 70,
    idempotency_key: str = "",
) -> dict:
    """Release funds or dispute based on quality score.

    Args:
        escrow_id: Escrow identifier.
        quality_score: Quality score 0-100.
        evidence: Optional evidence bundle for disputes.
        auto_dispute_threshold: Minimum score required for release.
        idempotency_key: Key used to deduplicate repeated requests.
    """
    if _is_mock_mode():
        if idempotency_key and idempotency_key in _MOCK_SETTLEMENTS_BY_KEY:
            cached = _MOCK_SETTLEMENTS_BY_KEY[idempotency_key].copy()
            cached["idempotent_replay"] = True
            return cached

        escrow = _MOCK_ESCROWS.get(escrow_id)
        if not escrow:
            return {
                "ok": False,
                **tool_error(
                    "escrow_id not found",
                    code="escrow_not_found",
                    fix="Call kamiyo_create_escrow_call first.",
                ),
            }

        action = "released" if quality_score >= auto_dispute_threshold else "disputed"
        refund_pct = _refund_pct_for_score(quality_score)
        escrow["status"] = action

        result = {
            "ok": True,
            "escrow_id": escrow_id,
            "action": action,
            "refund_pct": refund_pct,
            "settlement_tx": f"mock_tx_{uuid.uuid4().hex[:12]}",
            "trace_id": f"mock_{uuid.uuid4().hex[:10]}",
        }

        if idempotency_key:
            _MOCK_SETTLEMENTS_BY_KEY[idempotency_key] = result.copy()

        return result

    return _http_call(
        "POST",
        "/babyagi/v1/settlements/resolve",
        {
            "escrow_id": escrow_id,
            "quality_score": quality_score,
            "evidence": evidence or {},
            "auto_dispute_threshold": auto_dispute_threshold,
            "idempotency_key": idempotency_key,
        },
    )


@tool(env=["KAMIYO_ENABLED"])
def kamiyo_get_provider_reputation(provider_id: str, window_days: int = 30) -> dict:
    """Get provider reputation stats used for routing.

    Args:
        provider_id: Provider identifier.
        window_days: Evaluation window in days.
    """
    if _is_mock_mode():
        low_quality = "risky" in provider_id.lower() or "new" in provider_id.lower()
        return {
            "ok": True,
            "provider_id": provider_id,
            "reputation_score": 58.0 if low_quality else 91.0,
            "success_rate": 0.62 if low_quality else 0.95,
            "dispute_rate": 0.28 if low_quality else 0.03,
            "sample_size": 14 if low_quality else 184,
            "updated_at": _utc_now().isoformat(),
            "trace_id": f"mock_{uuid.uuid4().hex[:10]}",
            "window_days": window_days,
        }

    return _http_call(
        "GET",
        f"/babyagi/v1/providers/{provider_id}/reputation?window_days={window_days}",
    )
