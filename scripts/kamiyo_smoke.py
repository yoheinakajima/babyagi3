from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _assert_ok(resp: dict[str, Any], ctx: str) -> None:
    if resp.get("ok") is True:
        return
    details = resp.get("details")
    msg = f"{ctx} failed"
    if isinstance(details, str) and details:
        msg += f": {details}"
    raise AssertionError(f"{msg}: {resp}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Default to a deterministic, zero-network smoke test.
    os.environ.setdefault("KAMIYO_ENABLED", "1")
    os.environ.setdefault("KAMIYO_MODE", "mock")

    from tools.optional.kamiyo import (
        kamiyo_assess_quality,
        kamiyo_create_escrow_call,
        kamiyo_execute_paid_call,
        kamiyo_settle_or_dispute,
    )

    good = kamiyo_create_escrow_call(
        provider_id="alpha",
        amount=1.0,
        currency="USDC",
        transaction_id="smoke-good",
        idempotency_key="smoke-good-create",
    )
    _assert_ok(good, "create escrow (good)")

    good_exec = kamiyo_execute_paid_call(
        escrow_id=good["escrow_id"],
        url="https://example-good",
    )
    _assert_ok(good_exec, "execute paid call (good)")

    good_quality = kamiyo_assess_quality(
        escrow_id=good["escrow_id"],
        response=good_exec.get("response"),
        expected_fields=["data.result"],
        max_latency_ms=1_000,
        min_quality_score=70,
    )
    _assert_ok(good_quality, "assess quality (good)")

    good_settle = kamiyo_settle_or_dispute(
        escrow_id=good["escrow_id"],
        quality_score=int(good_quality["quality_score"]),
        auto_dispute_threshold=70,
        idempotency_key="smoke-good-settle",
    )
    _assert_ok(good_settle, "settle/dispute (good)")
    if good_settle.get("action") != "released":
        raise AssertionError(f'expected action="released" for good path, got: {good_settle}')

    bad = kamiyo_create_escrow_call(
        provider_id="beta",
        amount=1.0,
        currency="USDC",
        transaction_id="smoke-bad",
        idempotency_key="smoke-bad-create",
    )
    _assert_ok(bad, "create escrow (bad)")

    bad_exec = kamiyo_execute_paid_call(
        escrow_id=bad["escrow_id"],
        url="https://example-bad",
    )
    _assert_ok(bad_exec, "execute paid call (bad)")

    bad_quality = kamiyo_assess_quality(
        escrow_id=bad["escrow_id"],
        response=bad_exec.get("response"),
        expected_fields=["data.result", "data.missing"],
        max_latency_ms=500,
        min_quality_score=70,
    )
    _assert_ok(bad_quality, "assess quality (bad)")

    bad_settle = kamiyo_settle_or_dispute(
        escrow_id=bad["escrow_id"],
        quality_score=int(bad_quality["quality_score"]),
        auto_dispute_threshold=70,
        idempotency_key="smoke-bad-settle",
    )
    _assert_ok(bad_settle, "settle/dispute (bad)")
    if bad_settle.get("action") != "disputed":
        raise AssertionError(f'expected action="disputed" for bad path, got: {bad_settle}')

    print(
        json.dumps(
            {
                "mode": os.getenv("KAMIYO_MODE", "mock"),
                "good": {
                    "escrow_id": good["escrow_id"],
                    "quality_score": good_quality["quality_score"],
                    "action": good_settle["action"],
                },
                "bad": {
                    "escrow_id": bad["escrow_id"],
                    "quality_score": bad_quality["quality_score"],
                    "action": bad_settle["action"],
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
