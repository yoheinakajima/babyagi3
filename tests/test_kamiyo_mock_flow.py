def test_kamiyo_mock_flow_release_and_dispute(monkeypatch):
    monkeypatch.setenv("KAMIYO_ENABLED", "1")
    monkeypatch.setenv("KAMIYO_MODE", "mock")

    from tools.optional import kamiyo

    kamiyo._MOCK_ESCROWS.clear()
    kamiyo._MOCK_SETTLEMENTS_BY_KEY.clear()

    good = kamiyo.kamiyo_create_escrow_call(
        provider_id="alpha",
        amount=1.0,
        currency="USDC",
        transaction_id="test-good",
        idempotency_key="test-good-create",
    )
    assert good["ok"] is True

    good_exec = kamiyo.kamiyo_execute_paid_call(
        escrow_id=good["escrow_id"],
        url="https://example-good",
    )
    assert good_exec["ok"] is True

    good_quality = kamiyo.kamiyo_assess_quality(
        escrow_id=good["escrow_id"],
        response=good_exec["response"],
        expected_fields=["data.result"],
        max_latency_ms=1_000,
        min_quality_score=70,
    )
    assert good_quality["ok"] is True
    assert good_quality["passed"] is True

    good_settle = kamiyo.kamiyo_settle_or_dispute(
        escrow_id=good["escrow_id"],
        quality_score=good_quality["quality_score"],
        auto_dispute_threshold=70,
        idempotency_key="test-good-settle",
    )
    assert good_settle["ok"] is True
    assert good_settle["action"] == "released"

    bad = kamiyo.kamiyo_create_escrow_call(
        provider_id="beta",
        amount=1.0,
        currency="USDC",
        transaction_id="test-bad",
        idempotency_key="test-bad-create",
    )
    assert bad["ok"] is True

    bad_exec = kamiyo.kamiyo_execute_paid_call(
        escrow_id=bad["escrow_id"],
        url="https://example-bad",
    )
    assert bad_exec["ok"] is True

    bad_quality = kamiyo.kamiyo_assess_quality(
        escrow_id=bad["escrow_id"],
        response=bad_exec["response"],
        expected_fields=["data.result", "data.missing"],
        max_latency_ms=500,
        min_quality_score=70,
    )
    assert bad_quality["ok"] is True
    assert bad_quality["passed"] is False

    bad_settle = kamiyo.kamiyo_settle_or_dispute(
        escrow_id=bad["escrow_id"],
        quality_score=bad_quality["quality_score"],
        auto_dispute_threshold=70,
        idempotency_key="test-bad-settle",
    )
    assert bad_settle["ok"] is True
    assert bad_settle["action"] == "disputed"


def test_kamiyo_settlement_idempotency(monkeypatch):
    monkeypatch.setenv("KAMIYO_ENABLED", "1")
    monkeypatch.setenv("KAMIYO_MODE", "mock")

    from tools.optional import kamiyo

    kamiyo._MOCK_ESCROWS.clear()
    kamiyo._MOCK_SETTLEMENTS_BY_KEY.clear()

    created = kamiyo.kamiyo_create_escrow_call(
        provider_id="alpha",
        amount=1.0,
        currency="USDC",
        transaction_id="test-idem",
        idempotency_key="test-idem-create",
    )
    assert created["ok"] is True

    idem_key = "test-idem-settle"

    first = kamiyo.kamiyo_settle_or_dispute(
        escrow_id=created["escrow_id"],
        quality_score=100,
        idempotency_key=idem_key,
    )
    assert first["ok"] is True
    assert "idempotent_replay" not in first

    second = kamiyo.kamiyo_settle_or_dispute(
        escrow_id=created["escrow_id"],
        quality_score=0,
        idempotency_key=idem_key,
    )
    assert second["ok"] is True
    assert second["idempotent_replay"] is True
    assert second["action"] == first["action"]
