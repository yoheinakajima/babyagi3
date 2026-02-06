"""
Unit tests for metrics/costs.py

Tests cover:
- LLM cost calculation for known and unknown models
- Embedding cost calculation
- Token estimation from text
- Cost formatting for display
- Model info retrieval
- Pricing table integrity
"""

import pytest

from metrics.costs import (
    LLM_PRICING,
    EMBEDDING_PRICING,
    DEFAULT_LLM_PRICING,
    DEFAULT_EMBEDDING_PRICING,
    calculate_cost,
    calculate_embedding_cost,
    estimate_tokens,
    format_cost,
    get_model_info,
)


# =============================================================================
# LLM Cost Calculation Tests
# =============================================================================


class TestCalculateCost:
    """Test calculate_cost for various models."""

    def test_known_model_claude_sonnet(self):
        # claude-sonnet-4-20250514: $3.00/$15.00 per 1M tokens
        cost = calculate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        assert cost == 18.0  # $3 input + $15 output

    def test_known_model_gpt4o(self):
        # gpt-4o: $2.50/$10.00 per 1M tokens
        cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == 12.5  # $2.50 + $10.00

    def test_known_model_gpt4o_mini(self):
        # gpt-4o-mini: $0.15/$0.60 per 1M tokens
        cost = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == 0.75  # $0.15 + $0.60

    def test_unknown_model_uses_default(self):
        # Default: $3.00/$15.00 per 1M tokens
        cost = calculate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
        expected = (3.00 + 15.00)
        assert cost == expected

    def test_zero_tokens(self):
        cost = calculate_cost("claude-sonnet-4-20250514", 0, 0)
        assert cost == 0.0

    def test_small_token_count(self):
        # 1000 input, 500 output tokens on claude-sonnet-4
        cost = calculate_cost("claude-sonnet-4-20250514", 1000, 500)
        expected = round((1000 / 1_000_000) * 3.00 + (500 / 1_000_000) * 15.00, 6)
        assert cost == expected

    def test_rounding(self):
        cost = calculate_cost("claude-sonnet-4-20250514", 1, 1)
        # Should be rounded to 6 decimal places
        assert isinstance(cost, float)
        cost_str = f"{cost:.6f}"
        assert len(cost_str.split(".")[-1]) <= 6

    def test_claude_opus(self):
        # claude-opus-4-20250514: $15.00/$75.00 per 1M tokens
        cost = calculate_cost("claude-opus-4-20250514", 1_000_000, 1_000_000)
        assert cost == 90.0

    def test_haiku_cheap(self):
        # claude-3-5-haiku: $0.80/$4.00 per 1M tokens
        cost = calculate_cost("claude-3-5-haiku-20241022", 1_000_000, 1_000_000)
        assert cost == 4.8


# =============================================================================
# Embedding Cost Calculation Tests
# =============================================================================


class TestCalculateEmbeddingCost:
    """Test calculate_embedding_cost for various models."""

    def test_known_model_small(self):
        # text-embedding-3-small: $0.02 per 1M tokens
        cost = calculate_embedding_cost("text-embedding-3-small", 1_000_000)
        assert cost == 0.02

    def test_known_model_large(self):
        # text-embedding-3-large: $0.13 per 1M tokens
        cost = calculate_embedding_cost("text-embedding-3-large", 1_000_000)
        assert cost == 0.13

    def test_unknown_model_uses_default(self):
        cost = calculate_embedding_cost("unknown-embedding", 1_000_000)
        assert cost == DEFAULT_EMBEDDING_PRICING

    def test_zero_tokens(self):
        cost = calculate_embedding_cost("text-embedding-3-small", 0)
        assert cost == 0.0

    def test_small_count(self):
        cost = calculate_embedding_cost("text-embedding-3-small", 1000)
        expected = round((1000 / 1_000_000) * 0.02, 6)
        assert cost == expected


# =============================================================================
# Token Estimation Tests
# =============================================================================


class TestEstimateTokens:
    """Test estimate_tokens heuristic."""

    def test_empty_string(self):
        # Empty string should return at least 1
        assert estimate_tokens("") == 1

    def test_short_text(self):
        # ~4 chars per token
        assert estimate_tokens("hello world") == max(1, len("hello world") // 4)

    def test_longer_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_single_char(self):
        assert estimate_tokens("x") == 1


# =============================================================================
# Cost Formatting Tests
# =============================================================================


class TestFormatCost:
    """Test format_cost for display."""

    def test_very_small_cost(self):
        result = format_cost(0.000001)
        assert result.startswith("$")
        assert "0.000001" in result

    def test_small_cost(self):
        result = format_cost(0.005)
        assert result.startswith("$")
        assert "0.005" in result

    def test_medium_cost(self):
        result = format_cost(0.50)
        assert result == "$0.500"

    def test_large_cost(self):
        result = format_cost(15.50)
        assert result == "$15.50"

    def test_zero(self):
        result = format_cost(0.0)
        assert result.startswith("$")

    def test_boundary_very_small(self):
        result = format_cost(0.00009)
        assert "0.0000" in result  # 6 decimal places

    def test_boundary_small(self):
        result = format_cost(0.009)
        assert "0.009" in result  # 4 decimal places


# =============================================================================
# Model Info Tests
# =============================================================================


class TestGetModelInfo:
    """Test get_model_info retrieval."""

    def test_known_llm(self):
        info = get_model_info("claude-sonnet-4-20250514")
        assert info["type"] == "llm"
        assert info["model"] == "claude-sonnet-4-20250514"
        assert info["input_price_per_million"] == 3.00
        assert info["output_price_per_million"] == 15.00

    def test_known_embedding(self):
        info = get_model_info("text-embedding-3-small")
        assert info["type"] == "embedding"
        assert info["price_per_million"] == 0.02

    def test_unknown_model(self):
        info = get_model_info("totally-unknown-model")
        assert info["type"] == "unknown"
        assert "default pricing" in info["note"].lower()


# =============================================================================
# Pricing Table Integrity Tests
# =============================================================================


class TestPricingTables:
    """Verify pricing table structure and values."""

    def test_llm_pricing_all_tuples(self):
        for model, pricing in LLM_PRICING.items():
            assert isinstance(pricing, tuple), f"{model} pricing should be tuple"
            assert len(pricing) == 2, f"{model} pricing should have 2 values"
            assert pricing[0] >= 0, f"{model} input price should be non-negative"
            assert pricing[1] >= 0, f"{model} output price should be non-negative"

    def test_embedding_pricing_all_floats(self):
        for model, price in EMBEDDING_PRICING.items():
            assert isinstance(price, (int, float)), f"{model} price should be numeric"
            assert price >= 0, f"{model} price should be non-negative"

    def test_default_pricing_reasonable(self):
        assert DEFAULT_LLM_PRICING[0] > 0
        assert DEFAULT_LLM_PRICING[1] > 0
        assert DEFAULT_EMBEDDING_PRICING > 0

    def test_claude_models_present(self):
        assert "claude-sonnet-4-20250514" in LLM_PRICING
        assert "claude-opus-4-20250514" in LLM_PRICING

    def test_gpt_models_present(self):
        assert "gpt-4o" in LLM_PRICING
        assert "gpt-4o-mini" in LLM_PRICING

    def test_embedding_models_present(self):
        assert "text-embedding-3-small" in EMBEDDING_PRICING
        assert "text-embedding-3-large" in EMBEDDING_PRICING
