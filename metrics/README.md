# metrics/ - Cost Tracking & Instrumentation

Transparent instrumentation layer for all LLM and embedding API calls. Tracks token usage, cost, and latency across providers.

## Files

| File | Purpose |
|------|---------|
| [`__init__.py`](__init__.py) | Re-exports all clients, collector, cost functions, and models |
| [`clients.py`](clients.py) | Instrumented API clients for Anthropic, OpenAI, and LiteLLM |
| [`costs.py`](costs.py) | Cost calculation per model (pricing tables for Claude, GPT, embeddings) |
| [`collector.py`](collector.py) | `MetricsCollector` — aggregates metrics per session, per source |
| [`models.py`](models.py) | Dataclasses: `LLMCallMetric`, `EmbeddingCallMetric`, `SessionMetrics`, `MetricsSummary` |

## How It Works

Every LLM call goes through an instrumented client wrapper that:

1. Records the start time
2. Passes the call to the underlying provider
3. Captures response token counts
4. Calculates cost based on model pricing
5. Emits the metric to the collector
6. Tags the metric with a source (e.g., `"agent"`, `"extraction"`, `"learning"`)

## Source Tagging

Use `track_source()` to tag API calls by what triggered them:

```python
from metrics import track_source

with track_source("extraction"):
    response = await client.messages.create(...)
    # This call is tagged as "extraction" in metrics

with track_source("learning"):
    response = await client.messages.create(...)
    # Tagged as "learning"
```

## Use-Case Based Clients

```python
from metrics import get_llm_client, get_model_for_use_case

# Get a pre-configured client for a use case
client = get_llm_client("coding")     # Uses coding_model from config
client = get_llm_client("memory")     # Uses memory_model
client = get_llm_client("fast")       # Uses fast_model

model = get_model_for_use_case("agent")  # Returns model ID string
```

## Supported Providers

| Provider | Client | Wrapper |
|----------|--------|---------|
| Anthropic | `InstrumentedAsyncAnthropic` | Direct Claude API |
| OpenAI | `InstrumentedAsyncOpenAI` | Direct GPT API |
| LiteLLM | `InstrumentedLiteLLM` | Multi-provider (recommended) |
| LiteLLM | `AsyncLiteLLMAnthropicAdapter` | LiteLLM with Anthropic-compatible interface |

## Related Docs

- [MODELS.md](../MODELS.md) — Metric data models
- [`llm_config.py`](../llm_config.py) — Model configuration and provider detection
