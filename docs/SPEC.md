# ModelMux — spec

A single-binary LLM gateway that exposes one OpenAI-compatible API in front of
many model providers, with fallback, caching, per-tenant budgets and rate limits.
Every number in the README must come from a committed script in `bench/`.

## Goals (v1)

| # | Capability | Done when |
|---|---|---|
| G1 | OpenAI-compatible `POST /v1/chat/completions` (JSON + SSE streaming) and `GET /v1/models` | the official `openai` Python client works against it unmodified |
| G2 | Providers via two wire protocols: **OpenAI-compatible** (OpenAI, Groq, Gemini's OpenAI endpoint, Ollama, vLLM/mini-vLLM) and **Anthropic Messages** (translated both ways, incl. streaming) | contract tests per adapter against recorded fixtures |
| G3 | Model aliases → ordered fallback chains; retry the next target on 429 / 5xx / timeout / connection error **before the first byte is streamed**; per-provider circuit breaker | fault-injection tests show the chain advancing and the breaker opening/closing |
| G4 | Exact-match response cache (deterministic requests only) + semantic cache (embedding cosine ≥ τ, scoped per tenant+model) | hit/miss tests; semantic false-hit rate measured on a labeled paraphrase set |
| G5 | Tenants: hashed API keys, monthly token budgets, token-bucket rate limits per key and per client IP | budget exhaustion returns 429 with OpenAI-shaped error body |
| G6 | Usage events (tenant, model, provider, tokens in/out, latency, cache status) to a pluggable sink (stdout JSONL / HTTP webhook) — consumed later by the metering engine | event schema versioned in `docs/usage-event.md` |
| G7 | Observability: Prometheus `/metrics` (requests, latency histograms, fallbacks, cache hits, breaker state), structured JSON logs, `/healthz` | metrics asserted in tests |
| G8 | Playground page at `/` for the public demo | works on phone width |

## Non-goals (v1)

- Persistent storage (budgets and caches are in-memory; restart resets them — stated in README).
- Image/audio/embeddings passthrough APIs (only chat; embeddings used internally for semantic cache).
- Multi-node coordination of budgets/rate limits.
- Tool-calling translation for Anthropic beyond plain text and tool_use passthrough (documented gap if not done).

## Success metrics (measured, committed)

- **Gateway overhead:** added p99 latency < 5 ms at 2,000 req/s against a fixed-latency stub upstream (k6, same machine, cache disabled). Reported as `p99(via gateway) − p99(direct)`.
- **Fallback correctness:** 100% of injected upstream failures before first byte are recovered when a healthy target exists.
- **Semantic cache:** hit rate and false-hit rate per τ on a labeled paraphrase set; chosen default τ justified by the curve.
- **Coverage:** ≥ 85% statement coverage, CI-enforced.

## Architecture

```
client ─► auth (tenant) ─► rate limit ─► budget check ─► cache lookup ──hit──► response
                                                             │ miss
                                                             ▼
                                               router: alias → [target₁, target₂, …]
                                                             │   breaker + retry-before-first-byte
                                                             ▼
                                               provider adapter (openai-compat | anthropic)
                                                             │
                                  response/stream ◄──────────┘ ─► cache store ─► usage event ─► sink
```

Packages: `internal/api` (wire types), `internal/provider` (adapters + stub), `internal/router`,
`internal/cache`, `internal/tenant`, `internal/usage`, `internal/server` (HTTP, SSE, metrics), `cmd/modelmux`, `cmd/stubllm`.

## Complexity

- Routing, auth, rate limit, exact cache: O(1) per request.
- Semantic cache: flat inner-product scan, O(n·d) per lookup over n entries of dimension d, bounded by a per-scope capacity (default 2,000). Lookup time at capacity is measured and reported; an ANN index is out of scope unless the measurement says it's needed.
- Memory: O(cache capacity · d) + O(tenants).

## Stack

Go 1.23+, standard library `net/http` (no web framework), `prometheus/client_golang`, `gopkg.in/yaml.v3`.
Benchmarks: k6. Deploy: Docker image on Render (free tier) with Groq + Gemini free-tier keys.
