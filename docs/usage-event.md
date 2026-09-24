# Usage event schema (v1)

One JSON object per admitted request (after auth, validation, rate limit and budget checks).
Rejected requests (401/400/403/404/429) emit nothing. `id` is unique per request and is the
idempotency key consumers must dedupe on.

| field | type | notes |
|---|---|---|
| `v` | int | schema version, currently `1` |
| `id` | string | 128-bit random hex |
| `ts` | RFC 3339 | request start, UTC |
| `tenant` | string | tenant name, or `public` |
| `model` | string | the alias the client asked for |
| `provider` | string | provider that served it; empty on cache hits |
| `upstream_model` | string | the provider's model id |
| `prompt_tokens`, `completion_tokens` | int | from upstream usage when reported |
| `tokens_estimated` | bool | `true` when upstream sent no usage and counts are ~chars/4 |
| `latency_ms` | int | end to end, as seen by ModelMux |
| `cache` | string | `miss` · `exact` · `semantic` · `bypass` |
| `status` | int | HTTP status returned; `499` = client disconnected; `502` on a mid-stream failure |
| `stream` | bool | |
| `fallbacks` | int | targets that failed before the one that served |

Cache hits are recorded with the cached response's token counts but are **not** charged
against the tenant's budget.
