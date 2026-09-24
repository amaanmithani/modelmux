# ModelMux

One OpenAI-compatible API in front of many LLM providers — with fallback chains,
circuit breakers, exact + semantic caching, per-tenant budgets and rate limits.

> Work in progress. See [docs/SPEC.md](docs/SPEC.md) for scope and success metrics.
> Every number in this README will come from a committed script in `bench/`.

## Results

<!-- RESULTS:START -->
### Gateway overhead

k6 constant-arrival-rate 2000 req/s for 30s, 3 alternating rounds per path, median of rounds; stub upstream with fixed 50ms latency over HTTP; caches off; k6, gateway and stub on the same machine. Machine: Apple M1 Pro, 8 cores.

| | p50 | p95 | p99 |
|---|---|---|---|
| direct to upstream | 50.16 ms | 50.27 ms | 50.56 ms |
| through ModelMux | 50.30 ms | 50.48 ms | 51.08 ms |
| **added by ModelMux** | **0.14 ms** | **0.21 ms** | **0.52 ms** |

Achieved 1996 req/s with 0% errors and 0 dropped iterations. Gateway process: peak RSS 45.5 MB, median 36% of one core.

### Fallback under injected faults

2000 requests per scenario (half streaming), 32 concurrent clients, real HTTP end to end; chain = [a, b].

| scenario | success | recovered / recoverable | HTTP requests that reached `a` | p99 |
|---|---|---|---|---|
| flaky-503: a returns 503 on 30% of requests; breaker effectively off | 100.0% | 638/638 | 2000 | 7.3 ms |
| hanging: a hangs on 30% of requests (never sends a byte); first-byte/request timeout 200ms | 100.0% | 638/638 | 2000 | 205.1 ms |
| a-down-breaker-off: a returns 503 on every request; breaker off: every request pays a failed attempt | 100.0% | 2000/2000 | 2000 | 11.3 ms |
| a-down-breaker-on: a returns 503 on every request; breaker on (5 failures, 1s cooldown): a is skipped while open | 100.0% | 2000/2000 | 29 | 7.1 ms |
| a-refusing: a refuses TCP connections; breaker on | 100.0% | 2000/2000 | 0 | 6.6 ms |
| both-dead: control: no healthy target exists | 0.0% | — | 0 | 2.7 ms |
| midstream: a drops 30% of streams after the first chunk (not recoverable by design: bytes already sent) | 83.9% | 315/315 | 2000 | 7.1 ms |

Streams that fail *after* the first byte can't be retried by any proxy, because the client already has partial output. ModelMux ends those with an in-band error event and no `[DONE]` (323 in the midstream run), and never caches them.

### Semantic cache: how much can it safely hit?

1000 GLUE QQP validation pairs (500 duplicate / 500 non-duplicate, seeded sample); cosine similarity of embeddings via http://localhost:11434/v1 (the runtime path); a hit = similarity >= threshold (and, when guarded, cache.GuardKey equal — the same function the cache uses).

| embedding model | AUC | no guard: τ, hit rate @ ≤1% false hits | with lexical guard | embed p50 |
|---|---|---|---|---|
| nomic-embed-text | 0.801 | no threshold qualifies | τ=0.935, 17.2% (1.0% false) | 18 ms |
| all-minilm | 0.879 | τ=0.955, 16.0% (0.8% false) | τ=0.945, 13.2% (0.8% false) | 14 ms |
| mxbai-embed-large | 0.897 | τ=0.955, 19.6% (0.6% false) | τ=0.925, 21.4% (1.0% false) | 27 ms |

Embeddings score entity swaps as near-identical: with nomic-embed-text, *"best budget hotels in Udaipur"* vs *"…in Munnar"* has cosine ≈ 1.0. The lexical guard (`cache.GuardKey`) requires the same leading question word and the same capitalised/numeric tokens. Even so, a safe semantic cache catches only about a fifth of true paraphrases. That's why it's off by default, and off in the public demo.

Scan cost at capacity (2,000 entries × 1,024 dims, flat inner product): **0.99 ms** per lookup (`go test -bench SemanticLookup ./internal/cache`), small next to the embedding call itself.
<!-- RESULTS:END -->
