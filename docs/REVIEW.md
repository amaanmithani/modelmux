# Adversarial review, round 1

Three independent reviewers (correctness, benchmark honesty, security) read the
code and benchmarks with instructions to reproduce every claim with a failing
test or a rerun. All findings below were fixed in the same change, each with a
regression test unless noted.

## Correctness

| finding | fix |
|---|---|
| Semantic cache ignored image parts: the same text with a different image returned the first image's answer | requests with non-text content are ineligible for the semantic cache |
| Anthropic adapter silently dropped `image_url` parts | images translate to `base64`/`url` image blocks; parts it can't express return a fallbackable 501 so another provider can serve them |
| Number of semantic-cache scopes was unbounded (vary `max_tokens` → new scope) | scopes are LRU-evicted beyond `MaxScopes` (default 1000) |
| Requests that skipped an open breaker reported `fallbacks: 0` | a skip counts as a fallback (header, usage event and metric) |
| A request sent before the breaker opened could release, close or re-arm it during half-open | `Allow` returns a probe token; only the probe's outcome changes an open breaker |
| A malformed upstream chunk (`index: -1`) panicked the handler | choice and tool-call indexes are bounds-checked |
| A stream that stalled after its first byte was never timed out | whole-stream timeout (`timeouts.stream`, default 5m) |
| Anthropic's first chunk was the empty role chunk, so an overload right after `message_start` couldn't fall back | the role now rides on the first real content chunk |

## Security (public demo)

| finding | fix |
|---|---|
| Denied requests with random `provider/<model>` names minted unbounded metric series | label set only after authorisation; direct names collapse to `direct:<provider>` |
| Budget checked only against usage so far; `max_tokens` uncapped; no global limit protecting the owner's upstream quota | worst-case cost (prompt estimate + capped output) is reserved at admission and settled after; public `max_output_tokens`, `max_prompt_chars`, `global_rpm`, `global_daily_tokens` |
| IPv6 clients could rotate addresses within their /64 | IPv6 clients are keyed by /64 |
| Slow-trickled request bodies held connections | `ReadTimeout` 30s (responses still stream) |
| Anonymous callers could `force`-cache, saw raw upstream error text, and could read `/metrics` | `force` ignored for public; generic upstream errors for public; `metrics.require_key` |
| Cross-site `text/plain` POSTs could spend a visitor's quota | `Content-Type: application/json` required |
| No `.dockerignore` | added |
| `trusted_proxy_hops: 1` is only correct if the host appends exactly one X-Forwarded-For hop | not code: verify after deploy by sending spoofed `X-Forwarded-For` values and confirming they share one limit |

## Benchmark honesty

| finding | fix |
|---|---|
| The semantic threshold was chosen and reported on the same 1,000 pairs (the "1% false hits" broke on about half of random splits) | τ chosen on 2,000 calibration pairs; every reported number from 1,000 separate held-out pairs, with 95% Wilson intervals. The headline dropped from 21.4% to about 14% |
| Pairwise false-hit rate isn't what a cache user sees (lookup = nearest neighbour among many entries) | added a cache simulation over the held-out set |
| The Udaipur/Munnar example was hard-coded prose and only true for one model | examples are computed per model from the data |
| "Recovered 7,591 of 7,591 across seven scenarios" summed trivial cases, counted the control, and treated breaker skips as failures | per-scenario table with consistent definitions; no aggregate sum |
| Midstream reruns could collapse from client connection churn and still "pass" | pooled client transport; the run fails on any client transport error; 3 runs per scenario with ranges |
| Overhead shown as a single number from 3 rounds, measured only on a path with no limits | overhead computed per round and reported as median with min–max; added a tenant path with rate limit + budget active, and a streaming time-to-first-byte path |
| Claimed: `bench/data` is committed | not so: it's gitignored (checked with `git ls-files`) |
