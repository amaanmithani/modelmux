# ModelMux: design notes

What I set out to build, the decisions that shaped it, and the three places where
measurements changed the design.

## The problem

An application that depends on one LLM provider inherits that provider's rate
limits, outages and pricing. The usual fix, a retry loop with a second provider
in the client, spreads through every service and never gets the details right:
which errors to retry, what to do mid-stream, how not to hammer a provider
that's down, how to bill per customer. ModelMux moves that into one process
that speaks the API clients already use.

## Decisions

**Two wire protocols, not N adapters.** Most providers now expose an
OpenAI-compatible endpoint (Groq, Gemini, Ollama, vLLM, OpenAI itself). One
adapter covers all of them; only Anthropic's Messages API needs a translator.
The translator is where the subtle work is: system prompts move to a separate
field, consecutive tool results must merge into one user turn because the
Messages API requires alternating roles, tool arguments travel as a JSON
string one way and a JSON object the other, and streaming arrives as typed
events (`content_block_delta`, `input_json_delta`, `message_delta`) that have
to be re-cut into OpenAI chunks, usage included.

**Which errors fall back.** A 400 or 422 means the request itself is bad, so the
next provider will reject it too. Everything else that can be one provider's
problem falls back: 429, 5xx, timeouts, transport errors, and also 401, 403 and
404, because a revoked key or a model one provider doesn't host says nothing
about the others.

**Retry before the first byte, never after.** For streaming, the router opens the
upstream stream and waits for its first chunk under a separate first-byte
timeout. Only once a chunk exists is the response committed to that provider
and headers sent. A failure after that can't be retried by any proxy: the
client already has half an answer. ModelMux ends such streams with an in-band
error event and no `[DONE]`, and never caches them.

**Client disconnects are not provider failures.** A cancelled request must not
count toward a circuit breaker, or a burst of impatient users could open the
breaker on a healthy provider. The router distinguishes "the parent context was
cancelled" from "the attempt timed out", and a half-open probe that ends without
a verdict releases its slot so the breaker can't get stuck.

**Only admitted requests emit usage events.** Rejections (401, 403, 404, 429)
produce nothing, so a consumer summing events for billing never sees requests
that cost nothing. The event id is random and doubles as the idempotency key,
because webhook delivery is at-least-once.

## Where measurement changed the design

**1. The semantic cache's threshold alone is unsafe.** I started with the usual
design: embed the question, return a cached answer above cosine 0.92. Measuring
on 1,000 labeled Quora question pairs showed that with `nomic-embed-text` *no*
threshold keeps false hits under 1%, and 0.92 gave 6.8%. The worst cases were
entity swaps. *"Best budget hotels in Udaipur…"* and *"…in Munnar"* score about
1.0, as do county-for-county and city-for-city substitutions. Embeddings encode
the topic of a question more than its particulars. Raising the threshold can't
fix that, so hits now also require a lexical guard key: the same leading
question word and the same capitalised or numeric tokens. A review then pointed out that I had chosen the
threshold and reported it on the same pairs. Re-done properly, with the
threshold chosen on 2,000 calibration pairs and every number from 1,000 separate
held-out pairs, the best configuration answers about one paraphrase in seven
with well under 1% wrong answers (exact figures and intervals in the README).
That is much less than semantic-cache marketing suggests, and it's why the
feature is off by default.

**2. The official SDK can't be anonymous.** The public demo allowed requests
with no API key. The OpenAI Python SDK refuses to send a request without one, so
the demo was unusable from the most common client. Public access now also
accepts a placeholder key (`public`). Any other unknown key is still rejected,
so a mistyped tenant key can't silently fall through to anonymous limits.

**3. The scan was 2.6× slower than it needed to be.** A flat inner-product scan
over 2,000 × 1,024-dimension vectors took 2.55 ms. Unrolling the dot product
into four independent accumulators (and reslicing so the compiler drops bounds
checks) brought it to 0.99 ms, well under the embedding call it follows. That
made an approximate-nearest-neighbour index unnecessary at this scale.

## Review

Three adversarial reviewers (correctness, benchmark honesty, security) found
20 real problems, listed with their fixes in [REVIEW.md](REVIEW.md). The ones
that changed my thinking:

- **An image is not text.** Both the semantic cache and the Anthropic
  translator looked only at text parts, so the same question about two
  different images returned the same cached answer.
- **Per-IP limits don't protect a shared upstream quota.** A public demo needs a
  global ceiling, and a budget has to reserve a request's worst-case cost when
  it's admitted, not just charge afterwards.
- **A number with no spread invites doubt.** The overhead and fault results now
  come from repeated runs and show their ranges.

## Trade-offs I accepted

- **In-memory state.** Budgets, limits and caches don't survive restarts or span
  replicas. A shared store (Redis) is the obvious next step, but it would add a
  network hop to every request, which the overhead budget should measure first.
- **Reserve the worst case.** A request reserves its estimated prompt plus its
  (capped) output tokens at admission and settles the real count afterwards.
  That rejects some requests that would have fit, the price of never
  overshooting a budget.
- **Flat scan, not ANN.** Chosen after measuring, and bounded by per-scope capacity.
