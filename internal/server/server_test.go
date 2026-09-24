package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/cache"
	"github.com/amaanmithani/modelmux/internal/provider"
	"github.com/amaanmithani/modelmux/internal/router"
	"github.com/amaanmithani/modelmux/internal/tenant"
	"github.com/amaanmithani/modelmux/internal/usage"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

type memSink struct {
	mu  sync.Mutex
	evs []usage.Event
}

func (m *memSink) Emit(e usage.Event) { m.mu.Lock(); m.evs = append(m.evs, e); m.mu.Unlock() }
func (m *memSink) Close() error       { return nil }
func (m *memSink) all() []usage.Event {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]usage.Event(nil), m.evs...)
}

type fixture struct {
	srv     *httptest.Server
	a, b    *provider.Fake
	sink    *memSink
	metrics *Metrics
}

func newFixture(t *testing.T, mutate func(*Config)) *fixture {
	t.Helper()
	f := &fixture{
		a:    &provider.Fake{ProviderName: "a", Reply: "hello from a"},
		b:    &provider.Fake{ProviderName: "b", Reply: "hello from b"},
		sink: &memSink{},
	}
	f.metrics = NewMetrics()
	r, err := router.New([]provider.Provider{f.a, f.b}, map[string][]router.Target{
		"fast":  {{Provider: "a", Model: "a1"}, {Provider: "b", Model: "b1"}},
		"other": {{Provider: "b", Model: "b2"}},
	}, router.Options{Observer: f.metrics})
	if err != nil {
		t.Fatal(err)
	}
	tm, err := tenant.New(tenant.Config{
		Tenants: []tenant.TenantConfig{
			{Name: "acme", KeySHA256: tenant.HashKey("sk-acme"), MonthlyTokens: 1000},
			{Name: "tiny", KeySHA256: tenant.HashKey("sk-tiny"), RPM: 6, Models: []string{"fast"}},
		},
		Public: tenant.PublicConfig{Enabled: true, Models: []string{"fast"}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Router: r, Tenants: tm, Exact: cache.NewExact(100, time.Hour, nil), Sink: f.sink, Metrics: f.metrics}
	if mutate != nil {
		mutate(&cfg)
	}
	f.srv = httptest.NewServer(New(cfg).Handler())
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fixture) post(t *testing.T, key string, body any, hdr ...string) *http.Response {
	t.Helper()
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, f.srv.URL+"/v1/chat/completions", strings.NewReader(string(b)))
	if s, ok := body.(string); ok {
		req, _ = http.NewRequest(http.MethodPost, f.srv.URL+"/v1/chat/completions", strings.NewReader(s))
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func chatBody(model, text string, stream bool, temp *float64) map[string]any {
	m := map[string]any{"model": model, "stream": stream,
		"messages": []map[string]string{{"role": "user", "content": text}}}
	if temp != nil {
		m["temperature"] = *temp
	}
	return m
}

func decode[T any](t *testing.T, r io.Reader) T {
	t.Helper()
	var v T
	if err := json.NewDecoder(r).Decode(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

// readSSE returns decoded chunks and whether [DONE] was seen.
func readSSE(t *testing.T, r io.Reader) ([]map[string]any, bool) {
	t.Helper()
	var out []map[string]any
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return out, true
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatal(err)
		}
		out = append(out, m)
	}
	return out, false
}

func sseText(chunks []map[string]any) string {
	var sb strings.Builder
	for _, c := range chunks {
		for _, ch := range c["choices"].([]any) {
			d := ch.(map[string]any)["delta"].(map[string]any)
			if s, ok := d["content"].(string); ok {
				sb.WriteString(s)
			}
		}
	}
	return sb.String()
}

func TestChatNonStreaming(t *testing.T) {
	f := newFixture(t, nil)
	resp := f.post(t, "sk-acme", chatBody("fast", "hi", false, nil))
	if resp.StatusCode != 200 || resp.Header.Get(HeaderProvider) != "a" || resp.Header.Get(HeaderCache) != "miss" ||
		resp.Header.Get(HeaderFallbacks) != "0" {
		t.Fatalf("status %d headers %v", resp.StatusCode, resp.Header)
	}
	out := decode[api.ChatResponse](t, resp.Body)
	if out.Choices[0].Message.Content.Text() != "hello from a" {
		t.Fatal(out)
	}
	evs := f.sink.all()
	if len(evs) != 1 || evs[0].Tenant != "acme" || evs[0].Provider != "a" || evs[0].UpstreamModel != "a1" ||
		evs[0].PromptTokens != 1 || evs[0].CompletionTokens != 3 || evs[0].Status != 200 || evs[0].Estimated {
		t.Fatalf("usage event: %+v", evs)
	}
}

func TestChatStreamingWithAndWithoutUsage(t *testing.T) {
	f := newFixture(t, nil)
	resp := f.post(t, "sk-acme", chatBody("fast", "hi", true, nil))
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatal(ct)
	}
	chunks, done := readSSE(t, resp.Body)
	if !done || sseText(chunks) != "hello from a" {
		t.Fatalf("done=%v text=%q", done, sseText(chunks))
	}
	for _, c := range chunks {
		if _, has := c["usage"]; has {
			t.Fatal("usage chunk sent without stream_options.include_usage")
		}
	}
	body := chatBody("fast", "hi", true, nil)
	body["stream_options"] = map[string]bool{"include_usage": true}
	chunks, _ = readSSE(t, f.post(t, "sk-acme", body).Body)
	last := chunks[len(chunks)-1]
	if last["usage"] == nil || len(last["choices"].([]any)) != 0 {
		t.Fatalf("trailing usage chunk missing: %v", last)
	}
	evs := f.sink.all()
	if len(evs) != 2 || !evs[0].Stream || evs[0].CompletionTokens != 3 {
		t.Fatalf("stream usage: %+v", evs)
	}
}

func TestFallbackAndErrorMapping(t *testing.T) {
	f := newFixture(t, nil)
	f.a.FailNext(&provider.UpstreamError{Provider: "a", Status: 503})
	resp := f.post(t, "sk-acme", chatBody("fast", "hi", false, nil))
	if resp.Header.Get(HeaderProvider) != "b" || resp.Header.Get(HeaderFallbacks) != "1" {
		t.Fatalf("fallback headers: %v", resp.Header)
	}
	cases := []struct {
		err    error
		status int
		code   string
	}{
		{&provider.UpstreamError{Provider: "b", Status: 400, Message: "bad"}, 400, "upstream_rejected"},
		{&provider.UpstreamError{Provider: "b", Status: 429}, 429, "upstream_rate_limited"},
		{&provider.UpstreamError{Provider: "b", Status: 500}, 502, "upstream_error"},
		{&provider.UpstreamError{Provider: "b", Err: context.DeadlineExceeded}, 504, "upstream_timeout"},
		{errors.New("weird"), 502, "upstream_error"},
	}
	for _, c := range cases {
		f.b.FailNext(c.err)
		resp := f.post(t, "sk-acme", chatBody("other", "hi", false, nil))
		e := decode[api.ErrorBody](t, resp.Body)
		if resp.StatusCode != c.status || e.Error.Code == nil || *e.Error.Code != c.code {
			t.Errorf("%v: got %d %v", c.err, resp.StatusCode, e.Error)
		}
	}
	f.b.FailNext(&provider.UpstreamError{Provider: "b", Status: 503})
	resp = f.post(t, "sk-acme", chatBody("other", "hi", true, nil))
	if resp.StatusCode != 502 {
		t.Fatalf("stream error before first byte should be a normal HTTP error, got %d", resp.StatusCode)
	}
}

func TestNoHealthyUpstream(t *testing.T) {
	f := newFixture(t, nil)
	for i := 0; i < 5; i++ {
		f.b.FailNext(&provider.UpstreamError{Provider: "b", Status: 503})
		f.post(t, "sk-acme", chatBody("other", "hi", false, nil))
	}
	resp := f.post(t, "sk-acme", chatBody("other", "hi", false, nil))
	if resp.StatusCode != 503 {
		t.Fatalf("breaker open: got %d", resp.StatusCode)
	}
	if v := testutil.ToFloat64(f.metrics.breaker.WithLabelValues("b")); v != 1 {
		t.Fatalf("breaker gauge %v", v)
	}
}

func TestMidStreamErrorIsInBand(t *testing.T) {
	f := newFixture(t, nil)
	f.b.MidStreamErr = io.ErrUnexpectedEOF
	resp := f.post(t, "sk-acme", chatBody("other", "hi", true, api.Ptr(0.0)))
	chunks, done := readSSE(t, resp.Body)
	if done {
		t.Fatal("a failed stream must not end with [DONE]")
	}
	if chunks[len(chunks)-1]["error"] == nil {
		t.Fatalf("last event should be an error: %v", chunks[len(chunks)-1])
	}
	f.b.MidStreamErr = nil
	resp = f.post(t, "sk-acme", chatBody("other", "hi", false, api.Ptr(0.0)))
	if resp.Header.Get(HeaderCache) != "miss" {
		t.Fatal("a failed stream must not populate the cache")
	}
	if ev := f.sink.all()[0]; ev.Status != 502 {
		t.Fatalf("usage status for failed stream: %d", ev.Status)
	}
}

func TestExactCache(t *testing.T) {
	f := newFixture(t, nil)
	zero := api.Ptr(0.0)
	f.post(t, "sk-acme", chatBody("fast", "cache me", false, zero))
	resp := f.post(t, "sk-acme", chatBody("fast", "cache me", false, zero))
	if resp.Header.Get(HeaderCache) != "exact" || f.a.Calls() != 1 {
		t.Fatalf("second identical request should hit: %v calls=%d", resp.Header.Get(HeaderCache), f.a.Calls())
	}
	// A streamed request with the same content is served from the same entry.
	chunks, done := readSSE(t, f.post(t, "sk-acme", chatBody("fast", "cache me", true, zero)).Body)
	if !done || sseText(chunks) != "hello from a" || f.a.Calls() != 1 {
		t.Fatal("stream replay of cached response")
	}
	// Other tenant: no sharing.
	if r := f.post(t, "", chatBody("fast", "cache me", false, zero)); r.Header.Get(HeaderCache) != "miss" {
		t.Fatal("cache leaked across tenants")
	}
	// Non-deterministic: not cached unless forced; bypass skips.
	f.post(t, "sk-acme", chatBody("fast", "warm", false, api.Ptr(0.7)))
	if r := f.post(t, "sk-acme", chatBody("fast", "warm", false, api.Ptr(0.7))); r.Header.Get(HeaderCache) != "miss" {
		t.Fatal("temperature 0.7 should not be cached")
	}
	f.post(t, "sk-acme", chatBody("fast", "forced", false, api.Ptr(0.7)), HeaderCacheMode, "force")
	if r := f.post(t, "sk-acme", chatBody("fast", "forced", false, api.Ptr(0.7)), HeaderCacheMode, "force"); r.Header.Get(HeaderCache) != "exact" {
		t.Fatal("force should cache")
	}
	if r := f.post(t, "sk-acme", chatBody("fast", "cache me", false, zero), HeaderCacheMode, "bypass"); r.Header.Get(HeaderCache) != "bypass" {
		t.Fatal("bypass")
	}
	// Streamed miss populates the cache too.
	readSSE(t, f.post(t, "sk-acme", chatBody("fast", "streamed first", true, zero)).Body)
	if r := f.post(t, "sk-acme", chatBody("fast", "streamed first", false, zero)); r.Header.Get(HeaderCache) != "exact" {
		t.Fatal("streamed response should populate cache")
	}
	var hits int
	for _, e := range f.sink.all() {
		if e.Cache == "exact" {
			hits++
		}
	}
	if hits != 4 {
		t.Fatalf("exact hits in usage: %d", hits)
	}
}

func TestSemanticCache(t *testing.T) {
	var sem *cache.Semantic
	f := newFixture(t, func(c *Config) {
		sem = cache.NewSemantic(cache.SemanticConfig{Embedder: &provider.Fake{}, Threshold: 0.95})
		c.Semantic = sem
		c.Exact = nil
	})
	zero := api.Ptr(0.0)
	f.post(t, "sk-acme", chatBody("fast", "what is the capital of France", false, zero))
	resp := f.post(t, "sk-acme", chatBody("fast", "what the capital of France is", false, zero))
	if resp.Header.Get(HeaderCache) != "semantic" || resp.Header.Get(HeaderSimilarity) == "" || f.a.Calls() != 1 {
		t.Fatalf("semantic hit expected: %v", resp.Header)
	}
	if r := f.post(t, "sk-acme", chatBody("fast", "something unrelated entirely", false, zero)); r.Header.Get(HeaderCache) != "miss" {
		t.Fatal("unrelated should miss")
	}
}

type brokenEmbedder struct{}

func (brokenEmbedder) Embed(context.Context, string, []string) ([][]float32, error) {
	return nil, errors.New("embedder down")
}

func TestSemanticEmbedFailureDegradesToMiss(t *testing.T) {
	f := newFixture(t, func(c *Config) {
		c.Semantic = cache.NewSemantic(cache.SemanticConfig{Embedder: brokenEmbedder{}, Threshold: 0.9})
	})
	resp := f.post(t, "sk-acme", chatBody("fast", "hi", false, api.Ptr(0.0)))
	if resp.StatusCode != 200 {
		t.Fatal(resp.StatusCode)
	}
	if testutil.ToFloat64(f.metrics.embedErrs) != 1 {
		t.Fatal("embed error metric")
	}
}

func TestAuthAndValidation(t *testing.T) {
	f := newFixture(t, nil)
	cases := []struct {
		key    string
		body   any
		status int
		code   string
	}{
		{"wrong", chatBody("fast", "x", false, nil), 401, "invalid_api_key"},
		{"sk-acme", "{not json", 400, "invalid_json"},
		{"sk-acme", map[string]any{"model": "fast"}, 400, "missing_field"},
		{"sk-acme", chatBody("nope", "x", false, nil), 404, "model_not_found"},
		{"", chatBody("other", "x", false, nil), 403, "model_not_allowed"},
	}
	for _, c := range cases {
		resp := f.post(t, c.key, c.body)
		e := decode[api.ErrorBody](t, resp.Body)
		if resp.StatusCode != c.status || *e.Error.Code != c.code {
			t.Errorf("%v: %d %v", c.body, resp.StatusCode, e.Error)
		}
	}
	if len(f.sink.all()) != 0 {
		t.Fatal("rejected requests must not emit usage events")
	}
}

func TestRateLimitAndBudget(t *testing.T) {
	f := newFixture(t, nil)
	f.post(t, "sk-tiny", chatBody("fast", "x", false, nil))
	resp := f.post(t, "sk-tiny", chatBody("fast", "x", false, nil))
	e := decode[api.ErrorBody](t, resp.Body)
	if resp.StatusCode != 429 || *e.Error.Code != "rate_limit_exceeded" || resp.Header.Get("Retry-After") == "" {
		t.Fatalf("rate limit: %d %v", resp.StatusCode, e.Error)
	}
	// acme has 1000 tokens/month; each fake call costs prompt+completion.
	f.a.Reply = strings.Repeat("w ", 600)
	f.post(t, "sk-acme", chatBody("fast", "x", false, nil))
	f.post(t, "sk-acme", chatBody("fast", "x", false, nil))
	resp = f.post(t, "sk-acme", chatBody("fast", "x", false, nil))
	e = decode[api.ErrorBody](t, resp.Body)
	if resp.StatusCode != 429 || *e.Error.Code != "insufficient_quota" {
		t.Fatalf("budget: %d %v", resp.StatusCode, e.Error)
	}
}

func TestCacheHitsAreNotCharged(t *testing.T) {
	f := newFixture(t, nil)
	f.a.Reply = strings.Repeat("w ", 400) // ~401 tokens per miss
	zero := api.Ptr(0.0)
	for i := 0; i < 10; i++ {
		if r := f.post(t, "sk-acme", chatBody("fast", "same", false, zero)); r.StatusCode != 200 {
			t.Fatalf("request %d: %d (hits must not consume the 1000-token budget)", i, r.StatusCode)
		}
	}
}

func TestEstimatedTokensWhenUpstreamOmitsUsage(t *testing.T) {
	req := &api.ChatRequest{Messages: []api.Message{{Role: "user", Content: api.TextContent("12345678")}}}
	resp := &api.ChatResponse{Choices: []api.Choice{{Message: api.Message{Content: api.TextContent("abcd"),
		ToolCalls: []api.ToolCall{{Function: api.FunctionCall{Name: "f", Arguments: "{}"}}}}}}}
	in, out, est := estimateTokens(req, resp)
	if in != 2 || out != 2 || !est {
		t.Fatalf("%d %d %v", in, out, est)
	}
}

func TestModelsHealthMetricsPlayground(t *testing.T) {
	f := newFixture(t, nil)
	get := func(path, key string) *http.Response {
		req, _ := http.NewRequest(http.MethodGet, f.srv.URL+path, nil)
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { resp.Body.Close() })
		return resp
	}
	ml := decode[api.ModelList](t, get("/v1/models", "sk-acme").Body)
	if len(ml.Data) != 2 {
		t.Fatalf("acme sees all aliases: %+v", ml)
	}
	if ml := decode[api.ModelList](t, get("/v1/models", "").Body); len(ml.Data) != 1 || ml.Data[0].ID != "fast" {
		t.Fatalf("public sees only allowed: %+v", ml)
	}
	if get("/v1/models", "bad").StatusCode != 401 {
		t.Fatal("bad key on /v1/models")
	}
	if get("/healthz", "").StatusCode != 200 {
		t.Fatal("healthz")
	}
	f.post(t, "sk-acme", chatBody("fast", "hi", false, nil))
	b, _ := io.ReadAll(get("/metrics", "").Body)
	for _, want := range []string{`modelmux_requests_total{cache="miss",model="fast",status="200"} 1`,
		`modelmux_upstream_attempts_total{outcome="ok",provider="a"} 1`, `modelmux_tokens_total`} {
		if !strings.Contains(string(b), want) {
			t.Errorf("metrics missing %q", want)
		}
	}
	page := get("/", "")
	body, _ := io.ReadAll(page.Body)
	if page.StatusCode != 200 || !strings.Contains(string(body), "ModelMux") {
		t.Fatal("playground")
	}
}

func TestClientIP(t *testing.T) {
	s := New(Config{TrustedProxyHops: 1})
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "10.0.0.1:5555"
	r.Header.Add("X-Forwarded-For", "6.6.6.6, 1.2.3.4")
	if ip := s.clientIP(r); ip != "1.2.3.4" {
		t.Fatalf("rightmost trusted hop: %s (leftmost is spoofable)", ip)
	}
	s2 := New(Config{TrustedProxyHops: 2})
	if ip := s2.clientIP(r); ip != "6.6.6.6" {
		t.Fatalf("two hops: %s", ip)
	}
	s0 := New(Config{})
	if ip := s0.clientIP(r); ip != "10.0.0.1" {
		t.Fatalf("no proxy trust: %s", ip)
	}
	r.RemoteAddr = "weird"
	if ip := s0.clientIP(r); ip != "weird" {
		t.Fatal(ip)
	}
	r2 := httptest.NewRequest("GET", "/", nil)
	r2.RemoteAddr = "10.0.0.2:1"
	if ip := New(Config{TrustedProxyHops: 3}).clientIP(r2); ip != "10.0.0.2" {
		t.Fatalf("too few hops falls back to socket: %s", ip)
	}
}

func TestClientDisconnectMidStream(t *testing.T) {
	f := newFixture(t, nil)
	f.b.Reply = strings.Repeat("word ", 2000)
	ctx, cancel := context.WithCancel(context.Background())
	b, _ := json.Marshal(chatBody("other", "hi", true, nil))
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, f.srv.URL+"/v1/chat/completions", strings.NewReader(string(b)))
	req.Header.Set("Authorization", "Bearer sk-acme")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 64)
	_, _ = resp.Body.Read(buf)
	cancel()
	resp.Body.Close()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if evs := f.sink.all(); len(evs) == 1 {
			if evs[0].Status != 499 && evs[0].Status != 200 {
				t.Fatalf("disconnect status %d", evs[0].Status)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("no usage event after disconnect")
}

func TestSimulationHeaderAndRoutes(t *testing.T) {
	f := newFixture(t, func(c *Config) { c.AllowSimulation = true })
	resp := f.post(t, "sk-acme", chatBody("fast", "hi", false, api.Ptr(0.0)), HeaderSimulate, "primary-down")
	if resp.Header.Get(HeaderProvider) != "b" || resp.Header.Get(HeaderFallbacks) != "1" || resp.Header.Get(HeaderCache) != "bypass" {
		t.Fatalf("simulated: %v", resp.Header)
	}
	off := newFixture(t, nil)
	if r := off.post(t, "sk-acme", chatBody("fast", "hi", false, nil), HeaderSimulate, "primary-down"); r.Header.Get(HeaderProvider) != "a" {
		t.Fatal("simulation must be ignored unless allowed")
	}
	req, _ := http.NewRequest(http.MethodGet, f.srv.URL+"/v1/routes", nil)
	rr, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer rr.Body.Close()
	var body struct {
		Simulation bool                           `json:"simulation"`
		Routes     map[string][]map[string]string `json:"routes"`
	}
	_ = json.NewDecoder(rr.Body).Decode(&body)
	if !body.Simulation || len(body.Routes) != 1 || body.Routes["fast"][1]["provider"] != "b" {
		t.Fatalf("public routes: %+v", body)
	}
	bad, _ := http.NewRequest(http.MethodGet, f.srv.URL+"/v1/routes", nil)
	bad.Header.Set("Authorization", "Bearer nope")
	br, _ := http.DefaultClient.Do(bad)
	br.Body.Close()
	if br.StatusCode != 401 {
		t.Fatal("bad key")
	}
}
