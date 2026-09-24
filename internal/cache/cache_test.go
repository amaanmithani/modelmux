package cache

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/provider"
)

func req(msgs ...api.Message) *api.ChatRequest {
	return &api.ChatRequest{Model: "m", Temperature: api.Ptr(0.0), Messages: msgs}
}

func user(s string) api.Message { return api.Message{Role: "user", Content: api.TextContent(s)} }
func sys(s string) api.Message  { return api.Message{Role: "system", Content: api.TextContent(s)} }

func resp(s string) *api.ChatResponse {
	return &api.ChatResponse{ID: s, Choices: []api.Choice{{Message: api.Message{Role: "assistant", Content: api.TextContent(s)}}}}
}

func TestCacheable(t *testing.T) {
	if !Cacheable(req(user("x"))) {
		t.Fatal("temperature 0 is cacheable")
	}
	r := req(user("x"))
	r.Temperature = nil
	if Cacheable(r) {
		t.Fatal("unset temperature is not deterministic")
	}
	r.Temperature = api.Ptr(0.7)
	if Cacheable(r) {
		t.Fatal("temperature 0.7 is not cacheable")
	}
}

func TestKeyIgnoresTransportFieldsButNotContent(t *testing.T) {
	a := req(user("x"))
	b := req(user("x"))
	b.Stream, b.User, b.StreamOptions = true, "u1", &api.StreamOptions{IncludeUsage: true}
	if Key("t", a) != Key("t", b) {
		t.Fatal("stream/user should not change key")
	}
	if Key("t", a) == Key("t2", a) {
		t.Fatal("tenants must not share keys")
	}
	if Key("t", a) == Key("t", req(user("y"))) {
		t.Fatal("content must change key")
	}
	c := req(user("x"))
	c.MaxTokens = api.Ptr(5)
	if Key("t", a) == Key("t", c) {
		t.Fatal("params must change key")
	}
}

func TestExactLRUAndTTL(t *testing.T) {
	now := time.Unix(0, 0)
	c := NewExact(2, time.Minute, func() time.Time { return now })
	c.Put("a", resp("a"))
	c.Put("b", resp("b"))
	if _, ok := c.Get("a"); !ok { // a is now most recent
		t.Fatal("a missing")
	}
	c.Put("c", resp("c")) // evicts b
	if _, ok := c.Get("b"); ok {
		t.Fatal("b should be evicted (LRU)")
	}
	c.Put("a", resp("a2"))
	if r, _ := c.Get("a"); r.ID != "a2" || c.Len() != 2 {
		t.Fatal("update in place")
	}
	now = now.Add(2 * time.Minute)
	if _, ok := c.Get("a"); ok {
		t.Fatal("expired entry returned")
	}
	if c.Len() != 1 {
		t.Fatalf("expired entry should be removed on read, len=%d", c.Len())
	}
	d := NewExact(0, 0, nil)
	if d.cap != 10_000 || d.ttl != time.Hour {
		t.Fatal("defaults")
	}
}

func TestSemanticQueryEligibility(t *testing.T) {
	if q, ok := SemanticQuery(req(sys("s"), user("hello"))); !ok || q != "hello" {
		t.Fatal("single turn with system is eligible")
	}
	multi := req(user("a"), api.Message{Role: "assistant", Content: api.TextContent("b")}, user("c"))
	if _, ok := SemanticQuery(multi); ok {
		t.Fatal("multi-turn is not eligible")
	}
	if _, ok := SemanticQuery(req(user("a"), user("b"))); ok {
		t.Fatal("two user messages not eligible")
	}
	tools := req(user("a"))
	tools.Tools = []api.Tool{{Type: "function"}}
	if _, ok := SemanticQuery(tools); ok {
		t.Fatal("tools not eligible")
	}
	if _, ok := SemanticQuery(req(user(""))); ok {
		t.Fatal("empty not eligible")
	}
}

func TestSemanticScopeSeparatesSystemPromptsAndParams(t *testing.T) {
	a := SemanticScope("t", req(sys("pirate"), user("hi")))
	if a != SemanticScope("t", req(sys("pirate"), user("different question"))) {
		t.Fatal("user text must not affect scope")
	}
	if a == SemanticScope("t", req(sys("lawyer"), user("hi"))) {
		t.Fatal("system prompt must affect scope")
	}
	if a == SemanticScope("u", req(sys("pirate"), user("hi"))) {
		t.Fatal("tenant must affect scope")
	}
}

func TestSemanticLookupPutThresholdAndCapacity(t *testing.T) {
	now := time.Unix(0, 0)
	s := NewSemantic(SemanticConfig{Embedder: &provider.Fake{}, Threshold: 0.9, Capacity: 2, TTL: time.Minute,
		Now: func() time.Time { return now }})
	ctx := context.Background()
	m, err := s.Lookup(ctx, "sc", "what is the capital of france")
	if err != nil || m.Resp != nil || m.Vec == nil {
		t.Fatalf("empty scope: %+v %v", m, err)
	}
	s.Put("sc", m.Vec, "what is the capital of france", resp("paris"))
	hit, _ := s.Lookup(ctx, "sc", "what the capital of france is") // same bag of words
	if hit.Resp == nil || hit.Resp.ID != "paris" || hit.Similarity < 0.99 {
		t.Fatalf("expected hit: %+v", hit)
	}
	miss, _ := s.Lookup(ctx, "sc", "how do volcanoes form")
	if miss.Resp != nil || miss.Similarity >= 0.9 {
		t.Fatalf("expected miss: %+v", miss)
	}
	if other, _ := s.Lookup(ctx, "other", "what is the capital of france"); other.Resp != nil {
		t.Fatal("scopes must be isolated")
	}
	// Capacity 2: the third put overwrites the oldest.
	for i, q := range []string{"alpha beta", "gamma delta"} {
		mm, _ := s.Lookup(ctx, "sc", q)
		s.Put("sc", mm.Vec, q, resp(fmt.Sprint(i)))
	}
	if gone, _ := s.Lookup(ctx, "sc", "what is the capital of france"); gone.Resp != nil {
		t.Fatal("oldest entry should be overwritten")
	}
	now = now.Add(2 * time.Minute)
	if exp, _ := s.Lookup(ctx, "sc", "alpha beta"); exp.Resp != nil {
		t.Fatal("expired entry returned")
	}
	s.Put("sc", nil, "x", resp("x")) // no vector: ignored
	if s.Threshold() != 0.9 {
		t.Fatal("threshold")
	}
}

type failEmbedder struct{}

func (failEmbedder) Embed(context.Context, string, []string) ([][]float32, error) {
	return nil, errors.New("down")
}

func TestSemanticEmbedErrorAndDefaults(t *testing.T) {
	s := NewSemantic(SemanticConfig{Embedder: failEmbedder{}, Threshold: 0.9})
	if _, err := s.Lookup(context.Background(), "s", "q"); err == nil {
		t.Fatal("embed error should propagate")
	}
	if s.capacity != 2000 || s.ttl != time.Hour {
		t.Fatal("defaults")
	}
	if v := normalize([]float32{0, 0}); v[0] != 0 {
		t.Fatal("zero vector")
	}
}

func TestGuardKey(t *testing.T) {
	same := [][2]string{
		{"How can I learn Go?", "how can i learn Go"},
		{"What is the capital of France?", "What's the capital city of France?"},
	}
	for _, p := range same {
		if GuardKey(p[0]) != GuardKey(p[1]) {
			t.Errorf("%q vs %q should share a guard: %s / %s", p[0], p[1], GuardKey(p[0]), GuardKey(p[1]))
		}
	}
	diff := [][2]string{
		{"Best budget hotels in Udaipur?", "Best budget hotels in Munnar?"},
		{"How can I learn drifting?", "Where can I learn drifting?"},
		{"Top 10 movies of 2019", "Top 10 movies of 2020"},
	}
	for _, p := range diff {
		if GuardKey(p[0]) == GuardKey(p[1]) {
			t.Errorf("%q vs %q must not share a guard", p[0], p[1])
		}
	}
	if GuardKey("") != "" || GuardKey("...") != "" {
		t.Fatal("empty")
	}
}

func TestSemanticGuardBlocksEntitySwaps(t *testing.T) {
	ctx := context.Background()
	for _, disable := range []bool{false, true} {
		s := NewSemantic(SemanticConfig{Embedder: &provider.Fake{}, Threshold: 0.5, DisableGuard: disable})
		m, _ := s.Lookup(ctx, "sc", "best hotels in Udaipur")
		s.Put("sc", m.Vec, "best hotels in Udaipur", resp("udaipur"))
		got, _ := s.Lookup(ctx, "sc", "best hotels in Munnar")
		if hit := got.Resp != nil; hit != disable {
			t.Errorf("guard disabled=%v: hit=%v", disable, hit)
		}
	}
}

type constEmbedder struct{ v []float32 }

func (c constEmbedder) Embed(context.Context, string, []string) ([][]float32, error) {
	return [][]float32{c.v}, nil
}

// BenchmarkSemanticLookup measures the flat scan at the default per-scope
// capacity (2000) and mxbai-embed-large's dimension (1024), excluding the
// embedding call itself.
func BenchmarkSemanticLookup(b *testing.B) {
	const dim, n = 1024, 2000
	vec := func(seed int) []float32 {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32((i*31+seed*17)%97) - 48
		}
		return normalize(v)
	}
	s := NewSemantic(SemanticConfig{Embedder: constEmbedder{vec(-1)}, Threshold: 0.99, Capacity: n})
	for i := 0; i < n; i++ {
		s.Put("sc", vec(i), "what is x", resp("x"))
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Lookup(ctx, "sc", "what is y"); err != nil {
			b.Fatal(err)
		}
	}
}

func TestDotMatchesNaive(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 7, 1024} {
		a, b := make([]float32, n), make([]float32, n)
		var want float64
		for i := range a {
			a[i], b[i] = float32(i%5)-2, float32(i%3)+0.5
			want += float64(a[i]) * float64(b[i])
		}
		if got := float64(dot(a, b)); got-want > 1e-3 || want-got > 1e-3 {
			t.Fatalf("n=%d: dot=%v want %v", n, got, want)
		}
	}
}
