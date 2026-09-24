package cache

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/provider"
)

// SemanticQuery returns the text to embed for a request and whether the
// request is eligible for semantic caching at all. Only single-turn requests
// qualify (system prompts plus exactly one user message, no tools): matching
// the last message of a multi-turn conversation would ignore its context.
func SemanticQuery(req *api.ChatRequest) (string, bool) {
	if len(req.Tools) > 0 {
		return "", false
	}
	var q string
	users := 0
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
		case "user":
			users++
			q = m.Content.Text()
		default:
			return "", false
		}
	}
	return q, users == 1 && q != ""
}

// SemanticScope keys the semantic index: tenant, model, system prompts and
// sampling parameters must all match for two requests to share answers.
func SemanticScope(tenant string, req *api.ChatRequest) string {
	r := req.Clone()
	r.Messages = r.Messages[:0]
	for _, m := range req.Messages {
		if m.Role != "user" {
			r.Messages = append(r.Messages, m)
		}
	}
	return Key("sem|"+tenant, r)
}

type semEntry struct {
	vec     []float32 // unit-normalised
	query   string
	resp    *api.ChatResponse
	expires time.Time
}

type semScope struct {
	entries []semEntry
	next    int // ring-buffer write position once full
}

// Semantic is an embedding-similarity cache. Lookup is a flat scan of the
// scope: O(n·d) for n entries of dimension d, with n bounded by capacity.
type Semantic struct {
	emb       provider.Embedder
	model     string
	threshold float32
	capacity  int
	ttl       time.Duration
	now       func() time.Time

	mu     sync.RWMutex
	scopes map[string]*semScope
}

// SemanticConfig configures a Semantic cache.
type SemanticConfig struct {
	Embedder  provider.Embedder
	Model     string
	Threshold float32 // cosine similarity required for a hit
	Capacity  int     // entries per scope; oldest overwritten first
	TTL       time.Duration
	Now       func() time.Time
}

// NewSemantic returns a semantic cache.
func NewSemantic(c SemanticConfig) *Semantic {
	if c.Capacity <= 0 {
		c.Capacity = 2000
	}
	if c.TTL <= 0 {
		c.TTL = time.Hour
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return &Semantic{emb: c.Embedder, model: c.Model, threshold: c.Threshold, capacity: c.Capacity,
		ttl: c.TTL, now: c.Now, scopes: map[string]*semScope{}}
}

// Match is the result of a lookup.
type Match struct {
	Resp       *api.ChatResponse // nil on miss
	Similarity float32           // best similarity seen (hit or not)
	Query      string            // the cached query that matched
	Vec        []float32         // the query's embedding, reusable for Put
}

// Lookup embeds query and returns the most similar live entry in scope if it
// clears the threshold.
func (s *Semantic) Lookup(ctx context.Context, scope, query string) (Match, error) {
	vecs, err := s.emb.Embed(ctx, s.model, []string{query})
	if err != nil {
		return Match{}, err
	}
	v := normalize(vecs[0])
	m := Match{Vec: v, Similarity: -1}
	now := s.now()
	s.mu.RLock()
	defer s.mu.RUnlock()
	sc := s.scopes[scope]
	if sc == nil {
		return m, nil
	}
	best := -1
	for i := range sc.entries {
		e := &sc.entries[i]
		if now.After(e.expires) || len(e.vec) != len(v) {
			continue
		}
		if sim := dot(v, e.vec); sim > m.Similarity {
			m.Similarity, best = sim, i
		}
	}
	if best >= 0 && m.Similarity >= s.threshold {
		m.Resp, m.Query = sc.entries[best].resp, sc.entries[best].query
	}
	return m, nil
}

// Put stores a response under the query embedding from a previous Lookup.
func (s *Semantic) Put(scope string, vec []float32, query string, resp *api.ChatResponse) {
	if len(vec) == 0 {
		return
	}
	e := semEntry{vec: vec, query: query, resp: resp, expires: s.now().Add(s.ttl)}
	s.mu.Lock()
	defer s.mu.Unlock()
	sc := s.scopes[scope]
	if sc == nil {
		sc = &semScope{}
		s.scopes[scope] = sc
	}
	if len(sc.entries) < s.capacity {
		sc.entries = append(sc.entries, e)
		return
	}
	sc.entries[sc.next] = e
	sc.next = (sc.next + 1) % s.capacity
}

// Threshold returns the configured similarity threshold.
func (s *Semantic) Threshold() float32 { return s.threshold }

func normalize(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	if n == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(n))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
