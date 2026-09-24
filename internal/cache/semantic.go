package cache

import (
	"container/list"
	"context"
	"math"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

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
			// Images etc. aren't embedded; two questions about different
			// images would look identical.
			if m.Content.HasNonText() {
				return "", false
			}
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

var tokenRE = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9'\-.]*`)

// GuardKey is a lexical fingerprint two queries must share for a semantic hit:
// the leading word (question type: what/how/where...) plus the set of
// named-entity-like tokens (capitalised words after the first, and anything
// containing a digit). Embeddings score entity swaps ("hotels in Udaipur" vs
// "hotels in Munnar") as near-identical; this guard rejects them.
// See bench/results/semantic.json for the measured effect.
func GuardKey(q string) string {
	toks := tokenRE.FindAllString(q, -1)
	if len(toks) == 0 {
		return ""
	}
	// Leading word, with contractions folded ("what's" -> "what").
	lead, _, _ := strings.Cut(strings.ToLower(toks[0]), "'")
	ents := map[string]bool{}
	for i, t := range toks {
		t = strings.Trim(t, "'-.")
		if t == "" {
			continue
		}
		hasDigit := strings.IndexFunc(t, unicode.IsDigit) >= 0
		capitalised := i > 0 && unicode.IsUpper(rune(t[0])) && t != "I"
		if hasDigit || capitalised {
			ents[strings.ToLower(t)] = true
		}
	}
	keys := make([]string, 0, len(ents))
	for k := range ents {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return lead + "|" + strings.Join(keys, ",")
}

type semEntry struct {
	vec     []float32 // unit-normalised
	guard   string
	query   string
	resp    *api.ChatResponse
	expires time.Time
}

type semScope struct {
	key     string
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
	guard     bool

	maxScopes int

	mu     sync.Mutex
	scopes map[string]*list.Element // of *semScope, most recently used at front
	lru    *list.List
}

// SemanticConfig configures a Semantic cache.
type SemanticConfig struct {
	Embedder  provider.Embedder
	Model     string
	Threshold float32 // cosine similarity required for a hit
	Capacity  int     // entries per scope; oldest overwritten first
	TTL       time.Duration
	Now       func() time.Time
	// DisableGuard turns off the lexical guard (GuardKey). Only for
	// measurement: without it, entity-swapped questions get wrong answers.
	DisableGuard bool
	// MaxScopes bounds the number of scopes (tenant × model × system prompt ×
	// params); the least recently used scope is evicted. Default 1000.
	MaxScopes int
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
	if c.MaxScopes <= 0 {
		c.MaxScopes = 1000
	}
	return &Semantic{emb: c.Embedder, model: c.Model, threshold: c.Threshold, capacity: c.Capacity,
		ttl: c.TTL, now: c.Now, guard: !c.DisableGuard, maxScopes: c.MaxScopes,
		scopes: map[string]*list.Element{}, lru: list.New()}
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
	g := GuardKey(query)
	m := Match{Vec: v, Similarity: -1}
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	el := s.scopes[scope]
	if el == nil {
		return m, nil
	}
	s.lru.MoveToFront(el)
	sc := el.Value.(*semScope)
	best := -1
	for i := range sc.entries {
		e := &sc.entries[i]
		if now.After(e.expires) || len(e.vec) != len(v) || (s.guard && e.guard != g) {
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
	e := semEntry{vec: vec, guard: GuardKey(query), query: query, resp: resp, expires: s.now().Add(s.ttl)}
	s.mu.Lock()
	defer s.mu.Unlock()
	var sc *semScope
	if el := s.scopes[scope]; el != nil {
		s.lru.MoveToFront(el)
		sc = el.Value.(*semScope)
	} else {
		sc = &semScope{key: scope}
		s.scopes[scope] = s.lru.PushFront(sc)
		for s.lru.Len() > s.maxScopes {
			old := s.lru.Back()
			s.lru.Remove(old)
			delete(s.scopes, old.Value.(*semScope).key)
		}
	}
	if len(sc.entries) < s.capacity {
		sc.entries = append(sc.entries, e)
		return
	}
	sc.entries[sc.next] = e
	sc.next = (sc.next + 1) % s.capacity
}

// Scopes returns the number of live scopes.
func (s *Semantic) Scopes() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lru.Len()
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

// dot is the inner product, unrolled with four independent accumulators so
// the adds pipeline; b is resliced so the compiler drops bounds checks.
func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}
