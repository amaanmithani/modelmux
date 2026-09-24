// Package cache implements ModelMux's response caches: an exact-match LRU and
// a semantic (embedding-similarity) cache.
package cache

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
)

// Cacheable reports whether a request is deterministic enough to cache by
// default: temperature explicitly 0. Callers may force caching per request.
func Cacheable(req *api.ChatRequest) bool {
	return req.Temperature != nil && *req.Temperature == 0
}

// Key derives the exact-cache key for a request within a scope (tenant).
// Fields that don't affect the completion (stream flags, user id) are ignored,
// so a streamed and a non-streamed request share entries.
func Key(scope string, req *api.ChatRequest) string {
	r := req.Clone()
	r.Stream, r.StreamOptions, r.User = false, nil, ""
	b, _ := json.Marshal(r)
	h := sha256.New()
	h.Write([]byte(scope))
	h.Write([]byte{0})
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}

type entry struct {
	key     string
	resp    *api.ChatResponse
	expires time.Time
}

// Exact is a thread-safe LRU cache with per-entry TTL. Get and Put are O(1).
type Exact struct {
	mu    sync.Mutex
	cap   int
	ttl   time.Duration
	now   func() time.Time
	ll    *list.List
	items map[string]*list.Element
}

// NewExact returns an LRU holding at most capacity entries for ttl each.
func NewExact(capacity int, ttl time.Duration, now func() time.Time) *Exact {
	if capacity <= 0 {
		capacity = 10_000
	}
	if ttl <= 0 {
		ttl = time.Hour
	}
	if now == nil {
		now = time.Now
	}
	return &Exact{cap: capacity, ttl: ttl, now: now, ll: list.New(), items: map[string]*list.Element{}}
}

// Get returns a copy-safe cached response (callers must not mutate it).
func (c *Exact) Get(key string) (*api.ChatResponse, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[key]
	if !ok {
		return nil, false
	}
	e := el.Value.(*entry)
	if c.now().After(e.expires) {
		c.ll.Remove(el)
		delete(c.items, key)
		return nil, false
	}
	c.ll.MoveToFront(el)
	return e.resp, true
}

// Put stores a response, evicting the least recently used entry when full.
func (c *Exact) Put(key string, resp *api.ChatResponse) {
	c.mu.Lock()
	defer c.mu.Unlock()
	exp := c.now().Add(c.ttl)
	if el, ok := c.items[key]; ok {
		e := el.Value.(*entry)
		e.resp, e.expires = resp, exp
		c.ll.MoveToFront(el)
		return
	}
	c.items[key] = c.ll.PushFront(&entry{key: key, resp: resp, expires: exp})
	for c.ll.Len() > c.cap {
		old := c.ll.Back()
		c.ll.Remove(old)
		delete(c.items, old.Value.(*entry).key)
	}
}

// Len returns the number of entries (including not-yet-evicted expired ones).
func (c *Exact) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ll.Len()
}
