// Package router resolves model aliases to ordered fallback chains of
// (provider, model) targets and executes requests against them.
package router

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/provider"
)

// Target is one (provider, upstream model) pair.
type Target struct {
	Provider string
	Model    string
}

func (t Target) String() string { return t.Provider + "/" + t.Model }

// ParseTarget parses "provider/model". The model part may itself contain
// slashes (e.g. "groq/meta-llama/llama-4").
func ParseTarget(s string) (Target, error) {
	p, m, ok := strings.Cut(s, "/")
	if !ok || p == "" || m == "" {
		return Target{}, fmt.Errorf("target %q: want provider/model", s)
	}
	return Target{Provider: p, Model: m}, nil
}

// ErrUnknownModel is returned for a model that is neither an alias nor a
// provider/model pair naming a configured provider.
var ErrUnknownModel = errors.New("unknown model")

// ErrNoHealthyTarget is returned when every target's breaker is open.
var ErrNoHealthyTarget = errors.New("no healthy upstream: all circuit breakers open")

// Observer receives per-attempt events (for metrics). All methods must be cheap.
type Observer interface {
	Attempt(provider string, outcome string, firstByte time.Duration)
	Fallback(alias string)
	BreakerState(provider string, open bool)
}

type nopObserver struct{}

func (nopObserver) Attempt(string, string, time.Duration) {}
func (nopObserver) Fallback(string)                       {}
func (nopObserver) BreakerState(string, bool)             {}

// Options tunes the router.
type Options struct {
	// Timeout bounds a whole non-streaming attempt.
	Timeout time.Duration
	// FirstByteTimeout bounds how long a streaming attempt may take to
	// produce its first chunk before we fall back.
	FirstByteTimeout time.Duration
	Breaker          BreakerConfig
	Observer         Observer
	Now              func() time.Time
}

// Router routes requests.
type Router struct {
	providers map[string]provider.Provider
	routes    map[string][]Target
	breakers  map[string]*Breaker
	opt       Options
}

// New builds a Router. Every route target must name a known provider.
func New(providers []provider.Provider, routes map[string][]Target, opt Options) (*Router, error) {
	if opt.Timeout <= 0 {
		opt.Timeout = 120 * time.Second
	}
	if opt.FirstByteTimeout <= 0 {
		opt.FirstByteTimeout = 20 * time.Second
	}
	if opt.Observer == nil {
		opt.Observer = nopObserver{}
	}
	if opt.Now == nil {
		opt.Now = time.Now
	}
	r := &Router{providers: map[string]provider.Provider{}, routes: routes, breakers: map[string]*Breaker{}, opt: opt}
	for _, p := range providers {
		if _, dup := r.providers[p.Name()]; dup {
			return nil, fmt.Errorf("duplicate provider %q", p.Name())
		}
		r.providers[p.Name()] = p
		r.breakers[p.Name()] = NewBreaker(opt.Breaker, opt.Now)
	}
	for alias, ts := range routes {
		if len(ts) == 0 {
			return nil, fmt.Errorf("route %q has no targets", alias)
		}
		for _, t := range ts {
			if _, ok := r.providers[t.Provider]; !ok {
				return nil, fmt.Errorf("route %q: unknown provider %q", alias, t.Provider)
			}
		}
	}
	return r, nil
}

// Resolve returns the chain for a model name: an alias, or a direct
// provider/model reference.
func (r *Router) Resolve(model string) ([]Target, error) {
	if ts, ok := r.routes[model]; ok {
		return ts, nil
	}
	if t, err := ParseTarget(model); err == nil {
		if _, ok := r.providers[t.Provider]; ok {
			return []Target{t}, nil
		}
	}
	return nil, fmt.Errorf("%w: %q", ErrUnknownModel, model)
}

// Aliases returns configured alias names, sorted.
func (r *Router) Aliases() []string {
	out := make([]string, 0, len(r.routes))
	for a := range r.routes {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

type simKey struct{}

// WithSimulatedOutage marks the first n targets of the chain as failed for
// this request only: they are not called, don't touch breakers, and count as
// fallbacks. Used by the public demo to show fallback on demand.
func WithSimulatedOutage(ctx context.Context, n int) context.Context {
	return context.WithValue(ctx, simKey{}, n)
}

func simulated(ctx context.Context, i int) bool {
	n, _ := ctx.Value(simKey{}).(int)
	return i < n
}

// Routes returns every alias with its chain.
func (r *Router) Routes() map[string][]Target {
	out := make(map[string][]Target, len(r.routes))
	for a, ts := range r.routes {
		out[a] = append([]Target(nil), ts...)
	}
	return out
}

// Result describes how a request was served.
type Result struct {
	Target    Target
	Fallbacks int // number of targets that failed before the one that served
}

// Chat runs a non-streaming request down the chain.
func (r *Router) Chat(ctx context.Context, req *api.ChatRequest) (*api.ChatResponse, Result, error) {
	chain, err := r.Resolve(req.Model)
	if err != nil {
		return nil, Result{}, err
	}
	var lastErr error
	fallbacks := 0
	for i, t := range chain {
		if simulated(ctx, i) {
			lastErr = &provider.UpstreamError{Provider: t.Provider, Status: 503, Message: "simulated outage"}
			fallbacks++
			continue
		}
		b := r.breakers[t.Provider]
		if !b.Allow() {
			continue
		}
		up := req.Clone()
		up.Model = t.Model
		start := r.opt.Now()
		actx, cancel := context.WithTimeout(ctx, r.opt.Timeout)
		resp, err := r.providers[t.Provider].Chat(actx, up)
		cancel()
		if err == nil {
			r.record(t.Provider, b, nil, r.opt.Now().Sub(start))
			return resp, Result{Target: t, Fallbacks: fallbacks}, nil
		}
		err = normalizeCtxErr(ctx, t.Provider, err)
		r.record(t.Provider, b, err, 0)
		if !provider.Fallbackable(err) {
			return nil, Result{Target: t, Fallbacks: fallbacks}, err
		}
		lastErr = err
		fallbacks++
		r.opt.Observer.Fallback(req.Model)
	}
	if lastErr == nil {
		return nil, Result{}, ErrNoHealthyTarget
	}
	return nil, Result{Fallbacks: fallbacks}, lastErr
}

// Stream runs a streaming request down the chain. A target counts as having
// failed if it errors before producing its first chunk; once a chunk has been
// produced the stream is committed to that target (bytes may already be on
// the wire to the client) and later errors surface to the caller.
func (r *Router) Stream(ctx context.Context, req *api.ChatRequest) (provider.Stream, Result, error) {
	chain, err := r.Resolve(req.Model)
	if err != nil {
		return nil, Result{}, err
	}
	var lastErr error
	fallbacks := 0
	for i, t := range chain {
		if simulated(ctx, i) {
			lastErr = &provider.UpstreamError{Provider: t.Provider, Status: 503, Message: "simulated outage"}
			fallbacks++
			continue
		}
		b := r.breakers[t.Provider]
		if !b.Allow() {
			continue
		}
		up := req.Clone()
		up.Model = t.Model
		start := r.opt.Now()
		actx, cancel := context.WithCancel(ctx)
		first, s, err := r.firstChunk(actx, t, up)
		if err == nil {
			r.record(t.Provider, b, nil, r.opt.Now().Sub(start))
			return &committedStream{first: first, s: s, cancel: cancel}, Result{Target: t, Fallbacks: fallbacks}, nil
		}
		cancel()
		err = normalizeCtxErr(ctx, t.Provider, err)
		r.record(t.Provider, b, err, 0)
		if !provider.Fallbackable(err) {
			return nil, Result{Target: t, Fallbacks: fallbacks}, err
		}
		lastErr = err
		fallbacks++
		r.opt.Observer.Fallback(req.Model)
	}
	if lastErr == nil {
		return nil, Result{}, ErrNoHealthyTarget
	}
	return nil, Result{Fallbacks: fallbacks}, lastErr
}

// firstChunk opens the stream and waits for its first chunk, bounded by
// FirstByteTimeout. The timer only covers the wait for the first chunk.
func (r *Router) firstChunk(ctx context.Context, t Target, req *api.ChatRequest) (*api.ChatChunk, provider.Stream, error) {
	type res struct {
		c   *api.ChatChunk
		s   provider.Stream
		err error
	}
	// ctx is the per-attempt context; the caller cancels it on failure, which
	// also stops this goroutine's upstream call.
	ch := make(chan res, 1)
	go func() {
		s, err := r.providers[t.Provider].Stream(ctx, req)
		if err != nil {
			ch <- res{err: err}
			return
		}
		c, err := s.Recv()
		if err != nil {
			_ = s.Close()
			if err == io.EOF {
				err = &provider.UpstreamError{Provider: t.Provider, Status: http.StatusBadGateway, Message: "empty stream"}
			}
			ch <- res{err: err}
			return
		}
		ch <- res{c: c, s: s}
	}()
	timer := time.NewTimer(r.opt.FirstByteTimeout)
	defer timer.Stop()
	select {
	case v := <-ch:
		if v.err != nil {
			return nil, nil, v.err
		}
		return v.c, v.s, nil
	case <-timer.C:
		go func() { // reap a stream that may still arrive
			if v := <-ch; v.s != nil {
				_ = v.s.Close()
			}
		}()
		return nil, nil, &provider.UpstreamError{Provider: t.Provider, Err: context.DeadlineExceeded,
			Message: "first-byte timeout"}
	case <-ctx.Done():
		go func() {
			if v := <-ch; v.s != nil {
				_ = v.s.Close()
			}
		}()
		return nil, nil, &provider.UpstreamError{Provider: t.Provider, Err: ctx.Err()}
	}
}

func (r *Router) record(name string, b *Breaker, err error, firstByte time.Duration) {
	outcome := "ok"
	switch {
	case err == nil:
		b.Success()
	case errors.Is(err, context.Canceled):
		outcome = "canceled"
		b.Release()
	case provider.Fallbackable(err):
		outcome = "error"
		b.Failure()
	default:
		// Client errors say nothing about upstream health.
		outcome = "client_error"
		b.Release()
	}
	r.opt.Observer.Attempt(name, outcome, firstByte)
	r.opt.Observer.BreakerState(name, b.Open())
}

// normalizeCtxErr distinguishes "the client went away" (not the upstream's
// fault, no fallback) from attempt timeouts (upstream's fault, fall back).
func normalizeCtxErr(parent context.Context, name string, err error) error {
	if parent.Err() != nil {
		return &provider.UpstreamError{Provider: name, Err: context.Canceled}
	}
	return err
}

// BreakerOpen reports whether a provider's breaker is currently open.
func (r *Router) BreakerOpen(name string) bool {
	b, ok := r.breakers[name]
	return ok && b.Open()
}

type committedStream struct {
	first  *api.ChatChunk
	s      provider.Stream
	cancel context.CancelFunc
	once   sync.Once
}

func (c *committedStream) Recv() (*api.ChatChunk, error) {
	if c.first != nil {
		f := c.first
		c.first = nil
		return f, nil
	}
	return c.s.Recv()
}

func (c *committedStream) Close() error {
	err := c.s.Close()
	c.once.Do(c.cancel)
	return err
}
