package router

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/provider"
)

type obs struct {
	mu        sync.Mutex
	attempts  []string
	fallbacks int
	open      map[string]bool
}

func (o *obs) Attempt(p, outcome string, _ time.Duration) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.attempts = append(o.attempts, p+":"+outcome)
}
func (o *obs) Fallback(string) { o.mu.Lock(); o.fallbacks++; o.mu.Unlock() }
func (o *obs) BreakerState(p string, open bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.open == nil {
		o.open = map[string]bool{}
	}
	o.open[p] = open
}

func req(model string) *api.ChatRequest {
	return &api.ChatRequest{Model: model, Messages: []api.Message{{Role: "user", Content: api.TextContent("hi")}}}
}

func setup(t *testing.T, opt Options) (*Router, *provider.Fake, *provider.Fake, *obs) {
	t.Helper()
	a := &provider.Fake{ProviderName: "a", Reply: "from a"}
	b := &provider.Fake{ProviderName: "b", Reply: "from b"}
	o := &obs{}
	opt.Observer = o
	r, err := New([]provider.Provider{a, b}, map[string][]Target{
		"smart": {{"a", "a-model"}, {"b", "b-model"}},
	}, opt)
	if err != nil {
		t.Fatal(err)
	}
	return r, a, b, o
}

var (
	e503 = &provider.UpstreamError{Provider: "x", Status: 503}
	e400 = &provider.UpstreamError{Provider: "x", Status: 400}
)

func TestNewValidation(t *testing.T) {
	a := &provider.Fake{ProviderName: "a"}
	if _, err := New([]provider.Provider{a, a}, nil, Options{}); err == nil {
		t.Fatal("duplicate provider accepted")
	}
	if _, err := New([]provider.Provider{a}, map[string][]Target{"x": {}}, Options{}); err == nil {
		t.Fatal("empty route accepted")
	}
	if _, err := New([]provider.Provider{a}, map[string][]Target{"x": {{"nope", "m"}}}, Options{}); err == nil {
		t.Fatal("unknown provider accepted")
	}
}

func TestParseTargetAndResolve(t *testing.T) {
	tg, err := ParseTarget("groq/meta-llama/llama-4")
	if err != nil || tg.Provider != "groq" || tg.Model != "meta-llama/llama-4" || tg.String() != "groq/meta-llama/llama-4" {
		t.Fatalf("%+v %v", tg, err)
	}
	for _, bad := range []string{"nomodel", "/m", "p/"} {
		if _, err := ParseTarget(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	r, _, _, _ := setup(t, Options{})
	if ts, _ := r.Resolve("smart"); len(ts) != 2 {
		t.Fatal("alias")
	}
	if ts, err := r.Resolve("b/direct"); err != nil || ts[0].Model != "direct" {
		t.Fatal("direct provider/model")
	}
	if _, err := r.Resolve("zzz/m"); !errors.Is(err, ErrUnknownModel) {
		t.Fatal("unknown provider should be unknown model")
	}
	if len(r.Aliases()) != 1 || r.Aliases()[0] != "smart" {
		t.Fatal("aliases")
	}
}

func TestChatHappyPathRewritesModel(t *testing.T) {
	r, a, b, o := setup(t, Options{})
	resp, res, err := r.Chat(context.Background(), req("smart"))
	if err != nil || resp.Choices[0].Message.Content.Text() != "from a" {
		t.Fatalf("%v %v", resp, err)
	}
	if res.Target.Provider != "a" || res.Fallbacks != 0 || a.LastRequest().Model != "a-model" || b.Calls() != 0 {
		t.Fatalf("res %+v", res)
	}
	if o.attempts[0] != "a:ok" {
		t.Fatal(o.attempts)
	}
}

func TestChatFallsBackOnRetryable(t *testing.T) {
	r, a, b, o := setup(t, Options{})
	a.FailNext(e503)
	resp, res, err := r.Chat(context.Background(), req("smart"))
	if err != nil || resp.Choices[0].Message.Content.Text() != "from b" || res.Fallbacks != 1 || res.Target.Provider != "b" {
		t.Fatalf("%+v %v", res, err)
	}
	if b.LastRequest().Model != "b-model" || o.fallbacks != 1 {
		t.Fatal("fallback bookkeeping")
	}
}

func TestChatStopsOnClientError(t *testing.T) {
	r, a, b, _ := setup(t, Options{})
	a.FailNext(e400)
	_, _, err := r.Chat(context.Background(), req("smart"))
	if !errors.Is(err, e400) && err != error(e400) {
		t.Fatalf("got %v", err)
	}
	if b.Calls() != 0 {
		t.Fatal("400 must not fall back")
	}
}

func TestChatAllFailReturnsLastError(t *testing.T) {
	r, a, b, _ := setup(t, Options{})
	a.FailNext(e503)
	last := &provider.UpstreamError{Provider: "b", Status: 429}
	b.FailNext(last)
	_, res, err := r.Chat(context.Background(), req("smart"))
	if err != error(last) || res.Fallbacks != 2 {
		t.Fatalf("%v %+v", err, res)
	}
	if _, _, err := r.Chat(context.Background(), req("nope")); !errors.Is(err, ErrUnknownModel) {
		t.Fatal(err)
	}
}

func TestChatTimeoutFallsBack(t *testing.T) {
	r, a, _, _ := setup(t, Options{Timeout: 20 * time.Millisecond})
	a.Latency = time.Second
	_, res, err := r.Chat(context.Background(), req("smart"))
	if err != nil || res.Target.Provider != "b" {
		t.Fatalf("timeout should fall back: %+v %v", res, err)
	}
}

func TestChatClientCancelDoesNotFallBackOrTripBreaker(t *testing.T) {
	r, a, b, o := setup(t, Options{Breaker: BreakerConfig{Failures: 1}})
	a.Latency = time.Second
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(10 * time.Millisecond); cancel() }()
	_, _, err := r.Chat(ctx, req("smart"))
	if !errors.Is(err, context.Canceled) || b.Calls() != 0 {
		t.Fatalf("err=%v bcalls=%d", err, b.Calls())
	}
	if r.BreakerOpen("a") || o.attempts[0] != "a:canceled" {
		t.Fatalf("breaker tripped by client cancel: %v", o.attempts)
	}
}

func TestBreakerSkipsOpenProviderAndRecovers(t *testing.T) {
	now := time.Unix(0, 0)
	clock := func() time.Time { return now }
	r, a, b, o := setup(t, Options{Now: clock, Breaker: BreakerConfig{Failures: 2, Cooldown: time.Minute}})
	a.FailNext(e503, e503)
	for i := 0; i < 2; i++ {
		if _, res, _ := r.Chat(context.Background(), req("smart")); res.Target.Provider != "b" {
			t.Fatal("expected b")
		}
	}
	if !r.BreakerOpen("a") || !o.open["a"] {
		t.Fatal("breaker should be open after 2 failures")
	}
	callsA := a.Calls()
	_, res, _ := r.Chat(context.Background(), req("smart"))
	if a.Calls() != callsA || res.Fallbacks != 1 || res.Target.Provider != "b" {
		t.Fatal("open breaker should skip a (without calling it) and count the skip as a fallback")
	}
	now = now.Add(2 * time.Minute) // cooldown over: one probe
	_, res, _ = r.Chat(context.Background(), req("smart"))
	if res.Target.Provider != "a" || r.BreakerOpen("a") {
		t.Fatal("successful probe should close the breaker")
	}
	_ = b
}

func TestAllBreakersOpen(t *testing.T) {
	r, a, b, _ := setup(t, Options{Breaker: BreakerConfig{Failures: 1, Cooldown: time.Hour}})
	a.FailNext(e503)
	b.FailNext(e503)
	_, _, _ = r.Chat(context.Background(), req("smart"))
	if _, _, err := r.Chat(context.Background(), req("smart")); !errors.Is(err, ErrNoHealthyTarget) {
		t.Fatalf("chat: %v", err)
	}
	if _, _, err := r.Stream(context.Background(), req("smart")); !errors.Is(err, ErrNoHealthyTarget) {
		t.Fatalf("stream: %v", err)
	}
}

func collect(t *testing.T, s provider.Stream) string {
	t.Helper()
	defer s.Close()
	var acc api.Accumulator
	for {
		c, err := s.Recv()
		if err == io.EOF {
			return acc.Response().Choices[0].Message.Content.Text()
		}
		if err != nil {
			t.Fatal(err)
		}
		acc.Add(c)
	}
}

func TestStreamFallsBackBeforeFirstByte(t *testing.T) {
	r, a, _, _ := setup(t, Options{})
	a.FailNext(e503)
	s, res, err := r.Stream(context.Background(), req("smart"))
	if err != nil || res.Target.Provider != "b" || res.Fallbacks != 1 {
		t.Fatalf("%+v %v", res, err)
	}
	if got := collect(t, s); got != "from b" {
		t.Fatal(got)
	}
}

func TestStreamFirstByteTimeoutFallsBack(t *testing.T) {
	r, a, _, _ := setup(t, Options{FirstByteTimeout: 20 * time.Millisecond})
	a.Latency = time.Second
	s, res, err := r.Stream(context.Background(), req("smart"))
	if err != nil || res.Target.Provider != "b" {
		t.Fatalf("%+v %v", res, err)
	}
	if collect(t, s) != "from b" {
		t.Fatal("wrong stream")
	}
}

func TestStreamEmptyAndClientErrors(t *testing.T) {
	empty := &emptyProvider{}
	b := &provider.Fake{ProviderName: "b", Reply: "ok"}
	r, err := New([]provider.Provider{empty, b}, map[string][]Target{"m": {{"empty", "x"}, {"b", "y"}}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	s, res, err := r.Stream(context.Background(), req("m"))
	if err != nil || res.Target.Provider != "b" {
		t.Fatalf("empty stream should fall back: %+v %v", res, err)
	}
	s.Close()

	r2, a, b2, _ := setup(t, Options{})
	a.FailNext(e400)
	if _, _, err := r2.Stream(context.Background(), req("smart")); err != error(e400) || b2.Calls() != 0 {
		t.Fatalf("400 on stream: %v", err)
	}
	a.FailNext(e503)
	b2.FailNext(e503)
	if _, _, err := r2.Stream(context.Background(), req("smart")); err != error(e503) {
		t.Fatalf("all failed: %v", err)
	}
	if _, _, err := r2.Stream(context.Background(), req("unknown")); !errors.Is(err, ErrUnknownModel) {
		t.Fatal(err)
	}
}

func TestStreamClientCancelWhileWaiting(t *testing.T) {
	r, a, b, _ := setup(t, Options{})
	a.Latency = time.Second
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(10 * time.Millisecond); cancel() }()
	if _, _, err := r.Stream(ctx, req("smart")); !errors.Is(err, context.Canceled) || b.Calls() != 0 {
		t.Fatalf("err=%v", err)
	}
}

type emptyProvider struct{}

func (emptyProvider) Name() string { return "empty" }
func (emptyProvider) Chat(context.Context, *api.ChatRequest) (*api.ChatResponse, error) {
	return nil, errors.New("unused")
}
func (emptyProvider) Stream(context.Context, *api.ChatRequest) (provider.Stream, error) {
	return emptyStream{}, nil
}

type emptyStream struct{}

func (emptyStream) Recv() (*api.ChatChunk, error) { return nil, io.EOF }
func (emptyStream) Close() error                  { return nil }

func TestBreakerStateMachine(t *testing.T) {
	now := time.Unix(0, 0)
	b := NewBreaker(BreakerConfig{Failures: 2, Cooldown: 10 * time.Second}, func() time.Time { return now })
	allow := func() (bool, bool) { return b.Allow() }
	b.Failure(false)
	if ok, _ := allow(); b.Open() || !ok {
		t.Fatal("one failure should not open")
	}
	b.Failure(false)
	if ok, _ := allow(); !b.Open() || ok {
		t.Fatal("two failures should open")
	}
	now = now.Add(11 * time.Second)
	ok, probe := allow()
	if !ok || !probe {
		t.Fatal("cooldown elapsed: one probe allowed")
	}
	if ok, _ := allow(); ok {
		t.Fatal("only one probe at a time")
	}
	b.Release(false) // a stale, non-probe request finishing must not free the slot
	if ok, _ := allow(); ok {
		t.Fatal("stale release freed the probe slot")
	}
	b.Success(false) // stale success must not close an open breaker
	if !b.Open() {
		t.Fatal("stale success closed the breaker")
	}
	b.Failure(false) // stale failure must not push the cooldown
	b.Release(true)
	ok, probe = allow()
	if !ok || !probe {
		t.Fatal("released probe slot should be reusable")
	}
	b.Failure(true) // failed probe re-opens
	if ok, _ := allow(); !b.Open() || ok {
		t.Fatal("failed probe should re-open")
	}
	now = now.Add(11 * time.Second)
	_, probe = allow()
	b.Success(probe)
	if ok, _ := allow(); b.Open() || !ok {
		t.Fatal("successful probe should close")
	}
	d := NewBreaker(BreakerConfig{}, nil)
	if d.cfg.Failures != 5 || d.cfg.Cooldown != 30*time.Second {
		t.Fatal("defaults")
	}
}

func TestBreakerSkipCountsAsFallback(t *testing.T) {
	r, a, _, o := setup(t, Options{Breaker: BreakerConfig{Failures: 1, Cooldown: time.Hour}})
	a.FailNext(e503)
	if _, res, _ := r.Chat(context.Background(), req("smart")); res.Fallbacks != 1 {
		t.Fatal("first")
	}
	_, res, _ := r.Chat(context.Background(), req("smart"))
	if res.Fallbacks != 1 || res.Target.Provider != "b" || o.fallbacks != 2 {
		t.Fatalf("skip over open breaker must count: %+v obs=%d", res, o.fallbacks)
	}
	s, res, _ := r.Stream(context.Background(), req("smart"))
	if res.Fallbacks != 1 {
		t.Fatalf("stream skip: %+v", res)
	}
	s.Close()
}

func TestStreamTimeoutBoundsWholeStream(t *testing.T) {
	stall := &stallProvider{}
	r2, err := New([]provider.Provider{stall}, map[string][]Target{"m": {{"stall", "x"}}}, Options{StreamTimeout: 30 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	st, _, err := r2.Stream(context.Background(), req("m"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if _, err := st.Recv(); err != nil {
		t.Fatal("first chunk")
	}
	start := time.Now()
	if _, err := st.Recv(); err == nil || time.Since(start) > time.Second {
		t.Fatalf("stalled stream should end at the stream timeout: %v after %s", err, time.Since(start))
	}
}

// stallProvider sends one chunk, then blocks until its context ends.
type stallProvider struct{}

func (stallProvider) Name() string { return "stall" }
func (stallProvider) Chat(context.Context, *api.ChatRequest) (*api.ChatResponse, error) {
	return nil, errors.New("unused")
}
func (stallProvider) Stream(ctx context.Context, _ *api.ChatRequest) (provider.Stream, error) {
	return &stallStream{ctx: ctx}, nil
}

type stallStream struct {
	ctx  context.Context
	sent bool
}

func (s *stallStream) Recv() (*api.ChatChunk, error) {
	if !s.sent {
		s.sent = true
		return &api.ChatChunk{Choices: []api.ChunkChoice{{Delta: api.Delta{Content: api.Ptr("x")}}}}, nil
	}
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}
func (s *stallStream) Close() error { return nil }

func TestSimulatedOutage(t *testing.T) {
	r, a, b, _ := setup(t, Options{Breaker: BreakerConfig{Failures: 1}})
	ctx := WithSimulatedOutage(context.Background(), 1)
	_, res, err := r.Chat(ctx, req("smart"))
	if err != nil || res.Target.Provider != "b" || res.Fallbacks != 1 || a.Calls() != 0 {
		t.Fatalf("%+v %v calls=%d", res, err, a.Calls())
	}
	s, res, err := r.Stream(ctx, req("smart"))
	if err != nil || res.Target.Provider != "b" || res.Fallbacks != 1 {
		t.Fatalf("stream: %+v %v", res, err)
	}
	s.Close()
	if r.BreakerOpen("a") {
		t.Fatal("simulated outage must not trip the breaker")
	}
	if _, _, err := r.Chat(WithSimulatedOutage(context.Background(), 2), req("smart")); err == nil || b.Calls() != 2 {
		t.Fatalf("all simulated down: %v", err)
	}
	if len(r.Routes()["smart"]) != 2 {
		t.Fatal("routes")
	}
}
