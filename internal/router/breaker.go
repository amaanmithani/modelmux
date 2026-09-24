package router

import (
	"sync"
	"time"
)

// BreakerConfig tunes a circuit breaker.
type BreakerConfig struct {
	// Failures is the number of consecutive failures that opens the breaker.
	Failures int
	// Cooldown is how long the breaker stays open before letting one trial
	// request through (half-open).
	Cooldown time.Duration
}

// Breaker is a consecutive-failure circuit breaker with a single half-open probe.
type Breaker struct {
	cfg      BreakerConfig
	now      func() time.Time
	mu       sync.Mutex
	failures int
	openedAt time.Time
	open     bool
	probing  bool
}

// NewBreaker returns a closed breaker. Zero config values get defaults (5, 30s).
func NewBreaker(cfg BreakerConfig, now func() time.Time) *Breaker {
	if cfg.Failures <= 0 {
		cfg.Failures = 5
	}
	if cfg.Cooldown <= 0 {
		cfg.Cooldown = 30 * time.Second
	}
	if now == nil {
		now = time.Now
	}
	return &Breaker{cfg: cfg, now: now}
}

// Allow reports whether a request may be sent. When the cooldown has elapsed
// on an open breaker, exactly one caller is let through as a probe.
func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.open {
		return true
	}
	if b.probing || b.now().Sub(b.openedAt) < b.cfg.Cooldown {
		return false
	}
	b.probing = true
	return true
}

// Success records a successful call and closes the breaker.
func (b *Breaker) Success() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures, b.open, b.probing = 0, false, false
}

// Failure records a failed call; it opens (or re-opens) the breaker when the
// threshold is reached or a half-open probe fails.
func (b *Breaker) Failure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures++
	if b.probing || b.failures >= b.cfg.Failures {
		b.open, b.probing, b.openedAt = true, false, b.now()
	}
}

// Release ends a half-open probe that produced no verdict about upstream
// health (client cancelled, or a client-side request error), so a later
// request can probe instead.
func (b *Breaker) Release() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.probing = false
}

// Open reports whether the breaker is open (including half-open).
func (b *Breaker) Open() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.open
}
