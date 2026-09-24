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
// on an open breaker, exactly one caller is let through as a probe; probe is
// true for that caller only. The caller must report the outcome with the same
// probe flag.
func (b *Breaker) Allow() (ok, probe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.open {
		return true, false
	}
	if b.probing || b.now().Sub(b.openedAt) < b.cfg.Cooldown {
		return false, false
	}
	b.probing = true
	return true, true
}

// Success records a successful call. While the breaker is open, only the
// probe's result counts: a request sent before the breaker opened says
// nothing about the provider's health now.
func (b *Breaker) Success(probe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.open && !probe {
		return
	}
	b.failures, b.open, b.probing = 0, false, false
}

// Failure records a failed call; it opens the breaker at the threshold, and
// re-opens it when the probe fails. Stale failures while open are ignored.
func (b *Breaker) Failure(probe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch {
	case b.open && probe:
		b.probing, b.openedAt = false, b.now()
	case b.open:
		// stale: sent before the breaker opened
	default:
		b.failures++
		if b.failures >= b.cfg.Failures {
			b.open, b.openedAt = true, b.now()
		}
	}
}

// Release ends a probe that produced no verdict about upstream health
// (client cancelled, or a client-side request error), so another request can
// probe. Only the probe itself can release the slot.
func (b *Breaker) Release(probe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if probe {
		b.probing = false
	}
}

// Open reports whether the breaker is open (including half-open).
func (b *Breaker) Open() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.open
}
