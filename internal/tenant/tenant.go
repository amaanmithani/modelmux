// Package tenant authenticates callers and enforces per-tenant (or, for the
// public demo, per-client-IP) rate limits and token budgets.
//
// State is in-memory: a restart resets budgets. That is a documented v1 limit.
package tenant

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

// Errors returned by Authenticate and Admit.
var (
	ErrUnauthorized = errors.New("invalid or missing API key")
	ErrModelDenied  = errors.New("model not allowed for this key")
)

// LimitError is a rate-limit or budget rejection.
type LimitError struct {
	Kind       string // "rate_limit" or "budget"
	RetryAfter time.Duration
	Msg        string
}

func (e *LimitError) Error() string { return e.Msg }

// TenantConfig is one API-key tenant.
type TenantConfig struct {
	Name          string   `yaml:"name"`
	KeySHA256     string   `yaml:"key_sha256"`
	MonthlyTokens int64    `yaml:"monthly_tokens"` // 0 = unlimited
	RPM           int      `yaml:"rpm"`            // 0 = unlimited
	Models        []string `yaml:"models"`         // empty = all
}

// PublicConfig enables unauthenticated access, limited per client IP.
type PublicConfig struct {
	Enabled          bool     `yaml:"enabled"`
	RPMPerIP         int      `yaml:"rpm_per_ip"`
	DailyTokensPerIP int64    `yaml:"daily_tokens_per_ip"`
	Models           []string `yaml:"models"`
	// Key is a placeholder API key that also selects public access, for SDKs
	// that refuse to send a request without one. Default "public".
	Key string `yaml:"key"`
}

// Config configures a Manager.
type Config struct {
	Tenants []TenantConfig `yaml:"tenants"`
	Public  PublicConfig   `yaml:"public"`
}

// Principal is an authenticated caller.
type Principal struct {
	Tenant  string // tenant name, or "public"
	key     string // rate-limit/budget key
	rpm     int
	budget  int64
	daily   bool
	models  map[string]bool
	manager *Manager
}

// AllowsModel reports whether the principal may use a model name.
func (p *Principal) AllowsModel(model string) bool {
	return len(p.models) == 0 || p.models[model]
}

// Manager holds tenants and limiter state.
type Manager struct {
	byHash map[string]TenantConfig
	public PublicConfig
	now    func() time.Time

	mu      sync.Mutex
	buckets map[string]*bucket
	budgets map[string]*budget
	ops     int
}

type bucket struct {
	tokens float64
	last   time.Time
}

type budget struct {
	period string
	used   int64
	seen   time.Time
}

// HashKey returns the hex SHA-256 of an API key, the form stored in config.
func HashKey(key string) string {
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:])
}

// New validates cfg and returns a Manager.
func New(cfg Config, now func() time.Time) (*Manager, error) {
	if now == nil {
		now = time.Now
	}
	m := &Manager{byHash: map[string]TenantConfig{}, public: cfg.Public, now: now,
		buckets: map[string]*bucket{}, budgets: map[string]*budget{}}
	for _, t := range cfg.Tenants {
		h := strings.ToLower(strings.TrimSpace(t.KeySHA256))
		if len(h) != 64 {
			return nil, fmt.Errorf("tenant %q: key_sha256 must be 64 hex chars", t.Name)
		}
		if _, err := hex.DecodeString(h); err != nil {
			return nil, fmt.Errorf("tenant %q: key_sha256: %w", t.Name, err)
		}
		if _, dup := m.byHash[h]; dup {
			return nil, fmt.Errorf("tenant %q: duplicate key", t.Name)
		}
		if t.Name == "" || t.Name == "public" {
			return nil, fmt.Errorf("tenant name %q is invalid or reserved", t.Name)
		}
		m.byHash[h] = t
	}
	return m, nil
}

func set(models []string) map[string]bool {
	if len(models) == 0 {
		return nil
	}
	out := map[string]bool{}
	for _, s := range models {
		out[s] = true
	}
	return out
}

// Authenticate resolves an Authorization header value ("Bearer <key>") to a
// principal. With no key and public access enabled, the caller is limited by
// clientIP. A key that is present but wrong is always rejected.
func (m *Manager) Authenticate(authorization, clientIP string) (*Principal, error) {
	key := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(authorization), "Bearer "))
	pubKey := m.public.Key
	if pubKey == "" {
		pubKey = "public"
	}
	if key == "" || key == pubKey {
		if !m.public.Enabled {
			return nil, ErrUnauthorized
		}
		return &Principal{Tenant: "public", key: "ip:" + clientIP, rpm: m.public.RPMPerIP,
			budget: m.public.DailyTokensPerIP, daily: true, models: set(m.public.Models), manager: m}, nil
	}
	t, ok := m.byHash[HashKey(key)]
	if !ok {
		return nil, ErrUnauthorized
	}
	return &Principal{Tenant: t.Name, key: "t:" + t.Name, rpm: t.RPM, budget: t.MonthlyTokens,
		models: set(t.Models), manager: m}, nil
}

// Admit applies the rate limit and budget check for one request.
func (p *Principal) Admit() error {
	m := p.manager
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.now()
	m.ops++
	if m.ops%1024 == 0 {
		m.sweep(now)
	}
	if p.budget > 0 {
		b := m.budgetFor(p, now)
		if b.used >= p.budget {
			return &LimitError{Kind: "budget", RetryAfter: untilNextPeriod(now, p.daily),
				Msg: fmt.Sprintf("token budget of %d exhausted for this %s", p.budget, periodName(p.daily))}
		}
	}
	if p.rpm > 0 {
		rate := float64(p.rpm) / 60 // tokens per second
		burst := math.Max(1, float64(p.rpm)/6)
		bk := m.buckets[p.key]
		if bk == nil {
			bk = &bucket{tokens: burst, last: now}
			m.buckets[p.key] = bk
		}
		bk.tokens = math.Min(burst, bk.tokens+now.Sub(bk.last).Seconds()*rate)
		bk.last = now
		if bk.tokens < 1 {
			wait := time.Duration((1 - bk.tokens) / rate * float64(time.Second))
			return &LimitError{Kind: "rate_limit", RetryAfter: wait,
				Msg: fmt.Sprintf("rate limit of %d requests/minute exceeded", p.rpm)}
		}
		bk.tokens--
	}
	return nil
}

// Charge adds used tokens to the principal's budget.
func (p *Principal) Charge(tokens int) {
	if p.budget <= 0 || tokens <= 0 {
		return
	}
	m := p.manager
	m.mu.Lock()
	defer m.mu.Unlock()
	m.budgetFor(p, m.now()).used += int64(tokens)
}

// Used returns tokens used in the current period.
func (p *Principal) Used() int64 {
	m := p.manager
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.budgetFor(p, m.now()).used
}

func (m *Manager) budgetFor(p *Principal, now time.Time) *budget {
	period := periodKey(now, p.daily)
	b := m.budgets[p.key]
	if b == nil || b.period != period {
		b = &budget{period: period}
		m.budgets[p.key] = b
	}
	b.seen = now
	return b
}

// sweep drops limiter state idle for over a day so per-IP maps can't grow
// without bound. Called with m.mu held.
func (m *Manager) sweep(now time.Time) {
	for k, b := range m.buckets {
		if now.Sub(b.last) > 24*time.Hour {
			delete(m.buckets, k)
		}
	}
	for k, b := range m.budgets {
		if now.Sub(b.seen) > 48*time.Hour && strings.HasPrefix(k, "ip:") {
			delete(m.budgets, k)
		}
	}
}

func periodKey(t time.Time, daily bool) string {
	t = t.UTC()
	if daily {
		return t.Format("2006-01-02")
	}
	return t.Format("2006-01")
}

func periodName(daily bool) string {
	if daily {
		return "day"
	}
	return "month"
}

func untilNextPeriod(t time.Time, daily bool) time.Duration {
	t = t.UTC()
	var next time.Time
	if daily {
		next = time.Date(t.Year(), t.Month(), t.Day()+1, 0, 0, 0, 0, time.UTC)
	} else {
		next = time.Date(t.Year(), t.Month()+1, 1, 0, 0, 0, 0, time.UTC)
	}
	return next.Sub(t)
}
