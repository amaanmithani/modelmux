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
	"net/netip"
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
	// MaxOutputTokens caps max_tokens per request (0 = no cap).
	MaxOutputTokens int `yaml:"max_output_tokens"`
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
	// MaxOutputTokens caps (and, if unset, sets) max_tokens. Default 512.
	MaxOutputTokens int `yaml:"max_output_tokens"`
	// MaxPromptChars rejects larger prompts. Default 16000.
	MaxPromptChars int `yaml:"max_prompt_chars"`
	// GlobalRPM and GlobalDailyTokens bound all anonymous traffic together,
	// so many IPs can't jointly exhaust the owner's upstream quota. 0 = off.
	GlobalRPM         int   `yaml:"global_rpm"`
	GlobalDailyTokens int64 `yaml:"global_daily_tokens"`
}

// Config configures a Manager.
type Config struct {
	Tenants []TenantConfig `yaml:"tenants"`
	Public  PublicConfig   `yaml:"public"`
}

// Principal is an authenticated caller.
type Principal struct {
	Tenant    string // tenant name, or "public"
	key       string // rate-limit/budget key
	rpm       int
	budget    int64
	daily     bool
	models    map[string]bool
	maxOut    int
	maxPrompt int
	manager   *Manager
}

// Public reports whether this is an anonymous caller.
func (p *Principal) Public() bool { return p.Tenant == "public" }

// MaxOutputTokens is the per-request output cap (0 = none).
func (p *Principal) MaxOutputTokens() int { return p.maxOut }

// MaxPromptChars is the prompt-size cap (0 = none).
func (p *Principal) MaxPromptChars() int { return p.maxPrompt }

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
	period   string
	used     int64
	reserved int64 // admitted but not yet settled
	seen     time.Time
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
		maxOut, maxPrompt := m.public.MaxOutputTokens, m.public.MaxPromptChars
		if maxOut <= 0 {
			maxOut = 512
		}
		if maxPrompt <= 0 {
			maxPrompt = 16000
		}
		return &Principal{Tenant: "public", key: "ip:" + ipKey(clientIP), rpm: m.public.RPMPerIP,
			budget: m.public.DailyTokensPerIP, daily: true, models: set(m.public.Models),
			maxOut: maxOut, maxPrompt: maxPrompt, manager: m}, nil
	}
	t, ok := m.byHash[HashKey(key)]
	if !ok {
		return nil, ErrUnauthorized
	}
	return &Principal{Tenant: t.Name, key: "t:" + t.Name, rpm: t.RPM, budget: t.MonthlyTokens,
		models: set(t.Models), maxOut: t.MaxOutputTokens, manager: m}, nil
}

// ipKey groups IPv6 clients by /64: a single host typically controls a
// whole /64, so per-address limits would be trivially bypassed.
func ipKey(ip string) string {
	a, err := netip.ParseAddr(ip)
	if err != nil {
		return ip
	}
	a = a.Unmap()
	if a.Is6() {
		p, _ := a.Prefix(64)
		return p.String()
	}
	return a.String()
}

const globalKey = "public:global"

// Admit applies rate limits and reserves reserve tokens against the budget
// (the request's worst-case cost: estimated prompt plus capped output).
// Callers must Settle exactly once afterwards. For anonymous callers the
// global public limits apply as well.
func (p *Principal) Admit(reserve int64) error {
	m := p.manager
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.now()
	m.ops++
	if m.ops%1024 == 0 {
		m.sweep(now)
	}
	type check struct {
		key    string
		rpm    int
		budget int64
		daily  bool
		scope  string
	}
	checks := []check{{p.key, p.rpm, p.budget, p.daily, ""}}
	if p.Public() {
		checks = append(checks, check{globalKey, m.public.GlobalRPM, m.public.GlobalDailyTokens, true, "demo-wide "})
	}
	// Check everything before mutating anything, so a rejection leaves no trace.
	for _, c := range checks {
		if c.budget > 0 {
			b := m.budgetFor(c.key, c.daily, now)
			if b.used >= c.budget || b.used+b.reserved+reserve > c.budget {
				return &LimitError{Kind: "budget", RetryAfter: untilNextPeriod(now, c.daily),
					Msg: fmt.Sprintf("%stoken budget of %d would be exceeded this %s", c.scope, c.budget, periodName(c.daily))}
			}
		}
		if c.rpm > 0 {
			bk := m.refill(c.key, c.rpm, now)
			if bk.tokens < 1 {
				rate := float64(c.rpm) / 60
				wait := time.Duration((1 - bk.tokens) / rate * float64(time.Second))
				return &LimitError{Kind: "rate_limit", RetryAfter: wait,
					Msg: fmt.Sprintf("%srate limit of %d requests/minute exceeded", c.scope, c.rpm)}
			}
		}
	}
	for _, c := range checks {
		if c.rpm > 0 {
			m.buckets[c.key].tokens--
		}
		if c.budget > 0 {
			m.budgetFor(c.key, c.daily, now).reserved += reserve
		}
	}
	return nil
}

// refill brings a bucket up to date. Called with m.mu held.
func (m *Manager) refill(key string, rpm int, now time.Time) *bucket {
	rate := float64(rpm) / 60 // tokens per second
	burst := math.Max(1, float64(rpm)/6)
	bk := m.buckets[key]
	if bk == nil {
		bk = &bucket{tokens: burst, last: now}
		m.buckets[key] = bk
	}
	bk.tokens = math.Min(burst, bk.tokens+now.Sub(bk.last).Seconds()*rate)
	bk.last = now
	return bk
}

// Settle releases a reservation made by Admit and charges the tokens actually
// used (0 for cache hits and failures).
func (p *Principal) Settle(reserved int64, used int) {
	m := p.manager
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.now()
	keys := []struct {
		key    string
		budget int64
		daily  bool
	}{{p.key, p.budget, p.daily}}
	if p.Public() {
		keys = append(keys, struct {
			key    string
			budget int64
			daily  bool
		}{globalKey, m.public.GlobalDailyTokens, true})
	}
	for _, k := range keys {
		if k.budget <= 0 {
			continue
		}
		b := m.budgetFor(k.key, k.daily, now)
		b.reserved = max(0, b.reserved-reserved)
		if used > 0 {
			b.used += int64(used)
		}
	}
}

// Used returns tokens used in the current period.
func (p *Principal) Used() int64 {
	m := p.manager
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.budgetFor(p.key, p.daily, m.now()).used
}

func (m *Manager) budgetFor(key string, daily bool, now time.Time) *budget {
	period := periodKey(now, daily)
	b := m.budgets[key]
	if b == nil || b.period != period {
		// A reservation made in the previous period is settled against the
		// new one's reserved counter; carry it so it can be released.
		carried := int64(0)
		if b != nil {
			carried = b.reserved
		}
		b = &budget{period: period, reserved: carried}
		m.budgets[key] = b
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
		if now.Sub(b.seen) > 48*time.Hour && b.reserved == 0 && strings.HasPrefix(k, "ip:") {
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
