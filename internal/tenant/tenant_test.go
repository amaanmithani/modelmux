package tenant

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func mgr(t *testing.T, now *time.Time) *Manager {
	t.Helper()
	m, err := New(Config{
		Tenants: []TenantConfig{
			{Name: "acme", KeySHA256: HashKey("sk-acme"), MonthlyTokens: 100, RPM: 60, Models: []string{"fast"}},
			{Name: "free", KeySHA256: strings.ToUpper(HashKey("sk-free"))},
		},
		Public: PublicConfig{Enabled: true, RPMPerIP: 6, DailyTokensPerIP: 50, Models: []string{"fast"}},
	}, func() time.Time { return *now })
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestNewValidation(t *testing.T) {
	bad := []TenantConfig{
		{Name: "a", KeySHA256: "short"},
		{Name: "a", KeySHA256: strings.Repeat("z", 64)},
		{Name: "public", KeySHA256: HashKey("k")},
		{Name: "", KeySHA256: HashKey("k")},
	}
	for _, tc := range bad {
		if _, err := New(Config{Tenants: []TenantConfig{tc}}, nil); err == nil {
			t.Errorf("accepted %+v", tc)
		}
	}
	dup := []TenantConfig{{Name: "a", KeySHA256: HashKey("k")}, {Name: "b", KeySHA256: HashKey("k")}}
	if _, err := New(Config{Tenants: dup}, nil); err == nil {
		t.Fatal("duplicate key accepted")
	}
}

func TestAuthenticate(t *testing.T) {
	now := time.Unix(0, 0)
	m := mgr(t, &now)
	p, err := m.Authenticate("Bearer sk-acme", "1.1.1.1")
	if err != nil || p.Tenant != "acme" || !p.AllowsModel("fast") || p.AllowsModel("smart") {
		t.Fatalf("%+v %v", p, err)
	}
	if p, err := m.Authenticate("Bearer sk-free", ""); err != nil || !p.AllowsModel("anything") {
		t.Fatal("uppercase hash in config should work; no model list = all")
	}
	if _, err := m.Authenticate("Bearer wrong", "1.1.1.1"); !errors.Is(err, ErrUnauthorized) {
		t.Fatal("wrong key must be rejected, not downgraded to public")
	}
	pub, err := m.Authenticate("", "9.9.9.9")
	if err != nil || pub.Tenant != "public" || pub.key != "ip:9.9.9.9" {
		t.Fatalf("public: %+v %v", pub, err)
	}
	if p, err := m.Authenticate("Bearer public", "9.9.9.9"); err != nil || p.Tenant != "public" {
		t.Fatal("placeholder key should select public access")
	}
	custom, _ := New(Config{Public: PublicConfig{Enabled: true, Key: "demo"}}, nil)
	if p, err := custom.Authenticate("Bearer demo", "1"); err != nil || p.Tenant != "public" {
		t.Fatal("custom placeholder key")
	}
	closed, _ := New(Config{}, nil)
	if _, err := closed.Authenticate("Bearer public", "x"); !errors.Is(err, ErrUnauthorized) {
		t.Fatal("placeholder key must not work when public access is off")
	}
	if _, err := closed.Authenticate("", "x"); !errors.Is(err, ErrUnauthorized) {
		t.Fatal("public disabled should reject anonymous")
	}
}

func TestRateLimitTokenBucket(t *testing.T) {
	now := time.Unix(0, 0)
	m := mgr(t, &now)
	p, _ := m.Authenticate("", "1.2.3.4") // 6 rpm => burst 1, 1 token per 10s
	if err := p.Admit(0); err != nil {
		t.Fatal(err)
	}
	err := p.Admit(0)
	var le *LimitError
	if !errors.As(err, &le) || le.Kind != "rate_limit" || le.RetryAfter <= 0 || le.RetryAfter > 10*time.Second {
		t.Fatalf("second request should be limited: %v", err)
	}
	other, _ := m.Authenticate("", "5.6.7.8")
	if other.Admit(0) != nil {
		t.Fatal("IPs must be limited independently")
	}
	now = now.Add(10 * time.Second)
	if p.Admit(0) != nil {
		t.Fatal("bucket should refill")
	}
}

func TestBudgetDailyWindow(t *testing.T) {
	now := time.Date(2026, 9, 24, 23, 0, 0, 0, time.UTC)
	m := mgr(t, &now)
	p, _ := m.Authenticate("", "1.2.3.4")
	p.Settle(0, 60)
	if p.Used() != 60 {
		t.Fatal("used")
	}
	now = now.Add(time.Minute)
	err := p.Admit(0)
	var le *LimitError
	if !errors.As(err, &le) || le.Kind != "budget" || le.RetryAfter != 59*time.Minute {
		t.Fatalf("budget: %v", err)
	}
	now = now.Add(time.Hour) // next UTC day
	if err := p.Admit(0); err != nil {
		t.Fatalf("new day should reset budget: %v", err)
	}
	p.Settle(0, 0) // no-op
}

func TestBudgetMonthlyAndUnlimited(t *testing.T) {
	now := time.Date(2026, 9, 30, 12, 0, 0, 0, time.UTC)
	m := mgr(t, &now)
	p, _ := m.Authenticate("Bearer sk-acme", "")
	p.Settle(0, 100)
	var le *LimitError
	if err := p.Admit(0); !errors.As(err, &le) || !strings.Contains(le.Error(), "month") {
		t.Fatalf("monthly: %v", err)
	}
	if le.RetryAfter != 12*time.Hour {
		t.Fatalf("retry after %s", le.RetryAfter)
	}
	now = now.Add(24 * time.Hour)
	if p.Admit(0) != nil {
		t.Fatal("new month resets")
	}
	free, _ := m.Authenticate("Bearer sk-free", "")
	free.Settle(0, 1<<40)
	for i := 0; i < 1000; i++ {
		if free.Admit(0) != nil {
			t.Fatal("no rpm/budget means unlimited")
		}
	}
}

func TestSweepDropsIdleIPState(t *testing.T) {
	now := time.Unix(0, 0)
	m := mgr(t, &now)
	p, _ := m.Authenticate("", "1.1.1.1")
	_ = p.Admit(0)
	p.Settle(0, 1)
	now = now.Add(72 * time.Hour)
	m.mu.Lock()
	m.sweep(now)
	nb, nbud := len(m.buckets), len(m.budgets)
	m.mu.Unlock()
	if nb != 0 || nbud != 0 {
		t.Fatalf("sweep left %d buckets %d budgets", nb, nbud)
	}
	q, _ := m.Authenticate("", "2.2.2.2")
	for i := 0; i < 1100; i++ { // crosses the 1024-op sweep trigger
		_ = q.Admit(0)
	}
}

func TestReservationBlocksOvershootAndSettles(t *testing.T) {
	now := time.Unix(0, 0)
	m := mgr(t, &now) // acme: 100 tokens/month
	p, _ := m.Authenticate("Bearer sk-acme", "")
	if err := p.Admit(80); err != nil {
		t.Fatal(err)
	}
	var le *LimitError
	if err := p.Admit(30); !errors.As(err, &le) || le.Kind != "budget" {
		t.Fatalf("80 reserved + 30 > 100 must be rejected: %v", err)
	}
	p.Settle(80, 10) // the first request actually used 10
	if err := p.Admit(30); err != nil {
		t.Fatalf("after settling, 10 used + 30 fits: %v", err)
	}
	if p.Used() != 10 {
		t.Fatalf("used %d", p.Used())
	}
}

func TestPublicCapsAndGlobalLimits(t *testing.T) {
	now := time.Unix(0, 0)
	m, _ := New(Config{Public: PublicConfig{Enabled: true, GlobalRPM: 6, GlobalDailyTokens: 100}}, func() time.Time { return now })
	a, _ := m.Authenticate("", "1.1.1.1")
	if !a.Public() || a.MaxOutputTokens() != 512 || a.MaxPromptChars() != 16000 {
		t.Fatalf("defaults: %+v", a)
	}
	if err := a.Admit(0); err != nil {
		t.Fatal(err)
	}
	b, _ := m.Authenticate("", "2.2.2.2")
	var le *LimitError
	if err := b.Admit(0); !errors.As(err, &le) || !strings.Contains(le.Error(), "demo-wide") {
		t.Fatalf("global rpm should bind across IPs: %v", err)
	}
	now = now.Add(time.Minute)
	if err := b.Admit(150); !errors.As(err, &le) || le.Kind != "budget" {
		t.Fatalf("global daily budget: %v", err)
	}
	// A rejected admit must leave no trace: b's own bucket wasn't consumed.
	if err := b.Admit(50); err != nil {
		t.Fatalf("within budget: %v", err)
	}
	t2, _ := New(Config{Tenants: []TenantConfig{{Name: "x", KeySHA256: HashKey("k"), MaxOutputTokens: 99}}}, nil)
	x, _ := t2.Authenticate("Bearer k", "")
	if x.Public() || x.MaxOutputTokens() != 99 || x.MaxPromptChars() != 0 {
		t.Fatal("tenant caps")
	}
}

func TestIPv6GroupedBy64(t *testing.T) {
	now := time.Unix(0, 0)
	m, _ := New(Config{Public: PublicConfig{Enabled: true, RPMPerIP: 1}}, func() time.Time { return now })
	a, _ := m.Authenticate("", "2001:db8:1:2::1")
	b, _ := m.Authenticate("", "2001:db8:1:2:ffff::9")
	if a.Admit(0) != nil || b.Admit(0) == nil {
		t.Fatal("addresses in one /64 must share a limit")
	}
	c, _ := m.Authenticate("", "2001:db8:1:3::1")
	if c.Admit(0) != nil {
		t.Fatal("a different /64 is a different client")
	}
	if ipKey("::ffff:1.2.3.4") != "1.2.3.4" || ipKey("garbage") != "garbage" {
		t.Fatal("ipKey")
	}
}

func TestReservationCarriesAcrossPeriodBoundary(t *testing.T) {
	now := time.Date(2026, 9, 24, 23, 59, 0, 0, time.UTC)
	m, _ := New(Config{Public: PublicConfig{Enabled: true, DailyTokensPerIP: 100}}, func() time.Time { return now })
	p, _ := m.Authenticate("", "1.1.1.1")
	_ = p.Admit(60)
	now = now.Add(2 * time.Minute) // next day, request still in flight
	p.Settle(60, 5)
	if err := p.Admit(95); err != nil {
		t.Fatalf("reservation released across the boundary: %v", err)
	}
}
