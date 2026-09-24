// Package config loads ModelMux's YAML configuration and builds the runtime
// components from it.
package config

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/amaanmithani/modelmux/internal/cache"
	"github.com/amaanmithani/modelmux/internal/provider"
	"github.com/amaanmithani/modelmux/internal/router"
	"github.com/amaanmithani/modelmux/internal/server"
	"github.com/amaanmithani/modelmux/internal/tenant"
	"github.com/amaanmithani/modelmux/internal/usage"
	"gopkg.in/yaml.v3"
)

// Provider configures one upstream.
type Provider struct {
	Name      string            `yaml:"name"`
	Type      string            `yaml:"type"` // openai | anthropic | stub
	BaseURL   string            `yaml:"base_url"`
	APIKeyEnv string            `yaml:"api_key_env"`
	Headers   map[string]string `yaml:"headers"`
	// StreamUsage requests stream_options.include_usage upstream (openai type).
	StreamUsage *bool `yaml:"stream_usage"`
	// Optional: skip this provider (with a warning) when its key env is unset,
	// instead of failing startup. Lets one config serve partial deployments.
	Optional bool `yaml:"optional"`
	// Stub only.
	Reply   string        `yaml:"reply"`
	Latency time.Duration `yaml:"latency"`
}

// Route maps an alias to an ordered target list ("provider/model").
type Route struct {
	Alias   string   `yaml:"alias"`
	Targets []string `yaml:"targets"`
}

// Config is the whole file.
type Config struct {
	Listen    string     `yaml:"listen"`
	Providers []Provider `yaml:"providers"`
	Routes    []Route    `yaml:"routes"`
	Timeouts  struct {
		Request   time.Duration `yaml:"request"`
		FirstByte time.Duration `yaml:"first_byte"`
	} `yaml:"timeouts"`
	Breaker struct {
		Failures int           `yaml:"failures"`
		Cooldown time.Duration `yaml:"cooldown"`
	} `yaml:"breaker"`
	Cache struct {
		Exact struct {
			Enabled  bool          `yaml:"enabled"`
			Capacity int           `yaml:"capacity"`
			TTL      time.Duration `yaml:"ttl"`
		} `yaml:"exact"`
		Semantic struct {
			Enabled   bool          `yaml:"enabled"`
			Embedder  string        `yaml:"embedder"` // a provider name (openai or stub type)
			Model     string        `yaml:"model"`
			Threshold float32       `yaml:"threshold"`
			Capacity  int           `yaml:"capacity_per_scope"`
			TTL       time.Duration `yaml:"ttl"`
		} `yaml:"semantic"`
	} `yaml:"cache"`
	Tenants          []tenant.TenantConfig `yaml:"tenants"`
	Public           tenant.PublicConfig   `yaml:"public"`
	TrustedProxyHops int                   `yaml:"trusted_proxy_hops"`
	Demo             struct {
		AllowSimulation bool `yaml:"allow_simulation"`
	} `yaml:"demo"`
	Usage struct {
		Sink          string `yaml:"sink"` // none | stdout | webhook
		WebhookURLEnv string `yaml:"webhook_url_env"`
	} `yaml:"usage"`
}

// Load parses a YAML config. Unknown fields are errors, to catch typos.
func Load(r io.Reader) (*Config, error) {
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	var c Config
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if c.Listen == "" {
		c.Listen = ":8080"
	}
	return &c, nil
}

// LoadFile loads a config from disk.
func LoadFile(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f)
}

// Built holds everything constructed from a Config.
type Built struct {
	Server   *server.Server
	Router   *router.Router
	Metrics  *server.Metrics
	Sink     usage.Sink
	Skipped  []string // optional providers skipped for missing keys
	Handler  http.Handler
	Listen   string
	Warnings []string
}

// Build constructs the runtime. getenv resolves api_key_env names (os.Getenv
// in production).
func (c *Config) Build(getenv func(string) string, logger *slog.Logger, stdout io.Writer) (*Built, error) {
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	b := &Built{Listen: c.Listen, Metrics: server.NewMetrics()}
	var provs []provider.Provider
	byName := map[string]provider.Provider{}
	client := &http.Client{Transport: &http.Transport{
		Proxy: http.ProxyFromEnvironment, MaxIdleConns: 256, MaxIdleConnsPerHost: 128,
		IdleConnTimeout: 90 * time.Second, ForceAttemptHTTP2: true, ResponseHeaderTimeout: 0,
	}}
	skipped := map[string]bool{}
	for _, pc := range c.Providers {
		key := ""
		if pc.APIKeyEnv != "" {
			key = getenv(pc.APIKeyEnv)
			if key == "" {
				if pc.Optional {
					skipped[pc.Name] = true
					b.Skipped = append(b.Skipped, pc.Name)
					logger.Warn("skipping provider: key env unset", "provider", pc.Name, "env", pc.APIKeyEnv)
					continue
				}
				return nil, fmt.Errorf("provider %q: env %s is empty", pc.Name, pc.APIKeyEnv)
			}
		}
		var p provider.Provider
		switch pc.Type {
		case "openai":
			if pc.BaseURL == "" {
				return nil, fmt.Errorf("provider %q: base_url required", pc.Name)
			}
			su := true
			if pc.StreamUsage != nil {
				su = *pc.StreamUsage
			}
			p = provider.NewOpenAI(provider.OpenAIConfig{Name: pc.Name, BaseURL: pc.BaseURL, APIKey: key,
				Headers: pc.Headers, Client: client, StreamUsage: su})
		case "anthropic":
			p = provider.NewAnthropic(provider.AnthropicConfig{Name: pc.Name, BaseURL: pc.BaseURL, APIKey: key, Client: client})
		case "stub":
			reply := pc.Reply
			if reply == "" {
				reply = "This is a stub reply from ModelMux."
			}
			p = &provider.Fake{ProviderName: pc.Name, Reply: reply, Latency: pc.Latency}
		default:
			return nil, fmt.Errorf("provider %q: unknown type %q", pc.Name, pc.Type)
		}
		provs = append(provs, p)
		byName[pc.Name] = p
	}

	routes := map[string][]router.Target{}
	for _, rc := range c.Routes {
		if _, dup := routes[rc.Alias]; dup {
			return nil, fmt.Errorf("duplicate route alias %q", rc.Alias)
		}
		var ts []router.Target
		for _, s := range rc.Targets {
			t, err := router.ParseTarget(s)
			if err != nil {
				return nil, fmt.Errorf("route %q: %w", rc.Alias, err)
			}
			if skipped[t.Provider] {
				continue
			}
			ts = append(ts, t)
		}
		if len(ts) == 0 {
			b.Warnings = append(b.Warnings, fmt.Sprintf("route %q dropped: all targets skipped", rc.Alias))
			logger.Warn("route dropped: no available targets", "alias", rc.Alias)
			continue
		}
		routes[rc.Alias] = ts
	}

	r, err := router.New(provs, routes, router.Options{
		Timeout: c.Timeouts.Request, FirstByteTimeout: c.Timeouts.FirstByte,
		Breaker:  router.BreakerConfig{Failures: c.Breaker.Failures, Cooldown: c.Breaker.Cooldown},
		Observer: b.Metrics,
	})
	if err != nil {
		return nil, err
	}
	b.Router = r

	tm, err := tenant.New(tenant.Config{Tenants: c.Tenants, Public: c.Public}, nil)
	if err != nil {
		return nil, err
	}

	var exact *cache.Exact
	if c.Cache.Exact.Enabled {
		exact = cache.NewExact(c.Cache.Exact.Capacity, c.Cache.Exact.TTL, nil)
	}
	var sem *cache.Semantic
	if s := c.Cache.Semantic; s.Enabled {
		ep, ok := byName[s.Embedder]
		if !ok {
			return nil, fmt.Errorf("semantic cache: embedder %q is not a configured provider", s.Embedder)
		}
		emb, ok := ep.(provider.Embedder)
		if !ok {
			return nil, fmt.Errorf("semantic cache: provider %q cannot embed", s.Embedder)
		}
		if s.Threshold <= 0 || s.Threshold > 1 {
			return nil, fmt.Errorf("semantic cache: threshold must be in (0,1]")
		}
		sem = cache.NewSemantic(cache.SemanticConfig{Embedder: emb, Model: s.Model, Threshold: s.Threshold,
			Capacity: s.Capacity, TTL: s.TTL})
	}

	switch c.Usage.Sink {
	case "", "none":
		b.Sink = usage.Nop{}
	case "stdout":
		b.Sink = usage.NewJSONL(stdout)
	case "webhook":
		u := getenv(c.Usage.WebhookURLEnv)
		if u == "" {
			return nil, fmt.Errorf("usage webhook: env %q is empty", c.Usage.WebhookURLEnv)
		}
		b.Sink = usage.NewWebhook(usage.WebhookConfig{URL: u})
	default:
		return nil, fmt.Errorf("usage: unknown sink %q", c.Usage.Sink)
	}

	b.Server = server.New(server.Config{Router: r, Tenants: tm, Exact: exact, Semantic: sem, Sink: b.Sink,
		Metrics: b.Metrics, Logger: logger, TrustedProxyHops: c.TrustedProxyHops, AllowSimulation: c.Demo.AllowSimulation})
	b.Handler = b.Server.Handler()
	return b, nil
}
