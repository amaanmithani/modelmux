package config

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/tenant"
)

func build(t *testing.T, yaml string, env map[string]string) (*Built, error) {
	t.Helper()
	c, err := Load(strings.NewReader(yaml))
	if err != nil {
		return nil, err
	}
	var out bytes.Buffer
	return c.Build(func(k string) string { return env[k] }, nil, &out)
}

func TestLoadDefaultsAndUnknownFields(t *testing.T) {
	c, err := Load(strings.NewReader("providers: []"))
	if err != nil || c.Listen != ":8080" {
		t.Fatalf("%+v %v", c, err)
	}
	if _, err := Load(strings.NewReader("provders: []")); err == nil {
		t.Fatal("typo'd field must be rejected")
	}
	if _, err := LoadFile("/does/not/exist"); err == nil {
		t.Fatal("missing file")
	}
}

func TestExampleConfigsParseAndBuild(t *testing.T) {
	matches, _ := filepath.Glob("../../configs/*.yaml")
	if len(matches) == 0 {
		t.Fatal("no example configs found")
	}
	env := map[string]string{"GROQ_API_KEY": "g", "GEMINI_API_KEY": "m", "ANTHROPIC_API_KEY": "a",
		"OPENAI_API_KEY": "o", "USAGE_WEBHOOK_URL": "http://127.0.0.1:1/events"}
	for _, m := range matches {
		c, err := LoadFile(m)
		if err != nil {
			t.Fatalf("%s: %v", m, err)
		}
		b, err := c.Build(func(k string) string { return env[k] }, nil, os.Stdout)
		if err != nil {
			t.Fatalf("%s: %v", m, err)
		}
		_ = b.Sink.Close()
	}
}

const full = `
listen: ":9999"
providers:
  - {name: groq, type: openai, base_url: "https://api.groq.com/openai/v1", api_key_env: GROQ_API_KEY, optional: true}
  - {name: ollama, type: openai, base_url: "http://localhost:11434/v1", stream_usage: false}
  - {name: claude, type: anthropic, api_key_env: ANTHROPIC_API_KEY}
  - {name: stub, type: stub}
routes:
  - {alias: fast, targets: [groq/llama-3.1-8b-instant, stub/x]}
  - {alias: groq-only, targets: [groq/llama-3.1-8b-instant]}
  - {alias: smart, targets: [claude/claude-sonnet-5, ollama/llama3.1:8b]}
cache:
  exact: {enabled: true, capacity: 10, ttl: 1m}
  semantic: {enabled: true, embedder: stub, threshold: 0.9}
public: {enabled: true, rpm_per_ip: 5, models: [fast]}
usage: {sink: stdout}
`

func TestBuildSkipsOptionalProviders(t *testing.T) {
	b, err := build(t, full, map[string]string{"ANTHROPIC_API_KEY": "k"})
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Skipped) != 1 || b.Skipped[0] != "groq" {
		t.Fatalf("skipped %v", b.Skipped)
	}
	if got := strings.Join(b.Router.Aliases(), ","); got != "fast,smart" {
		t.Fatalf("aliases %s (groq-only should be dropped)", got)
	}
	if len(b.Warnings) != 1 || b.Listen != ":9999" {
		t.Fatalf("warnings %v", b.Warnings)
	}
	ts, _ := b.Router.Resolve("fast")
	if len(ts) != 1 || ts[0].Provider != "stub" {
		t.Fatalf("fast chain %v", ts)
	}
	srv := httptest.NewServer(b.Handler)
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"fast","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("serve: %v %v", resp, err)
	}
	resp.Body.Close()
}

func TestBuildErrors(t *testing.T) {
	cases := map[string]string{
		"missing required key": `providers: [{name: c, type: anthropic, api_key_env: NOPE}]`,
		"openai no base_url":   `providers: [{name: o, type: openai}]`,
		"unknown type":         `providers: [{name: o, type: cohere}]`,
		"bad target":           "providers: [{name: s, type: stub}]\nroutes: [{alias: a, targets: [nomodel]}]",
		"dup alias":            "providers: [{name: s, type: stub}]\nroutes: [{alias: a, targets: [s/x]}, {alias: a, targets: [s/y]}]",
		"unknown provider":     "providers: [{name: s, type: stub}]\nroutes: [{alias: a, targets: [zz/x]}]",
		"bad tenant":           "tenants: [{name: t, key_sha256: short}]",
		"bad embedder":         "cache: {semantic: {enabled: true, embedder: none, threshold: 0.9}}",
		"non-embedder":         "providers: [{name: c, type: anthropic}]\ncache: {semantic: {enabled: true, embedder: c, threshold: 0.9}}",
		"bad threshold":        "providers: [{name: s, type: stub}]\ncache: {semantic: {enabled: true, embedder: s, threshold: 1.5}}",
		"webhook no url":       "usage: {sink: webhook, webhook_url_env: NOPE}",
		"bad sink":             "usage: {sink: kafka}",
	}
	for name, y := range cases {
		if _, err := build(t, y, nil); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestBuildWebhookSinkAndTenants(t *testing.T) {
	y := "usage: {sink: webhook, webhook_url_env: U}\ntenants: [{name: t, key_sha256: " + tenant.HashKey("k") + "}]\n" +
		"providers: [{name: s, type: stub, reply: yo}]\nroutes: [{alias: a, targets: [s/x]}]"
	b, err := build(t, y, map[string]string{"U": "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Sink.Close()
	resp, _, err := b.Router.Chat(context.Background(), &api.ChatRequest{Model: "a",
		Messages: []api.Message{{Role: "user", Content: api.TextContent("x")}}})
	if err != nil || resp.Choices[0].Message.Content.Text() != "yo" {
		t.Fatal("stub reply")
	}
}
