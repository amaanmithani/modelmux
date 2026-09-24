package main

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const testConfig = `
listen: "127.0.0.1:0"
providers:
  - {name: stub, type: stub, reply: "pong"}
routes:
  - {alias: fast, targets: [stub/any]}
public: {enabled: true}
`

func TestRunServesAndShutsDown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.yaml")
	if err := os.WriteFile(path, []byte(testConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	ready := make(chan string, 1)
	errc := make(chan error, 1)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	go func() { errc <- run(ctx, path, func(string) string { return "" }, logger, ready) }()
	var addr string
	select {
	case addr = <-ready:
	case err := <-errc:
		t.Fatal(err)
	case <-time.After(5 * time.Second):
		t.Fatal("not ready")
	}
	resp, err := http.Post("http://"+addr+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"fast","messages":[{"role":"user","content":"ping"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(b), "pong") {
		t.Fatalf("body %s", b)
	}
	cancel()
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func TestRunErrors(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	env := func(string) string { return "" }
	if err := run(context.Background(), "/nonexistent.yaml", env, logger, nil); err == nil {
		t.Fatal("missing file")
	}
	path := filepath.Join(t.TempDir(), "c.yaml")
	_ = os.WriteFile(path, []byte("providers: [{name: x, type: nope}]"), 0o600)
	if err := run(context.Background(), path, env, logger, nil); err == nil {
		t.Fatal("bad provider type")
	}
	_ = os.WriteFile(path, []byte(testConfig), 0o600)
	portEnv := func(k string) string {
		if k == "PORT" {
			return "not-a-port"
		}
		return ""
	}
	if err := run(context.Background(), path, portEnv, logger, nil); err == nil {
		t.Fatal("bad PORT should fail to listen")
	}
	if envOr("MODELMUX_DEFINITELY_UNSET", "d") != "d" {
		t.Fatal("envOr")
	}
}
