// Command modelmux runs the ModelMux LLM gateway.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/amaanmithani/modelmux/internal/config"
)

func main() {
	path := flag.String("config", envOr("MODELMUX_CONFIG", "config.yaml"), "path to YAML config")
	flag.Parse()
	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, *path, os.Getenv, logger, nil); err != nil {
		logger.Error("fatal", "err", err)
		os.Exit(1)
	}
}

// run serves until ctx is cancelled. ready, if non-nil, receives the bound
// address once listening (used by tests).
func run(ctx context.Context, path string, getenv func(string) string, logger *slog.Logger, ready chan<- string) error {
	cfg, err := config.LoadFile(path)
	if err != nil {
		return err
	}
	b, err := cfg.Build(getenv, logger, os.Stdout)
	if err != nil {
		return err
	}
	listen := b.Listen
	if p := getenv("PORT"); p != "" { // PaaS convention
		listen = ":" + p
	}
	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	// ReadTimeout bounds reading the whole request (a slow-trickled body can't
	// hold a connection); there is no WriteTimeout because responses stream.
	// Stream lifetime is bounded by the router's stream timeout instead.
	srv := &http.Server{Handler: b.Handler, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second,
		IdleTimeout: 120 * time.Second}
	errc := make(chan error, 1)
	go func() {
		logger.Info("modelmux listening", "addr", ln.Addr().String(), "routes", b.Router.Aliases(), "skipped", b.Skipped)
		errc <- srv.Serve(ln)
	}()
	if ready != nil {
		ready <- ln.Addr().String()
	}
	select {
	case err := <-errc:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
		logger.Info("shutting down")
		sctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		_ = srv.Shutdown(sctx)
	}
	return b.Sink.Close()
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
