package server

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Metrics holds ModelMux's Prometheus collectors. It implements router.Observer.
type Metrics struct {
	Registry   *prometheus.Registry
	requests   *prometheus.CounterVec
	duration   *prometheus.HistogramVec
	attempts   *prometheus.CounterVec
	firstByte  *prometheus.HistogramVec
	fallbacks  *prometheus.CounterVec
	breaker    *prometheus.GaugeVec
	tokens     *prometheus.CounterVec
	similarity prometheus.Histogram
	embedErrs  prometheus.Counter
}

// NewMetrics registers collectors on a fresh registry.
func NewMetrics() *Metrics {
	m := &Metrics{
		Registry: prometheus.NewRegistry(),
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelmux_requests_total", Help: "Chat requests by model, HTTP status and cache result.",
		}, []string{"model", "status", "cache"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "modelmux_request_duration_seconds", Help: "End-to-end request duration.",
			Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60},
		}, []string{"stream"}),
		attempts: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelmux_upstream_attempts_total", Help: "Upstream attempts by provider and outcome.",
		}, []string{"provider", "outcome"}),
		firstByte: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "modelmux_upstream_first_byte_seconds", Help: "Time to first byte (stream) or full response.",
			Buckets: []float64{.05, .1, .25, .5, 1, 2, 5, 10, 20},
		}, []string{"provider"}),
		fallbacks: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelmux_fallbacks_total", Help: "Times a request moved to the next target in its chain.",
		}, []string{"model"}),
		breaker: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "modelmux_breaker_open", Help: "1 if the provider's circuit breaker is open.",
		}, []string{"provider"}),
		tokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelmux_tokens_total", Help: "Tokens served, by tenant and kind (prompt|completion).",
		}, []string{"tenant", "kind"}),
		similarity: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name: "modelmux_semantic_best_similarity", Help: "Best cosine similarity seen per semantic lookup.",
			Buckets: []float64{.5, .6, .7, .8, .85, .9, .92, .94, .96, .98, 1},
		}),
		embedErrs: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "modelmux_semantic_embed_errors_total", Help: "Semantic-cache embedding failures (treated as misses).",
		}),
	}
	m.Registry.MustRegister(m.requests, m.duration, m.attempts, m.firstByte, m.fallbacks, m.breaker,
		m.tokens, m.similarity, m.embedErrs)
	return m
}

// Attempt implements router.Observer.
func (m *Metrics) Attempt(provider, outcome string, firstByte time.Duration) {
	m.attempts.WithLabelValues(provider, outcome).Inc()
	if outcome == "ok" {
		m.firstByte.WithLabelValues(provider).Observe(firstByte.Seconds())
	}
}

// Fallback implements router.Observer.
func (m *Metrics) Fallback(model string) { m.fallbacks.WithLabelValues(model).Inc() }

// BreakerState implements router.Observer.
func (m *Metrics) BreakerState(provider string, open bool) {
	v := 0.0
	if open {
		v = 1
	}
	m.breaker.WithLabelValues(provider).Set(v)
}
