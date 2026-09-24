// Package server exposes ModelMux over HTTP.
package server

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/cache"
	"github.com/amaanmithani/modelmux/internal/provider"
	"github.com/amaanmithani/modelmux/internal/router"
	"github.com/amaanmithani/modelmux/internal/tenant"
	"github.com/amaanmithani/modelmux/internal/usage"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

//go:embed static
var staticFS embed.FS

// Response headers describing how a request was served.
const (
	HeaderProvider   = "X-ModelMux-Provider"
	HeaderCache      = "X-ModelMux-Cache"
	HeaderFallbacks  = "X-ModelMux-Fallbacks"
	HeaderSimilarity = "X-ModelMux-Similarity"
	// HeaderCacheMode is a request header: "bypass" skips caches, "force"
	// caches a request even when it isn't deterministic (temperature != 0).
	HeaderCacheMode = "X-ModelMux-Cache"
)

// Config wires a Server.
type Config struct {
	Router   *router.Router
	Tenants  *tenant.Manager
	Exact    *cache.Exact    // nil disables exact caching
	Semantic *cache.Semantic // nil disables semantic caching
	Sink     usage.Sink      // nil = discard
	Metrics  *Metrics        // nil = fresh registry
	Logger   *slog.Logger
	// TrustedProxyHops is how many reverse proxies in front of ModelMux append
	// to X-Forwarded-For. 0 ignores the header and uses the socket address.
	TrustedProxyHops int
	MaxBodyBytes     int64
	Now              func() time.Time
}

// Server is the HTTP front end.
type Server struct {
	cfg Config
}

// New returns a Server with defaults applied.
func New(c Config) *Server {
	if c.Sink == nil {
		c.Sink = usage.Nop{}
	}
	if c.Metrics == nil {
		c.Metrics = NewMetrics()
	}
	if c.Logger == nil {
		c.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if c.MaxBodyBytes <= 0 {
		c.MaxBodyBytes = 2 << 20
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return &Server{cfg: c}
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.Handle("GET /metrics", promhttp.HandlerFor(s.cfg.Metrics.Registry, promhttp.HandlerOpts{}))
	sub, _ := fs.Sub(staticFS, "static")
	mux.Handle("GET /", http.FileServer(http.FS(sub)))
	return mux
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, typ, code, msg string) {
	writeJSON(w, status, api.NewError(typ, code, msg))
}

// clientIP returns the caller's address, honouring TrustedProxyHops: with n
// trusted proxies the client is the n-th entry from the right of
// X-Forwarded-For (entries further left are client-controlled).
func (s *Server) clientIP(r *http.Request) string {
	if n := s.cfg.TrustedProxyHops; n > 0 {
		var hops []string
		for _, h := range r.Header.Values("X-Forwarded-For") {
			for _, p := range strings.Split(h, ",") {
				if p = strings.TrimSpace(p); p != "" {
					hops = append(hops, p)
				}
			}
		}
		if len(hops) >= n {
			return hops[len(hops)-n]
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	p, err := s.cfg.Tenants.Authenticate(r.Header.Get("Authorization"), s.clientIP(r))
	if err != nil {
		writeErr(w, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key", err.Error())
		return
	}
	list := api.ModelList{Object: "list", Data: []api.Model{}}
	for _, a := range s.cfg.Router.Aliases() {
		if p.AllowsModel(a) {
			list.Data = append(list.Data, api.Model{ID: a, Object: "model", OwnedBy: "modelmux"})
		}
	}
	writeJSON(w, http.StatusOK, list)
}

// reqState carries per-request bookkeeping for the usage event and metrics.
type reqState struct {
	start     time.Time
	principal *tenant.Principal
	req       *api.ChatRequest
	model     string // metrics label: the alias if known, else "unknown"
	status    int
	cache     string
	result    router.Result
	resp      *api.ChatResponse
	admitted  bool // passed auth, validation and limits; only these emit usage
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	st := &reqState{start: s.cfg.Now(), model: "unknown", cache: "miss"}
	defer s.finish(st)

	p, err := s.cfg.Tenants.Authenticate(r.Header.Get("Authorization"), s.clientIP(r))
	if err != nil {
		st.status = http.StatusUnauthorized
		writeErr(w, st.status, "invalid_request_error", "invalid_api_key", err.Error())
		return
	}
	st.principal = p

	var req api.ChatRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, s.cfg.MaxBodyBytes))
	if err := dec.Decode(&req); err != nil {
		st.status = http.StatusBadRequest
		writeErr(w, st.status, "invalid_request_error", "invalid_json", "invalid JSON body: "+err.Error())
		return
	}
	st.req = &req
	if req.Model == "" || len(req.Messages) == 0 {
		st.status = http.StatusBadRequest
		writeErr(w, st.status, "invalid_request_error", "missing_field", "model and messages are required")
		return
	}
	if _, err := s.cfg.Router.Resolve(req.Model); err != nil {
		st.status = http.StatusNotFound
		writeErr(w, st.status, "invalid_request_error", "model_not_found", err.Error())
		return
	}
	st.model = req.Model
	if !p.AllowsModel(req.Model) {
		st.status = http.StatusForbidden
		writeErr(w, st.status, "invalid_request_error", "model_not_allowed", tenant.ErrModelDenied.Error())
		return
	}
	if err := p.Admit(); err != nil {
		var le *tenant.LimitError
		if errors.As(err, &le) {
			w.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(le.RetryAfter.Seconds()))))
			st.status = http.StatusTooManyRequests
			code := "rate_limit_exceeded"
			if le.Kind == "budget" {
				code = "insufficient_quota"
			}
			writeErr(w, st.status, "rate_limit_error", code, le.Error())
			return
		}
		st.status = http.StatusInternalServerError
		writeErr(w, st.status, "server_error", "", err.Error())
		return
	}

	st.admitted = true

	mode := strings.ToLower(r.Header.Get(HeaderCacheMode))
	cacheable := mode != "bypass" && (mode == "force" || cache.Cacheable(&req))
	if mode == "bypass" {
		st.cache = "bypass"
	}
	var exactKey string
	if cacheable && s.cfg.Exact != nil {
		exactKey = cache.Key(p.Tenant, &req)
		if hit, ok := s.cfg.Exact.Get(exactKey); ok {
			st.cache, st.resp = "exact", hit
			s.serveCached(w, st, hit)
			return
		}
	}
	var sem cache.Match
	var semScope, semQuery string
	if cacheable && s.cfg.Semantic != nil {
		if q, ok := cache.SemanticQuery(&req); ok {
			semScope, semQuery = cache.SemanticScope(p.Tenant, &req), q
			m, err := s.cfg.Semantic.Lookup(r.Context(), semScope, q)
			if err != nil {
				s.cfg.Metrics.embedErrs.Inc()
				s.cfg.Logger.Warn("semantic lookup failed", "err", err)
			} else {
				sem = m
				s.cfg.Metrics.similarity.Observe(float64(max(m.Similarity, 0)))
				if m.Resp != nil {
					st.cache, st.resp = "semantic", m.Resp
					w.Header().Set(HeaderSimilarity, strconv.FormatFloat(float64(m.Similarity), 'f', 4, 32))
					s.serveCached(w, st, m.Resp)
					return
				}
			}
		}
	}
	store := func(resp *api.ChatResponse) {
		if !cacheable {
			return
		}
		if exactKey != "" {
			s.cfg.Exact.Put(exactKey, resp)
		}
		if semScope != "" && sem.Vec != nil {
			s.cfg.Semantic.Put(semScope, sem.Vec, semQuery, resp)
		}
	}

	if req.Stream {
		s.stream(w, r, st, store)
		return
	}
	resp, res, err := s.cfg.Router.Chat(r.Context(), &req)
	st.result = res
	if err != nil {
		s.upstreamErr(w, st, err)
		return
	}
	st.resp = resp
	store(resp)
	setServedHeaders(w, st)
	st.status = http.StatusOK
	writeJSON(w, st.status, resp)
}

func setServedHeaders(w http.ResponseWriter, st *reqState) {
	h := w.Header()
	h.Set(HeaderCache, st.cache)
	if st.result.Target.Provider != "" {
		h.Set(HeaderProvider, st.result.Target.Provider)
	}
	h.Set(HeaderFallbacks, strconv.Itoa(st.result.Fallbacks))
}

func (s *Server) serveCached(w http.ResponseWriter, st *reqState, resp *api.ChatResponse) {
	setServedHeaders(w, st)
	st.status = http.StatusOK
	if !st.req.Stream {
		writeJSON(w, st.status, resp)
		return
	}
	sse := newSSEWriter(w)
	includeUsage := st.req.StreamOptions != nil && st.req.StreamOptions.IncludeUsage
	for _, c := range api.ResponseToChunks(resp, includeUsage) {
		if sse.chunk(c) != nil {
			return
		}
	}
	sse.done()
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request, st *reqState, store func(*api.ChatResponse)) {
	up, res, err := s.cfg.Router.Stream(r.Context(), st.req)
	st.result = res
	if err != nil {
		s.upstreamErr(w, st, err)
		return
	}
	defer up.Close()
	setServedHeaders(w, st)
	st.status = http.StatusOK
	sse := newSSEWriter(w)
	includeUsage := st.req.StreamOptions != nil && st.req.StreamOptions.IncludeUsage
	var acc api.Accumulator
	for {
		c, err := up.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			// Headers are sent; report in-band like OpenAI does, and don't cache.
			st.resp = acc.Response()
			if r.Context().Err() == nil {
				s.cfg.Logger.Warn("stream failed mid-response", "err", err, "provider", res.Target.Provider)
				_ = sse.event(api.NewError("server_error", "upstream_stream_error", err.Error()))
				st.status = http.StatusBadGateway
			} else {
				st.status = 499
			}
			return
		}
		acc.Add(c)
		if len(c.Choices) == 0 && c.Usage != nil && !includeUsage {
			continue // upstream usage chunk the client didn't ask for
		}
		if !includeUsage {
			c.Usage = nil
		}
		if sse.chunk(c) != nil {
			st.status = 499
			st.resp = acc.Response()
			return
		}
	}
	sse.done()
	st.resp = acc.Response()
	store(st.resp)
}

func (s *Server) upstreamErr(w http.ResponseWriter, st *reqState, err error) {
	var ue *provider.UpstreamError
	switch {
	case errors.Is(err, router.ErrNoHealthyTarget):
		st.status = http.StatusServiceUnavailable
		writeErr(w, st.status, "server_error", "no_healthy_upstream", err.Error())
	case errors.As(err, &ue) && errors.Is(ue.Err, context.Canceled):
		st.status = 499 // client closed request; nothing useful to write
	case errors.As(err, &ue):
		switch {
		case ue.Status == http.StatusBadRequest || ue.Status == http.StatusUnprocessableEntity ||
			ue.Status == http.StatusRequestEntityTooLarge:
			st.status = ue.Status
			writeErr(w, st.status, "invalid_request_error", "upstream_rejected", ue.Error())
		case ue.Status == http.StatusTooManyRequests:
			st.status = http.StatusTooManyRequests
			writeErr(w, st.status, "rate_limit_error", "upstream_rate_limited", ue.Error())
		case ue.Status == 0 && errors.Is(ue.Err, context.DeadlineExceeded):
			st.status = http.StatusGatewayTimeout
			writeErr(w, st.status, "server_error", "upstream_timeout", ue.Error())
		default:
			st.status = http.StatusBadGateway
			writeErr(w, st.status, "server_error", "upstream_error", ue.Error())
		}
	default:
		st.status = http.StatusBadGateway
		writeErr(w, st.status, "server_error", "upstream_error", err.Error())
	}
}

// finish records metrics and emits the usage event.
func (s *Server) finish(st *reqState) {
	elapsed := s.cfg.Now().Sub(st.start)
	stream := st.req != nil && st.req.Stream
	s.cfg.Metrics.requests.WithLabelValues(st.model, strconv.Itoa(st.status), st.cache).Inc()
	s.cfg.Metrics.duration.WithLabelValues(strconv.FormatBool(stream)).Observe(elapsed.Seconds())
	if !st.admitted {
		return
	}
	ev := usage.Event{Version: usage.SchemaVersion, ID: usage.NewID(), Time: st.start.UTC(),
		Tenant: st.principal.Tenant, Model: st.req.Model, Provider: st.result.Target.Provider,
		UpstreamModel: st.result.Target.Model, LatencyMs: elapsed.Milliseconds(), Cache: st.cache,
		Status: st.status, Stream: stream, Fallbacks: st.result.Fallbacks}
	if st.resp != nil {
		if u := st.resp.Usage; u != nil && (u.PromptTokens > 0 || u.CompletionTokens > 0) {
			ev.PromptTokens, ev.CompletionTokens = u.PromptTokens, u.CompletionTokens
		} else {
			ev.PromptTokens, ev.CompletionTokens, ev.Estimated = estimateTokens(st.req, st.resp)
		}
	}
	if st.cache != "exact" && st.cache != "semantic" {
		st.principal.Charge(ev.PromptTokens + ev.CompletionTokens)
	}
	if ev.PromptTokens+ev.CompletionTokens > 0 {
		s.cfg.Metrics.tokens.WithLabelValues(ev.Tenant, "prompt").Add(float64(ev.PromptTokens))
		s.cfg.Metrics.tokens.WithLabelValues(ev.Tenant, "completion").Add(float64(ev.CompletionTokens))
	}
	s.cfg.Sink.Emit(ev)
	s.cfg.Logger.Info("request", "tenant", ev.Tenant, "model", ev.Model, "provider", ev.Provider,
		"status", ev.Status, "cache", ev.Cache, "fallbacks", ev.Fallbacks, "latency_ms", ev.LatencyMs, "stream", stream)
}

// estimateTokens approximates token counts at ~4 characters per token when
// the upstream reports no usage. Events carry Estimated=true in that case.
func estimateTokens(req *api.ChatRequest, resp *api.ChatResponse) (int, int, bool) {
	in, out := 0, 0
	for _, m := range req.Messages {
		in += len(m.Content.Text())
	}
	for _, c := range resp.Choices {
		out += len(c.Message.Content.Text())
		for _, tc := range c.Message.ToolCalls {
			out += len(tc.Function.Arguments) + len(tc.Function.Name)
		}
	}
	return (in + 3) / 4, (out + 3) / 4, true
}

type sseWriter struct {
	w  http.ResponseWriter
	rc *http.ResponseController
}

func newSSEWriter(w http.ResponseWriter) *sseWriter {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	return &sseWriter{w: w, rc: http.NewResponseController(w)}
}

func (s *sseWriter) event(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	return s.rc.Flush()
}

func (s *sseWriter) chunk(c *api.ChatChunk) error { return s.event(c) }

func (s *sseWriter) done() {
	_, _ = io.WriteString(s.w, "data: [DONE]\n\n")
	_ = s.rc.Flush()
}
