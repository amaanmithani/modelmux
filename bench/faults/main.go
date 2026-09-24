// Command faults measures ModelMux's fallback behaviour under injected
// upstream failures and writes bench/results/faults.json.
//
// Two real HTTP upstreams: "a" (faulty, first in the chain) and "b" (healthy
// unless the scenario kills it). Requests go through the real HTTP server
// built from a YAML config, half streaming and half not.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/amaanmithani/modelmux/internal/config"
	"github.com/amaanmithani/modelmux/internal/stub"
)

const reply = "alpha bravo charlie delta echo foxtrot"

type faultMode string

const (
	ok503     faultMode = "503"       // fail with 503 before any byte
	hang      faultMode = "hang"      // accept, then send nothing for longer than the first-byte timeout
	midStream faultMode = "midstream" // stream one chunk then drop the connection
)

// faulty wraps the stub upstream and injects faults with probability p.
type faulty struct {
	h           http.Handler
	mode        faultMode
	p           float64
	mu          sync.Mutex
	rng         *rand.Rand
	hits        atomic.Int64
	fault       atomic.Int64
	faultStream atomic.Int64 // faults injected into streaming requests
}

func (f *faulty) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.hits.Add(1)
	body, _ := io.ReadAll(r.Body)
	r.Body = io.NopCloser(strings.NewReader(string(body)))
	isStream := strings.Contains(string(body), `"stream":true`)
	f.mu.Lock()
	inject := f.rng.Float64() < f.p
	f.mu.Unlock()
	if !inject {
		f.h.ServeHTTP(w, r)
		return
	}
	f.fault.Add(1)
	if isStream {
		f.faultStream.Add(1)
	}
	switch f.mode {
	case ok503:
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"injected outage"}}`)
	case hang:
		select {
		case <-time.After(3 * time.Second):
		case <-r.Context().Done():
		}
	case midStream:
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"alpha "}}]}`+"\n\n")
		_ = http.NewResponseController(w).Flush()
		if hj, ok := w.(http.Hijacker); ok {
			if c, _, err := hj.Hijack(); err == nil {
				_ = c.Close()
			}
		}
	}
}

type scenario struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Mode        string  `json:"fault_mode"`
	FaultRate   float64 `json:"fault_rate_on_a"`
	BDown       bool    `json:"b_down"`
	ADown       bool    `json:"a_down"`
	Breaker     string  `json:"breaker"`
}

type result struct {
	scenario
	Requests          int            `json:"requests"`
	Succeeded         int            `json:"succeeded"`
	SuccessRate       float64        `json:"success_rate"`
	InjectedFaults    int64          `json:"injected_faults_on_a"`
	InjectedInStreams int64          `json:"injected_faults_on_a_streaming"`
	UpstreamHitsA     int64          `json:"upstream_requests_to_a"`
	UpstreamHitsB     int64          `json:"upstream_requests_to_b"`
	Recoverable       int64          `json:"recoverable_failures"`
	Recovered         int64          `json:"recovered"`
	ServedBy          map[string]int `json:"served_by"`
	StatusCodes       map[string]int `json:"status_codes"`
	CorruptResponses  int            `json:"corrupt_responses"`
	InBandStreamError int            `json:"in_band_stream_errors"`
	LatencyP50Ms      float64        `json:"latency_p50_ms"`
	LatencyP99Ms      float64        `json:"latency_p99_ms"`
}

func main() {
	n := flag.Int("n", 2000, "requests per scenario")
	workers := flag.Int("c", 32, "concurrent clients")
	out := flag.String("out", "bench/results/faults.json", "output file")
	flag.Parse()

	scenarios := []scenario{
		{Name: "flaky-503", Description: "a returns 503 on 30% of requests; breaker effectively off", Mode: string(ok503), FaultRate: 0.3, Breaker: "off"},
		{Name: "hanging", Description: "a hangs on 30% of requests (never sends a byte); first-byte/request timeout 200ms", Mode: string(hang), FaultRate: 0.3, Breaker: "off"},
		{Name: "a-down-breaker-off", Description: "a returns 503 on every request; breaker off: every request pays a failed attempt", Mode: string(ok503), FaultRate: 1, Breaker: "off"},
		{Name: "a-down-breaker-on", Description: "a returns 503 on every request; breaker on (5 failures, 1s cooldown): a is skipped while open", Mode: string(ok503), FaultRate: 1, Breaker: "on"},
		{Name: "a-refusing", Description: "a refuses TCP connections; breaker on", ADown: true, Breaker: "on"},
		{Name: "both-dead", Description: "control: no healthy target exists", ADown: true, BDown: true, Breaker: "on"},
		{Name: "midstream", Description: "a drops 30% of streams after the first chunk (not recoverable by design: bytes already sent)", Mode: string(midStream), FaultRate: 0.3, Breaker: "off"},
	}
	var results []result
	for _, s := range scenarios {
		r := run(s, *n, *workers)
		fmt.Fprintf(os.Stderr, "%-10s success %.4f  recovered %d/%d  a-hits %d  p99 %.1fms\n",
			r.Name, r.SuccessRate, r.Recovered, r.Recoverable, r.UpstreamHitsA, r.LatencyP99Ms)
		results = append(results, r)
	}
	b, _ := json.MarshalIndent(map[string]any{
		"what":      "fallback correctness under injected upstream faults",
		"method":    fmt.Sprintf("%d requests per scenario (half streaming), %d concurrent clients, real HTTP end to end; chain = [a, b]", *n, *workers),
		"date":      time.Now().Format("2006-01-02"),
		"scenarios": results,
	}, "", "  ")
	if err := os.WriteFile(*out, append(b, '\n'), 0o644); err != nil {
		panic(err)
	}
}

func deadURL() string {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := l.Addr().String()
	_ = l.Close() // nothing listens here now
	return "http://" + addr + "/v1"
}

func run(s scenario, n, workers int) result {
	a := &faulty{h: stub.Handler(reply, 0), mode: faultMode(s.Mode), p: s.FaultRate, rng: rand.New(rand.NewSource(42))}
	b := &faulty{h: stub.Handler(reply, 0), rng: rand.New(rand.NewSource(7))}
	aURL, bURL := deadURL(), deadURL()
	if !s.ADown {
		sa := httptest.NewServer(a)
		defer sa.Close()
		aURL = sa.URL + "/v1"
	}
	if !s.BDown {
		sb := httptest.NewServer(b)
		defer sb.Close()
		bURL = sb.URL + "/v1"
	}
	breaker := "{failures: 1000000, cooldown: 1s}"
	if s.Breaker == "on" {
		breaker = "{failures: 5, cooldown: 1s}"
	}
	yaml := fmt.Sprintf(`
providers:
  - {name: a, type: openai, base_url: %q}
  - {name: b, type: openai, base_url: %q}
routes:
  - {alias: m, targets: [a/x, b/y]}
timeouts: {request: 200ms, first_byte: 200ms}
breaker: %s
public: {enabled: true}
`, aURL, bURL, breaker)
	cfg, err := config.Load(strings.NewReader(yaml))
	if err != nil {
		panic(err)
	}
	built, err := cfg.Build(os.Getenv, slog.New(slog.NewTextHandler(io.Discard, nil)), io.Discard)
	if err != nil {
		panic(err)
	}
	gw := httptest.NewServer(built.Handler)
	defer gw.Close()

	res := result{scenario: s, Requests: n, ServedBy: map[string]int{}, StatusCodes: map[string]int{}}
	var mu sync.Mutex
	lat := make([]float64, 0, n)
	jobs := make(chan int)
	var wg sync.WaitGroup
	client := &http.Client{Timeout: 10 * time.Second}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				stream := i%2 == 0
				body := fmt.Sprintf(`{"model":"m","stream":%v,"messages":[{"role":"user","content":"req %d"}]}`, stream, i)
				start := time.Now()
				resp, err := client.Post(gw.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
				code, served, okBody, inBand := "transport_error", "", false, false
				if err == nil {
					code, served = fmt.Sprint(resp.StatusCode), resp.Header.Get("X-ModelMux-Provider")
					if resp.StatusCode == 200 {
						okBody, inBand = checkBody(resp.Body, stream)
					}
					resp.Body.Close()
				}
				el := float64(time.Since(start).Microseconds()) / 1000
				mu.Lock()
				res.StatusCodes[code]++
				lat = append(lat, el)
				if okBody {
					res.Succeeded++
					res.ServedBy[served]++
				} else if code == "200" {
					if inBand {
						res.InBandStreamError++
					} else {
						res.CorruptResponses++
					}
				}
				mu.Unlock()
			}
		}()
	}
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	sort.Float64s(lat)
	res.LatencyP50Ms, res.LatencyP99Ms = lat[len(lat)/2], lat[len(lat)*99/100]
	res.SuccessRate = float64(res.Succeeded) / float64(n)
	res.InjectedInStreams = a.faultStream.Load()
	res.InjectedFaults, res.UpstreamHitsA, res.UpstreamHitsB = a.fault.Load(), a.hits.Load(), b.hits.Load()
	// Recoverable: requests a did not serve (it failed, hung, or its breaker
	// was open) while a healthy b existed. Mid-stream drops are excluded: bytes
	// had already reached the client, so no gateway can retry them.
	if !s.BDown {
		if s.Mode == string(midStream) {
			res.Recoverable = a.fault.Load() - a.faultStream.Load()
		} else {
			res.Recoverable = int64(n - res.ServedBy["a"])
		}
		res.Recovered = int64(res.ServedBy["b"])
	}
	return res
}

// checkBody verifies the full reply arrived intact. For streams it also
// reports whether the stream ended with an in-band error event.
func checkBody(r io.Reader, stream bool) (ok bool, inBandErr bool) {
	if !stream {
		var v struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.NewDecoder(r).Decode(&v) != nil || len(v.Choices) == 0 {
			return false, false
		}
		return v.Choices[0].Message.Content == reply, false
	}
	var sb strings.Builder
	sc := bufio.NewScanner(r)
	done := false
	for sc.Scan() {
		line := strings.TrimPrefix(sc.Text(), "data: ")
		if line == "[DONE]" {
			done = true
			break
		}
		if strings.Contains(line, `"error"`) {
			return false, true
		}
		var c struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(line), &c) == nil && len(c.Choices) > 0 {
			sb.WriteString(c.Choices[0].Delta.Content)
		}
	}
	return done && sb.String() == reply, false
}
