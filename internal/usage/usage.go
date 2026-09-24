// Package usage emits one event per completed request, for billing and
// analytics. The schema is documented in docs/usage-event.md.
package usage

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// SchemaVersion is bumped on any incompatible change to Event.
const SchemaVersion = 1

// Event is one usage record. ID is unique per request and is the idempotency
// key downstream consumers dedupe on.
type Event struct {
	Version          int       `json:"v"`
	ID               string    `json:"id"`
	Time             time.Time `json:"ts"`
	Tenant           string    `json:"tenant"`
	Model            string    `json:"model"`                    // as requested (alias)
	Provider         string    `json:"provider,omitempty"`       // who served it; empty on cache hit
	UpstreamModel    string    `json:"upstream_model,omitempty"` // provider's model id
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	Estimated        bool      `json:"tokens_estimated"` // true when upstream sent no usage
	LatencyMs        int64     `json:"latency_ms"`
	Cache            string    `json:"cache"` // miss | exact | semantic | bypass
	Status           int       `json:"status"`
	Stream           bool      `json:"stream"`
	Fallbacks        int       `json:"fallbacks"`
}

// NewID returns a random 128-bit hex id.
func NewID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// Sink receives events. Emit must not block the request path.
type Sink interface {
	Emit(Event)
	Close() error
}

// Nop discards events.
type Nop struct{}

// Emit implements Sink.
func (Nop) Emit(Event) {}

// Close implements Sink.
func (Nop) Close() error { return nil }

// JSONL writes one JSON object per line.
type JSONL struct {
	mu sync.Mutex
	w  io.Writer
}

// NewJSONL returns a JSONL sink on w.
func NewJSONL(w io.Writer) *JSONL { return &JSONL{w: w} }

// Emit implements Sink.
func (s *JSONL) Emit(e Event) {
	b, _ := json.Marshal(e)
	s.mu.Lock()
	defer s.mu.Unlock()
	_, _ = s.w.Write(append(b, '\n'))
}

// Close implements Sink.
func (s *JSONL) Close() error { return nil }

// Webhook POSTs batches of events as a JSON array. Emit enqueues without
// blocking; when the buffer is full the event is dropped and counted.
type Webhook struct {
	url      string
	client   *http.Client
	ch       chan Event
	batch    int
	interval time.Duration
	dropped  atomic.Int64
	failed   atomic.Int64
	done     chan struct{}
	mu       sync.RWMutex // guards closed vs. sends on ch
	closed   bool
}

// WebhookConfig configures a Webhook sink.
type WebhookConfig struct {
	URL       string
	Client    *http.Client
	Buffer    int           // default 10000
	BatchSize int           // default 500
	Interval  time.Duration // default 1s
}

// NewWebhook starts a Webhook sink's background sender.
func NewWebhook(c WebhookConfig) *Webhook {
	if c.Client == nil {
		c.Client = &http.Client{Timeout: 10 * time.Second}
	}
	if c.Buffer <= 0 {
		c.Buffer = 10_000
	}
	if c.BatchSize <= 0 {
		c.BatchSize = 500
	}
	if c.Interval <= 0 {
		c.Interval = time.Second
	}
	w := &Webhook{url: c.URL, client: c.Client, ch: make(chan Event, c.Buffer), batch: c.BatchSize,
		interval: c.Interval, done: make(chan struct{})}
	go w.run()
	return w
}

// Emit implements Sink.
func (w *Webhook) Emit(e Event) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		w.dropped.Add(1)
		return
	}
	select {
	case w.ch <- e:
	default:
		w.dropped.Add(1)
	}
}

// Dropped returns the number of events dropped because the buffer was full.
func (w *Webhook) Dropped() int64 { return w.dropped.Load() }

// Failed returns the number of events in batches the endpoint rejected.
func (w *Webhook) Failed() int64 { return w.failed.Load() }

func (w *Webhook) run() {
	defer close(w.done)
	t := time.NewTicker(w.interval)
	defer t.Stop()
	buf := make([]Event, 0, w.batch)
	flush := func() {
		if len(buf) == 0 {
			return
		}
		if err := w.send(buf); err != nil {
			w.failed.Add(int64(len(buf)))
		}
		buf = buf[:0]
	}
	for {
		select {
		case e, ok := <-w.ch:
			if !ok {
				flush()
				return
			}
			buf = append(buf, e)
			if len(buf) >= w.batch {
				flush()
			}
		case <-t.C:
			flush()
		}
	}
}

func (w *Webhook) send(events []Event) error {
	b, _ := json.Marshal(events)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.url, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := w.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("webhook: HTTP %d", resp.StatusCode)
	}
	return nil
}

// Close flushes buffered events and stops the sender.
func (w *Webhook) Close() error {
	w.mu.Lock()
	if !w.closed {
		w.closed = true
		close(w.ch)
	}
	w.mu.Unlock()
	<-w.done
	return nil
}
