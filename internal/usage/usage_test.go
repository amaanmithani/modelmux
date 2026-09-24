package usage

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewIDUnique(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 1000; i++ {
		id := NewID()
		if len(id) != 32 || seen[id] {
			t.Fatal("bad or duplicate id")
		}
		seen[id] = true
	}
}

func TestJSONLAndNop(t *testing.T) {
	var buf bytes.Buffer
	s := NewJSONL(&buf)
	s.Emit(Event{Version: 1, ID: "a", Tenant: "t"})
	s.Emit(Event{Version: 1, ID: "b"})
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	var e Event
	if len(lines) != 2 || json.Unmarshal([]byte(lines[0]), &e) != nil || e.ID != "a" || e.Tenant != "t" {
		t.Fatalf("jsonl: %q", buf.String())
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
	Nop{}.Emit(Event{})
	if (Nop{}).Close() != nil {
		t.Fatal("nop close")
	}
}

func TestWebhookBatchesAndFlushesOnClose(t *testing.T) {
	var mu sync.Mutex
	var got []Event
	batches := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var evs []Event
		_ = json.NewDecoder(r.Body).Decode(&evs)
		mu.Lock()
		got = append(got, evs...)
		batches++
		mu.Unlock()
	}))
	defer srv.Close()
	w := NewWebhook(WebhookConfig{URL: srv.URL, BatchSize: 3, Interval: time.Hour})
	for i := 0; i < 7; i++ {
		w.Emit(Event{ID: NewID()})
	}
	_ = w.Close()
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 7 || batches != 3 { // 3 + 3 + final flush of 1
		t.Fatalf("got %d events in %d batches", len(got), batches)
	}
	w.Emit(Event{}) // after close: dropped, no panic
	if w.Dropped() != 1 {
		t.Fatal("emit after close should count as dropped")
	}
	_ = w.Close() // idempotent
}

func TestWebhookDropsWhenFullAndCountsFailures(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	w := NewWebhook(WebhookConfig{URL: srv.URL, Buffer: 2, BatchSize: 1, Interval: time.Hour})
	for i := 0; i < 50; i++ {
		w.Emit(Event{})
	}
	if w.Dropped() == 0 {
		t.Fatal("full buffer should drop")
	}
	close(block)
	_ = w.Close()
	if w.Failed() == 0 {
		t.Fatal("5xx batches should count as failed")
	}
	bad := NewWebhook(WebhookConfig{URL: "http://127.0.0.1:1", Interval: 10 * time.Millisecond})
	bad.Emit(Event{})
	time.Sleep(50 * time.Millisecond)
	_ = bad.Close()
	if bad.Failed() != 1 {
		t.Fatalf("unreachable endpoint: failed=%d", bad.Failed())
	}
	inv := NewWebhook(WebhookConfig{URL: "::bad-url"})
	inv.Emit(Event{})
	_ = inv.Close()
	if inv.Failed() != 1 {
		t.Fatal("invalid url should fail the batch")
	}
}
