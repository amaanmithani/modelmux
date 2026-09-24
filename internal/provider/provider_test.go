package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
	"github.com/amaanmithani/modelmux/internal/stub"
)

func userReq(model, text string) *api.ChatRequest {
	return &api.ChatRequest{Model: model, Messages: []api.Message{{Role: "user", Content: api.TextContent(text)}}}
}

func drain(t *testing.T, s Stream) *api.ChatResponse {
	t.Helper()
	defer s.Close()
	var a api.Accumulator
	for {
		c, err := s.Recv()
		if err == io.EOF {
			return a.Response()
		}
		if err != nil {
			t.Fatalf("recv: %v", err)
		}
		a.Add(c)
	}
}

func TestFallbackable(t *testing.T) {
	cases := []struct {
		err  error
		want bool
	}{
		{nil, false},
		{&UpstreamError{Status: 429}, true},
		{&UpstreamError{Status: 503}, true},
		{&UpstreamError{Status: 401}, true},
		{&UpstreamError{Status: 404}, true},
		{&UpstreamError{Status: 400}, false},
		{&UpstreamError{Status: 422}, false},
		{&UpstreamError{Status: 418}, false},
		{&UpstreamError{Err: errors.New("conn refused")}, true},
		{&UpstreamError{Err: context.Canceled}, false},
		{context.Canceled, false},
		{context.DeadlineExceeded, true},
		{io.ErrUnexpectedEOF, true},
		{&net.OpError{Op: "dial", Err: errors.New("x")}, true},
		{errors.New("other"), false},
	}
	for _, c := range cases {
		if got := Fallbackable(c.err); got != c.want {
			t.Errorf("Fallbackable(%v) = %v, want %v", c.err, got, c.want)
		}
	}
}

func TestUpstreamErrorMessage(t *testing.T) {
	e := &UpstreamError{Provider: "p", Status: 500, Message: "boom"}
	if e.Error() != "p: HTTP 500: boom" {
		t.Fatal(e.Error())
	}
	e2 := &UpstreamError{Provider: "p", Err: io.EOF}
	if !strings.Contains(e2.Error(), "transport error") || !errors.Is(e2, io.EOF) {
		t.Fatal(e2.Error())
	}
}

func TestOpenAIChatAndStreamAgainstStub(t *testing.T) {
	srv := httptest.NewServer(stub.Handler("one two three", 0))
	defer srv.Close()
	p := NewOpenAI(OpenAIConfig{Name: "s", BaseURL: srv.URL + "/v1/", StreamUsage: true})
	if p.Name() != "s" {
		t.Fatal("name")
	}
	resp, err := p.Chat(context.Background(), userReq("m", "hi"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content.Text() != "one two three" || resp.Model != "m" {
		t.Fatalf("chat: %+v", resp)
	}
	s, err := p.Stream(context.Background(), userReq("m", "hi"))
	if err != nil {
		t.Fatal(err)
	}
	got := drain(t, s)
	if got.Choices[0].Message.Content.Text() != "one two three" || got.Usage == nil || got.Usage.CompletionTokens != 3 {
		t.Fatalf("stream: %+v usage=%+v", got.Choices, got.Usage)
	}
}

func TestOpenAISendsAuthHeadersAndNoUsageWhenDisabled(t *testing.T) {
	var gotAuth, gotX string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotX = r.Header.Get("Authorization"), r.Header.Get("X-Extra")
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, ": comment\n\ndata: {\"id\":\"a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	p := NewOpenAI(OpenAIConfig{Name: "s", BaseURL: srv.URL, APIKey: "sk-1", Headers: map[string]string{"X-Extra": "1"}})
	s, err := p.Stream(context.Background(), userReq("m", "hi"))
	if err != nil {
		t.Fatal(err)
	}
	got := drain(t, s)
	if gotAuth != "Bearer sk-1" || gotX != "1" {
		t.Fatalf("headers: %q %q", gotAuth, gotX)
	}
	if _, ok := body["stream_options"]; ok {
		t.Fatal("stream_options sent although StreamUsage is false")
	}
	if got.Choices[0].Message.Content.Text() != "x" {
		t.Fatal("content")
	}
}

func TestOpenAIErrors(t *testing.T) {
	status := http.StatusTooManyRequests
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
	}))
	defer srv.Close()
	p := NewOpenAI(OpenAIConfig{Name: "s", BaseURL: srv.URL})
	_, err := p.Chat(context.Background(), userReq("m", "hi"))
	var ue *UpstreamError
	if !errors.As(err, &ue) || ue.Status != 429 || ue.Message != "slow down" {
		t.Fatalf("got %v", err)
	}
	status = 500
	if _, err := p.Stream(context.Background(), userReq("m", "hi")); !Fallbackable(err) {
		t.Fatalf("stream 500 should be fallbackable: %v", err)
	}
	dead := NewOpenAI(OpenAIConfig{Name: "d", BaseURL: "http://127.0.0.1:1"})
	if _, err := dead.Chat(context.Background(), userReq("m", "hi")); !Fallbackable(err) {
		t.Fatalf("connection refused should be fallbackable: %v", err)
	}
}

func TestOpenAIStreamErrorsInBand(t *testing.T) {
	cases := map[string]string{
		"error event": "data: {\"error\":{\"message\":\"overloaded\"}}\n\n",
		"truncated":   "data: {\"id\":\"a\",\"choices\":[]}\n\n",
		"bad json":    "data: {nope\n\n",
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, body)
			}))
			defer srv.Close()
			s, err := NewOpenAI(OpenAIConfig{Name: "s", BaseURL: srv.URL}).Stream(context.Background(), userReq("m", "x"))
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			var last error
			for i := 0; i < 3; i++ {
				if _, last = s.Recv(); last != nil {
					break
				}
			}
			if last == nil || last == io.EOF {
				t.Fatalf("want error, got %v", last)
			}
		})
	}
}

func TestOpenAIEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]}`)
	}))
	defer srv.Close()
	p := NewOpenAI(OpenAIConfig{Name: "e", BaseURL: srv.URL})
	v, err := p.Embed(context.Background(), "m", []string{"a", "b"})
	if err != nil || v[0][0] != 1 || v[1][1] != 1 {
		t.Fatalf("embed: %v %v", v, err)
	}
	if _, err := p.Embed(context.Background(), "m", []string{"a", "b", "c"}); err == nil {
		t.Fatal("missing embedding should error")
	}
}

func TestSSEReaderMultilineAndEvents(t *testing.T) {
	r := newSSEReader(strings.NewReader("event: a\ndata: x\ndata: y\n\n\n: ping\nid: 3\ndata: z"))
	ev, d, err := r.next()
	if err != nil || ev != "a" || string(d) != "x\ny" {
		t.Fatalf("%q %q %v", ev, d, err)
	}
	_, d, err = r.next()
	if err != nil || string(d) != "z" {
		t.Fatalf("trailing event without blank line: %q %v", d, err)
	}
	if _, _, err := r.next(); err != io.EOF {
		t.Fatalf("want EOF, got %v", err)
	}
}

func TestFakeProvider(t *testing.T) {
	f := &Fake{ProviderName: "f", Reply: "a b"}
	f.FailNext(&UpstreamError{Status: 503})
	if _, err := f.Chat(context.Background(), userReq("m", "x y z")); err == nil {
		t.Fatal("queued failure not returned")
	}
	r, err := f.Chat(context.Background(), userReq("m", "x y z"))
	if err != nil || r.Usage.PromptTokens != 3 || r.Usage.CompletionTokens != 2 {
		t.Fatalf("%+v %v", r, err)
	}
	s, _ := f.Stream(context.Background(), userReq("m", "x"))
	if got := drain(t, s); got.Choices[0].Message.Content.Text() != "a b" {
		t.Fatalf("stream text %q", got.Choices[0].Message.Content.Text())
	}
	if f.Calls() != 3 || f.LastRequest().Model != "m" {
		t.Fatal("call accounting")
	}
	slow := &Fake{ProviderName: "slow", Latency: time.Second}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := slow.Chat(ctx, userReq("m", "x")); err == nil {
		t.Fatal("cancelled ctx should abort latency wait")
	}
	mid := &Fake{ProviderName: "mid", Reply: "a b c", MidStreamErr: io.ErrUnexpectedEOF}
	ms, _ := mid.Stream(context.Background(), userReq("m", "x"))
	_, _ = ms.Recv()
	_, _ = ms.Recv()
	if _, err := ms.Recv(); err != io.ErrUnexpectedEOF {
		t.Fatalf("mid-stream error: %v", err)
	}
	v, _ := f.Embed(context.Background(), "", []string{"hello world", "world hello"})
	for i := range v[0] {
		if v[0][i] != v[1][i] {
			t.Fatal("bag-of-words embedding should ignore order")
		}
	}
}
