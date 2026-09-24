package provider

import (
	"context"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
)

// Fake is an in-process Provider for tests and the "stub" provider type.
// It replies with Reply split into word chunks, after Latency. Errors queued
// with FailNext are returned (one per call) before any successful reply.
type Fake struct {
	ProviderName string
	Reply        string
	Latency      time.Duration
	// MidStreamErr, if set, is returned by Recv after the first content chunk.
	MidStreamErr error

	mu    sync.Mutex
	fails []error
	calls int
	last  *api.ChatRequest
}

// Name implements Provider.
func (f *Fake) Name() string { return f.ProviderName }

// FailNext queues errors returned by the next calls, in order.
func (f *Fake) FailNext(errs ...error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fails = append(f.fails, errs...)
}

// Calls returns how many Chat/Stream calls were made.
func (f *Fake) Calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// LastRequest returns the most recent request received.
func (f *Fake) LastRequest() *api.ChatRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.last
}

func (f *Fake) begin(ctx context.Context, req *api.ChatRequest) error {
	f.mu.Lock()
	f.calls++
	f.last = req.Clone()
	var err error
	if len(f.fails) > 0 {
		err, f.fails = f.fails[0], f.fails[1:]
	}
	f.mu.Unlock()
	if f.Latency > 0 {
		t := time.NewTimer(f.Latency)
		defer t.Stop()
		select {
		case <-ctx.Done():
			return &UpstreamError{Provider: f.ProviderName, Err: ctx.Err()}
		case <-t.C:
		}
	}
	return err
}

func (f *Fake) usage(req *api.ChatRequest) *api.Usage {
	in := 0
	for _, m := range req.Messages {
		in += len(strings.Fields(m.Content.Text()))
	}
	out := len(strings.Fields(f.Reply))
	return &api.Usage{PromptTokens: in, CompletionTokens: out, TotalTokens: in + out}
}

// Chat implements Provider.
func (f *Fake) Chat(ctx context.Context, req *api.ChatRequest) (*api.ChatResponse, error) {
	if err := f.begin(ctx, req); err != nil {
		return nil, err
	}
	return &api.ChatResponse{ID: "chatcmpl-fake", Object: "chat.completion", Created: 1, Model: req.Model,
		Choices: []api.Choice{{Message: api.Message{Role: "assistant", Content: api.TextContent(f.Reply)}, FinishReason: "stop"}},
		Usage:   f.usage(req)}, nil
}

// Stream implements Provider.
func (f *Fake) Stream(ctx context.Context, req *api.ChatRequest) (Stream, error) {
	if err := f.begin(ctx, req); err != nil {
		return nil, err
	}
	mk := func(d api.Delta, fr *string) *api.ChatChunk {
		return &api.ChatChunk{ID: "chatcmpl-fake", Object: "chat.completion.chunk", Created: 1, Model: req.Model,
			Choices: []api.ChunkChoice{{Delta: d, FinishReason: fr}}}
	}
	all := []*api.ChatChunk{mk(api.Delta{Role: "assistant", Content: api.Ptr("")}, nil)}
	for _, w := range strings.SplitAfter(f.Reply, " ") {
		if w != "" {
			all = append(all, mk(api.Delta{Content: api.Ptr(w)}, nil))
		}
	}
	all = append(all, mk(api.Delta{}, api.Ptr("stop")))
	all = append(all, &api.ChatChunk{ID: "chatcmpl-fake", Object: "chat.completion.chunk", Created: 1, Model: req.Model,
		Choices: []api.ChunkChoice{}, Usage: f.usage(req)})
	return &fakeStream{chunks: all, midErr: f.MidStreamErr}, nil
}

type fakeStream struct {
	chunks []*api.ChatChunk
	i      int
	midErr error
}

func (s *fakeStream) Recv() (*api.ChatChunk, error) {
	if s.midErr != nil && s.i == 2 {
		return nil, s.midErr
	}
	if s.i >= len(s.chunks) {
		return nil, io.EOF
	}
	c := s.chunks[s.i]
	s.i++
	return c, nil
}

func (s *fakeStream) Close() error { return nil }

// Embed implements Embedder with a deterministic bag-of-words hash embedding,
// so tests can exercise the semantic cache without a model.
func (f *Fake) Embed(_ context.Context, _ string, texts []string) ([][]float32, error) {
	const dim = 64
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, dim)
		for _, w := range strings.Fields(strings.ToLower(t)) {
			h := uint32(2166136261)
			for j := 0; j < len(w); j++ {
				h = (h ^ uint32(w[j])) * 16777619
			}
			v[h%dim]++
		}
		out[i] = v
	}
	return out, nil
}
