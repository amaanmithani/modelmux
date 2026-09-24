package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/amaanmithani/modelmux/internal/api"
)

// OpenAI speaks the OpenAI chat-completions wire protocol. It covers OpenAI,
// Groq, Gemini's OpenAI-compatible endpoint, Ollama (/v1) and vLLM-style servers.
type OpenAI struct {
	name    string
	baseURL string
	apiKey  string
	headers map[string]string
	client  *http.Client
	// streamUsage asks the upstream for a trailing usage chunk
	// (stream_options.include_usage). Disable for servers that reject it.
	streamUsage bool
}

// OpenAIConfig configures an OpenAI-compatible provider.
type OpenAIConfig struct {
	Name        string
	BaseURL     string // e.g. https://api.groq.com/openai/v1
	APIKey      string
	Headers     map[string]string
	Client      *http.Client
	StreamUsage bool
}

// NewOpenAI returns an OpenAI-compatible provider.
func NewOpenAI(c OpenAIConfig) *OpenAI {
	cl := c.Client
	if cl == nil {
		cl = http.DefaultClient
	}
	return &OpenAI{name: c.Name, baseURL: strings.TrimRight(c.BaseURL, "/"), apiKey: c.APIKey,
		headers: c.Headers, client: cl, streamUsage: c.StreamUsage}
}

// Name implements Provider.
func (p *OpenAI) Name() string { return p.name }

func (p *OpenAI) post(ctx context.Context, path string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	for k, v := range p.headers {
		hreq.Header.Set(k, v)
	}
	resp, err := p.client.Do(hreq)
	if err != nil {
		return nil, &UpstreamError{Provider: p.name, Err: err}
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		return nil, readError(p.name, resp)
	}
	return resp, nil
}

// Chat implements Provider.
func (p *OpenAI) Chat(ctx context.Context, req *api.ChatRequest) (*api.ChatResponse, error) {
	r := req.Clone()
	r.Stream, r.StreamOptions = false, nil
	resp, err := p.post(ctx, "/chat/completions", r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out api.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, &UpstreamError{Provider: p.name, Err: fmt.Errorf("decode response: %w", err)}
	}
	return &out, nil
}

// Stream implements Provider.
func (p *OpenAI) Stream(ctx context.Context, req *api.ChatRequest) (Stream, error) {
	r := req.Clone()
	r.Stream = true
	if p.streamUsage {
		r.StreamOptions = &api.StreamOptions{IncludeUsage: true}
	} else {
		r.StreamOptions = nil
	}
	resp, err := p.post(ctx, "/chat/completions", r)
	if err != nil {
		return nil, err
	}
	return &openAIStream{name: p.name, body: resp.Body, sse: newSSEReader(resp.Body)}, nil
}

type openAIStream struct {
	name string
	body io.ReadCloser
	sse  *sseReader
	done bool
}

func (s *openAIStream) Recv() (*api.ChatChunk, error) {
	if s.done {
		return nil, io.EOF
	}
	_, data, err := s.sse.next()
	if err == io.EOF {
		// Body ended without [DONE]: treat as truncated.
		return nil, &UpstreamError{Provider: s.name, Err: io.ErrUnexpectedEOF}
	}
	if err != nil {
		return nil, &UpstreamError{Provider: s.name, Err: err}
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
		s.done = true
		return nil, io.EOF
	}
	var probe struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &probe) == nil && probe.Error != nil {
		return nil, &UpstreamError{Provider: s.name, Status: http.StatusBadGateway, Message: probe.Error.Message}
	}
	var c api.ChatChunk
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, &UpstreamError{Provider: s.name, Err: fmt.Errorf("decode chunk: %w", err)}
	}
	return &c, nil
}

func (s *openAIStream) Close() error { return s.body.Close() }

// Embed implements Embedder via POST /embeddings.
func (p *OpenAI) Embed(ctx context.Context, model string, texts []string) ([][]float32, error) {
	resp, err := p.post(ctx, "/embeddings", map[string]any{"model": model, "input": texts})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, &UpstreamError{Provider: p.name, Err: fmt.Errorf("decode embeddings: %w", err)}
	}
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index >= 0 && d.Index < len(vecs) {
			vecs[d.Index] = d.Embedding
		}
	}
	for i, v := range vecs {
		if v == nil {
			return nil, &UpstreamError{Provider: p.name, Err: fmt.Errorf("embedding %d missing", i)}
		}
	}
	return vecs, nil
}
