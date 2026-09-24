package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
)

// AnthropicVersion is the Messages API version header sent upstream.
const AnthropicVersion = "2023-06-01"

// defaultAnthropicMaxTokens is used when the client sets no cap; the Messages
// API requires one.
const defaultAnthropicMaxTokens = 1024

// Anthropic translates OpenAI chat completions to the Anthropic Messages API
// and back, including streaming and tool use.
type Anthropic struct {
	name    string
	baseURL string
	apiKey  string
	client  *http.Client
	now     func() time.Time
}

// AnthropicConfig configures an Anthropic provider.
type AnthropicConfig struct {
	Name    string
	BaseURL string // default https://api.anthropic.com/v1
	APIKey  string
	Client  *http.Client
}

// NewAnthropic returns an Anthropic provider.
func NewAnthropic(c AnthropicConfig) *Anthropic {
	base := c.BaseURL
	if base == "" {
		base = "https://api.anthropic.com/v1"
	}
	cl := c.Client
	if cl == nil {
		cl = http.DefaultClient
	}
	return &Anthropic{name: c.Name, baseURL: strings.TrimRight(base, "/"), apiKey: c.APIKey, client: cl, now: time.Now}
}

// Name implements Provider.
func (p *Anthropic) Name() string { return p.name }

type anthSource struct {
	Type      string `json:"type"` // base64 | url
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthBlock struct {
	Type      string          `json:"type"`
	Source    *anthSource     `json:"source,omitempty"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
}

type anthMsg struct {
	Role    string      `json:"role"`
	Content []anthBlock `json:"content"`
}

type anthTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthReq struct {
	Model         string     `json:"model"`
	System        string     `json:"system,omitempty"`
	Messages      []anthMsg  `json:"messages"`
	MaxTokens     int        `json:"max_tokens"`
	Temperature   *float64   `json:"temperature,omitempty"`
	TopP          *float64   `json:"top_p,omitempty"`
	StopSequences []string   `json:"stop_sequences,omitempty"`
	Stream        bool       `json:"stream,omitempty"`
	Tools         []anthTool `json:"tools,omitempty"`
	ToolChoice    any        `json:"tool_choice,omitempty"`
}

type anthUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthResp struct {
	ID         string      `json:"id"`
	Model      string      `json:"model"`
	Content    []anthBlock `json:"content"`
	StopReason string      `json:"stop_reason"`
	Usage      anthUsage   `json:"usage"`
}

// errUnsupportedPart marks content this adapter can't express (audio, files).
// It maps to a fallbackable 501 so another provider can take the request.
type errUnsupportedPart struct{ typ string }

func (e errUnsupportedPart) Error() string {
	return fmt.Sprintf("anthropic adapter: unsupported content part %q", e.typ)
}

// contentBlocks translates OpenAI content parts, in order, to Anthropic blocks.
func contentBlocks(c api.Content) ([]anthBlock, error) {
	var out []anthBlock
	for _, p := range c.Parts() {
		switch p.Type {
		case "text":
			if p.Text != "" {
				out = append(out, anthBlock{Type: "text", Text: p.Text})
			}
		case "image_url":
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return nil, errUnsupportedPart{typ: "image_url without url"}
			}
			u := p.ImageURL.URL
			if rest, ok := strings.CutPrefix(u, "data:"); ok {
				meta, data, found := strings.Cut(rest, ",")
				media, isB64 := strings.CutSuffix(meta, ";base64")
				if !found || !isB64 {
					return nil, errUnsupportedPart{typ: "non-base64 data URL"}
				}
				out = append(out, anthBlock{Type: "image", Source: &anthSource{Type: "base64", MediaType: media, Data: data}})
			} else {
				out = append(out, anthBlock{Type: "image", Source: &anthSource{Type: "url", URL: u}})
			}
		default:
			return nil, errUnsupportedPart{typ: p.Type}
		}
	}
	return out, nil
}

// toAnthropic translates an OpenAI request. It returns an error for requests
// that cannot be expressed (e.g. no messages after removing system prompts).
func toAnthropic(req *api.ChatRequest) (*anthReq, error) {
	out := &anthReq{Model: req.Model, MaxTokens: req.MaxOutputTokens(), TopP: req.TopP,
		StopSequences: req.StopSequences(), Stream: req.Stream}
	if out.MaxTokens <= 0 {
		out.MaxTokens = defaultAnthropicMaxTokens
	}
	if req.Temperature != nil {
		t := min(*req.Temperature, 1) // OpenAI allows 0–2, Anthropic 0–1
		out.Temperature = &t
	}
	var system []string
	for _, m := range req.Messages {
		var role string
		var blocks []anthBlock
		switch m.Role {
		case "system", "developer":
			if t := m.Content.Text(); t != "" {
				system = append(system, t)
			}
			continue
		case "user":
			role = "user"
			bs, err := contentBlocks(m.Content)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, bs...)
		case "assistant":
			role = "assistant"
			if t := m.Content.Text(); t != "" {
				blocks = append(blocks, anthBlock{Type: "text", Text: t})
			}
			for _, tc := range m.ToolCalls {
				input := json.RawMessage(tc.Function.Arguments)
				if !json.Valid(input) || len(bytes.TrimSpace(input)) == 0 {
					input = json.RawMessage("{}")
				}
				blocks = append(blocks, anthBlock{Type: "tool_use", ID: tc.ID, Name: tc.Function.Name, Input: input})
			}
		case "tool":
			role = "user"
			blocks = append(blocks, anthBlock{Type: "tool_result", ToolUseID: m.ToolCallID, Content: m.Content.Text()})
		default:
			return nil, fmt.Errorf("unsupported message role %q", m.Role)
		}
		if len(blocks) == 0 {
			continue
		}
		// The Messages API requires alternating roles: merge consecutive turns.
		if n := len(out.Messages); n > 0 && out.Messages[n-1].Role == role {
			out.Messages[n-1].Content = append(out.Messages[n-1].Content, blocks...)
		} else {
			out.Messages = append(out.Messages, anthMsg{Role: role, Content: blocks})
		}
	}
	if len(out.Messages) == 0 {
		return nil, fmt.Errorf("request has no user or assistant messages")
	}
	out.System = strings.Join(system, "\n\n")
	for _, t := range req.Tools {
		schema := t.Function.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out.Tools = append(out.Tools, anthTool{Name: t.Function.Name, Description: t.Function.Description, InputSchema: schema})
	}
	if len(req.ToolChoice) > 0 {
		var s string
		if json.Unmarshal(req.ToolChoice, &s) == nil {
			switch s {
			case "auto":
				out.ToolChoice = map[string]string{"type": "auto"}
			case "none":
				out.ToolChoice = map[string]string{"type": "none"}
			case "required":
				out.ToolChoice = map[string]string{"type": "any"}
			}
		} else {
			var obj struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if json.Unmarshal(req.ToolChoice, &obj) == nil && obj.Function.Name != "" {
				out.ToolChoice = map[string]string{"type": "tool", "name": obj.Function.Name}
			}
		}
	}
	return out, nil
}

func finishReason(stop string) string {
	switch stop {
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	default: // end_turn, stop_sequence, pause_turn
		return "stop"
	}
}

func (p *Anthropic) post(ctx context.Context, body *anthReq) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("x-api-key", p.apiKey)
	hreq.Header.Set("anthropic-version", AnthropicVersion)
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

func (p *Anthropic) translateErr(err error) error {
	status := http.StatusBadRequest
	var up errUnsupportedPart
	if errors.As(err, &up) {
		status = http.StatusNotImplemented // another provider may support it
	}
	return &UpstreamError{Provider: p.name, Status: status, Message: err.Error()}
}

// Chat implements Provider.
func (p *Anthropic) Chat(ctx context.Context, req *api.ChatRequest) (*api.ChatResponse, error) {
	ar, err := toAnthropic(req)
	if err != nil {
		return nil, p.translateErr(err)
	}
	ar.Stream = false
	resp, err := p.post(ctx, ar)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var r anthResp
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, &UpstreamError{Provider: p.name, Err: fmt.Errorf("decode response: %w", err)}
	}
	msg := api.Message{Role: "assistant"}
	var text strings.Builder
	for _, b := range r.Content {
		switch b.Type {
		case "text":
			text.WriteString(b.Text)
		case "tool_use":
			args := string(b.Input)
			if args == "" {
				args = "{}"
			}
			msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{ID: b.ID, Type: "function",
				Function: api.FunctionCall{Name: b.Name, Arguments: args}})
		}
	}
	msg.Content = api.TextContent(text.String())
	return &api.ChatResponse{
		ID: "chatcmpl-" + r.ID, Object: "chat.completion", Created: p.now().Unix(), Model: r.Model,
		Choices: []api.Choice{{Index: 0, Message: msg, FinishReason: finishReason(r.StopReason)}},
		Usage: &api.Usage{PromptTokens: r.Usage.InputTokens, CompletionTokens: r.Usage.OutputTokens,
			TotalTokens: r.Usage.InputTokens + r.Usage.OutputTokens},
	}, nil
}

// Stream implements Provider.
func (p *Anthropic) Stream(ctx context.Context, req *api.ChatRequest) (Stream, error) {
	ar, err := toAnthropic(req)
	if err != nil {
		return nil, p.translateErr(err)
	}
	ar.Stream = true
	resp, err := p.post(ctx, ar)
	if err != nil {
		return nil, err
	}
	return &anthStream{p: p, body: resp.Body, sse: newSSEReader(resp.Body), toolIdx: map[int]int{}}, nil
}

type anthStream struct {
	p       *Anthropic
	body    io.ReadCloser
	sse     *sseReader
	queue   []*api.ChatChunk
	id      string
	model   string
	created int64
	usage   anthUsage
	toolIdx map[int]int // Anthropic content-block index -> OpenAI tool_calls index
	done    bool
	// roleSent: the assistant role rides on the first real chunk rather than
	// a chunk of its own, so an error arriving right after message_start
	// happens before the first byte and can still fall back.
	roleSent bool
}

func (s *anthStream) push(c *api.ChatChunk) {
	if !s.roleSent && len(c.Choices) > 0 {
		c.Choices[0].Delta.Role = "assistant"
		s.roleSent = true
	}
	s.queue = append(s.queue, c)
}

func (s *anthStream) chunk() *api.ChatChunk {
	return &api.ChatChunk{ID: s.id, Object: "chat.completion.chunk", Created: s.created, Model: s.model}
}

func (s *anthStream) Recv() (*api.ChatChunk, error) {
	for len(s.queue) == 0 {
		if s.done {
			return nil, io.EOF
		}
		if err := s.step(); err != nil {
			return nil, err
		}
	}
	c := s.queue[0]
	s.queue = s.queue[1:]
	return c, nil
}

func (s *anthStream) step() error {
	event, data, err := s.sse.next()
	if err == io.EOF {
		return &UpstreamError{Provider: s.p.name, Err: io.ErrUnexpectedEOF}
	}
	if err != nil {
		return &UpstreamError{Provider: s.p.name, Err: err}
	}
	var ev struct {
		Type    string `json:"type"`
		Index   int    `json:"index"`
		Message struct {
			ID    string    `json:"id"`
			Model string    `json:"model"`
			Usage anthUsage `json:"usage"`
		} `json:"message"`
		ContentBlock anthBlock `json:"content_block"`
		Delta        struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			PartialJSON string `json:"partial_json"`
			StopReason  string `json:"stop_reason"`
		} `json:"delta"`
		Usage anthUsage `json:"usage"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &ev); err != nil {
		return &UpstreamError{Provider: s.p.name, Err: fmt.Errorf("decode event: %w", err)}
	}
	if ev.Type == "" {
		ev.Type = event
	}
	switch ev.Type {
	case "message_start":
		s.id, s.model, s.created = "chatcmpl-"+ev.Message.ID, ev.Message.Model, s.p.now().Unix()
		s.usage.InputTokens = ev.Message.Usage.InputTokens
	case "content_block_start":
		if ev.ContentBlock.Type == "tool_use" {
			idx := len(s.toolIdx)
			s.toolIdx[ev.Index] = idx
			c := s.chunk()
			c.Choices = []api.ChunkChoice{{Delta: api.Delta{ToolCalls: []api.ToolCall{{Index: api.Ptr(idx),
				ID: ev.ContentBlock.ID, Type: "function", Function: api.FunctionCall{Name: ev.ContentBlock.Name}}}}}}
			s.push(c)
		} else if ev.ContentBlock.Type == "text" && ev.ContentBlock.Text != "" {
			c := s.chunk()
			c.Choices = []api.ChunkChoice{{Delta: api.Delta{Content: api.Ptr(ev.ContentBlock.Text)}}}
			s.push(c)
		}
	case "content_block_delta":
		c := s.chunk()
		switch ev.Delta.Type {
		case "text_delta":
			c.Choices = []api.ChunkChoice{{Delta: api.Delta{Content: api.Ptr(ev.Delta.Text)}}}
		case "input_json_delta":
			idx, ok := s.toolIdx[ev.Index]
			if !ok {
				return nil
			}
			c.Choices = []api.ChunkChoice{{Delta: api.Delta{ToolCalls: []api.ToolCall{{Index: api.Ptr(idx),
				Function: api.FunctionCall{Arguments: ev.Delta.PartialJSON}}}}}}
		default:
			return nil // thinking/signature deltas are not exposed
		}
		s.push(c)
	case "message_delta":
		s.usage.OutputTokens = ev.Usage.OutputTokens
		if ev.Usage.InputTokens > 0 {
			s.usage.InputTokens = ev.Usage.InputTokens
		}
		fr := finishReason(ev.Delta.StopReason)
		c := s.chunk()
		c.Choices = []api.ChunkChoice{{Delta: api.Delta{}, FinishReason: &fr}}
		u := s.chunk()
		u.Choices = []api.ChunkChoice{}
		u.Usage = &api.Usage{PromptTokens: s.usage.InputTokens, CompletionTokens: s.usage.OutputTokens,
			TotalTokens: s.usage.InputTokens + s.usage.OutputTokens}
		s.push(c)
		s.queue = append(s.queue, u)
	case "message_stop":
		s.done = true
	case "error":
		status := http.StatusBadGateway
		if ev.Error.Type == "overloaded_error" {
			status = 529
		}
		return &UpstreamError{Provider: s.p.name, Status: status, Message: ev.Error.Message}
	}
	return nil
}

func (s *anthStream) Close() error { return s.body.Close() }
