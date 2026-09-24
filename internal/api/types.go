// Package api defines the OpenAI-compatible wire types ModelMux speaks to clients
// and to OpenAI-compatible upstreams.
package api

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Content is a message's content: either a plain string or an array of typed
// parts (OpenAI's multimodal form). The raw JSON is preserved so it can be
// forwarded to OpenAI-compatible upstreams untouched.
type Content struct {
	raw json.RawMessage
}

// TextContent returns Content holding a plain string.
func TextContent(s string) Content {
	b, _ := json.Marshal(s)
	return Content{raw: b}
}

// MarshalJSON implements json.Marshaler.
func (c Content) MarshalJSON() ([]byte, error) {
	if len(c.raw) == 0 {
		return []byte("null"), nil
	}
	return c.raw, nil
}

// UnmarshalJSON implements json.Unmarshaler.
func (c *Content) UnmarshalJSON(b []byte) error {
	c.raw = append(c.raw[:0], b...)
	return nil
}

// IsNull reports whether the content is absent or JSON null.
func (c Content) IsNull() bool {
	return len(c.raw) == 0 || bytes.Equal(bytes.TrimSpace(c.raw), []byte("null"))
}

// Text returns the textual content: the string itself, or the concatenation of
// all "text" parts. Non-text parts are ignored.
func (c Content) Text() string {
	if c.IsNull() {
		return ""
	}
	var s string
	if err := json.Unmarshal(c.raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(c.raw, &parts); err != nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// Part is one element of array-form content.
type Part struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url,omitempty"`
}

// Parts returns the content as typed parts: a plain string becomes one text
// part.
func (c Content) Parts() []Part {
	if c.IsNull() {
		return nil
	}
	var s string
	if err := json.Unmarshal(c.raw, &s); err == nil {
		return []Part{{Type: "text", Text: s}}
	}
	var parts []Part
	_ = json.Unmarshal(c.raw, &parts)
	return parts
}

// HasNonText reports whether the content includes anything but text
// (images, audio, files).
func (c Content) HasNonText() bool {
	for _, p := range c.Parts() {
		if p.Type != "text" {
			return true
		}
	}
	return false
}

// Message is one chat message.
type Message struct {
	Role       string     `json:"role"`
	Content    Content    `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is an assistant's request to call a function.
type ToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall is the function part of a ToolCall. Arguments is a JSON string.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// Tool is a function the model may call.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a callable function.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// StreamOptions controls streaming behaviour.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ChatRequest is the body of POST /v1/chat/completions.
type ChatRequest struct {
	Model               string          `json:"model"`
	Messages            []Message       `json:"messages"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	ResponseFormat      json.RawMessage `json:"response_format,omitempty"`
	User                string          `json:"user,omitempty"`
}

// Clone returns a shallow copy safe to mutate at the top level.
func (r *ChatRequest) Clone() *ChatRequest {
	c := *r
	c.Messages = append([]Message(nil), r.Messages...)
	return &c
}

// MaxOutputTokens returns the requested output-token cap, or 0 if unset.
func (r *ChatRequest) MaxOutputTokens() int {
	if r.MaxCompletionTokens != nil {
		return *r.MaxCompletionTokens
	}
	if r.MaxTokens != nil {
		return *r.MaxTokens
	}
	return 0
}

// StopSequences decodes Stop, which may be a string or an array of strings.
func (r *ChatRequest) StopSequences() []string {
	if len(r.Stop) == 0 {
		return nil
	}
	var one string
	if err := json.Unmarshal(r.Stop, &one); err == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many []string
	_ = json.Unmarshal(r.Stop, &many)
	return many
}

// Usage is token accounting for one completion.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Choice is one completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatResponse is a non-streaming completion.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Delta is the incremental message in a streaming chunk.
type Delta struct {
	Role      string     `json:"role,omitempty"`
	Content   *string    `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ChunkChoice is one choice in a streaming chunk.
type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// ChatChunk is one server-sent event of a streaming completion.
type ChatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// Model is one entry of GET /v1/models.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the body of GET /v1/models.
type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ErrorBody is OpenAI's error envelope.
type ErrorBody struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the inner error object.
type ErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Code    *string `json:"code"`
}

// NewError builds an ErrorBody.
func NewError(typ, code, msg string) ErrorBody {
	e := ErrorBody{Error: ErrorDetail{Message: msg, Type: typ}}
	if code != "" {
		e.Error.Code = &code
	}
	return e
}

// Ptr returns a pointer to v.
func Ptr[T any](v T) *T { return &v }

// MaxChoices bounds the choice index accepted from an upstream stream, so a
// malformed chunk can't panic the handler or grow memory without limit.
const MaxChoices = 16

// Accumulator folds streaming chunks into a complete ChatResponse. It is used
// to populate the cache and usage accounting from streamed completions.
type Accumulator struct {
	resp    ChatResponse
	content map[int]*strings.Builder
	tools   map[int]map[int]*ToolCall
	order   map[int][]int
}

// Add folds one chunk in.
func (a *Accumulator) Add(c *ChatChunk) {
	if a.content == nil {
		a.content = map[int]*strings.Builder{}
		a.tools = map[int]map[int]*ToolCall{}
		a.order = map[int][]int{}
		a.resp = ChatResponse{Object: "chat.completion"}
	}
	if a.resp.ID == "" {
		a.resp.ID, a.resp.Created, a.resp.Model = c.ID, c.Created, c.Model
	}
	if c.Usage != nil {
		u := *c.Usage
		a.resp.Usage = &u
	}
	for _, ch := range c.Choices {
		if ch.Index < 0 || ch.Index >= MaxChoices {
			continue
		}
		for len(a.resp.Choices) <= ch.Index {
			a.resp.Choices = append(a.resp.Choices, Choice{Index: len(a.resp.Choices), Message: Message{Role: "assistant"}})
		}
		if ch.Delta.Content != nil {
			sb := a.content[ch.Index]
			if sb == nil {
				sb = &strings.Builder{}
				a.content[ch.Index] = sb
			}
			sb.WriteString(*ch.Delta.Content)
		}
		for i, tc := range ch.Delta.ToolCalls {
			idx := i
			if tc.Index != nil {
				idx = *tc.Index
			}
			if idx < 0 || idx >= 128 {
				continue
			}
			m := a.tools[ch.Index]
			if m == nil {
				m = map[int]*ToolCall{}
				a.tools[ch.Index] = m
			}
			cur := m[idx]
			if cur == nil {
				cur = &ToolCall{Type: "function"}
				m[idx] = cur
				a.order[ch.Index] = append(a.order[ch.Index], idx)
			}
			if tc.ID != "" {
				cur.ID = tc.ID
			}
			if tc.Type != "" {
				cur.Type = tc.Type
			}
			if tc.Function.Name != "" {
				cur.Function.Name = tc.Function.Name
			}
			cur.Function.Arguments += tc.Function.Arguments
		}
		if ch.FinishReason != nil {
			a.resp.Choices[ch.Index].FinishReason = *ch.FinishReason
		}
	}
}

// Response returns the accumulated response.
func (a *Accumulator) Response() *ChatResponse {
	r := a.resp
	r.Choices = append([]Choice(nil), a.resp.Choices...)
	for i := range r.Choices {
		if sb := a.content[i]; sb != nil {
			r.Choices[i].Message.Content = TextContent(sb.String())
		} else if len(a.tools[i]) == 0 {
			r.Choices[i].Message.Content = TextContent("")
		}
		for _, idx := range a.order[i] {
			tc := *a.tools[i][idx]
			r.Choices[i].Message.ToolCalls = append(r.Choices[i].Message.ToolCalls, tc)
		}
	}
	if r.Object == "" {
		r.Object = "chat.completion"
	}
	return &r
}

// ResponseToChunks renders a complete response as a stream: a role chunk,
// one content chunk per choice (plus tool calls), a finish chunk and, when
// includeUsage is set, a trailing usage chunk. Used to replay cache hits.
func ResponseToChunks(r *ChatResponse, includeUsage bool) []*ChatChunk {
	base := func() *ChatChunk {
		return &ChatChunk{ID: r.ID, Object: "chat.completion.chunk", Created: r.Created, Model: r.Model}
	}
	var out []*ChatChunk
	for _, ch := range r.Choices {
		c := base()
		c.Choices = []ChunkChoice{{Index: ch.Index, Delta: Delta{Role: "assistant", Content: Ptr("")}}}
		out = append(out, c)
		if t := ch.Message.Content.Text(); t != "" {
			c := base()
			c.Choices = []ChunkChoice{{Index: ch.Index, Delta: Delta{Content: Ptr(t)}}}
			out = append(out, c)
		}
		if len(ch.Message.ToolCalls) > 0 {
			tcs := make([]ToolCall, len(ch.Message.ToolCalls))
			for i, tc := range ch.Message.ToolCalls {
				tc.Index = Ptr(i)
				tcs[i] = tc
			}
			c := base()
			c.Choices = []ChunkChoice{{Index: ch.Index, Delta: Delta{ToolCalls: tcs}}}
			out = append(out, c)
		}
		c = base()
		fr := ch.FinishReason
		if fr == "" {
			fr = "stop"
		}
		c.Choices = []ChunkChoice{{Index: ch.Index, FinishReason: &fr}}
		out = append(out, c)
	}
	if includeUsage && r.Usage != nil {
		c := base()
		c.Choices = []ChunkChoice{}
		u := *r.Usage
		c.Usage = &u
		out = append(out, c)
	}
	return out
}
