package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/amaanmithani/modelmux/internal/api"
)

func TestToAnthropicTranslation(t *testing.T) {
	req := &api.ChatRequest{
		Model:       "claude-x",
		Temperature: api.Ptr(1.7),
		Stop:        json.RawMessage(`"END"`),
		ToolChoice:  json.RawMessage(`"required"`),
		Tools: []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "get_weather", Description: "d",
			Parameters: json.RawMessage(`{"type":"object"}`)}}, {Type: "function", Function: api.ToolFunction{Name: "noargs"}}},
		Messages: []api.Message{
			{Role: "system", Content: api.TextContent("be brief")},
			{Role: "developer", Content: api.TextContent("and kind")},
			{Role: "user", Content: api.TextContent("weather?")},
			{Role: "assistant", Content: api.TextContent("checking"), ToolCalls: []api.ToolCall{
				{ID: "tu1", Type: "function", Function: api.FunctionCall{Name: "get_weather", Arguments: `{"city":"Pune"}`}},
				{ID: "tu2", Type: "function", Function: api.FunctionCall{Name: "noargs", Arguments: `not json`}},
			}},
			{Role: "tool", ToolCallID: "tu1", Content: api.TextContent("31C")},
			{Role: "tool", ToolCallID: "tu2", Content: api.TextContent("ok")},
			{Role: "user", Content: api.TextContent("thanks")},
		},
	}
	ar, err := toAnthropic(req)
	if err != nil {
		t.Fatal(err)
	}
	if ar.System != "be brief\n\nand kind" || ar.MaxTokens != defaultAnthropicMaxTokens || *ar.Temperature != 1 {
		t.Fatalf("header fields: %+v", ar)
	}
	if len(ar.StopSequences) != 1 || ar.StopSequences[0] != "END" {
		t.Fatalf("stop: %v", ar.StopSequences)
	}
	// user, assistant, user(tool_result, tool_result, text) — consecutive user turns merged.
	if len(ar.Messages) != 3 {
		t.Fatalf("want 3 alternating messages, got %d: %+v", len(ar.Messages), ar.Messages)
	}
	asst := ar.Messages[1]
	if asst.Content[1].Type != "tool_use" || string(asst.Content[1].Input) != `{"city":"Pune"}` || string(asst.Content[2].Input) != "{}" {
		t.Fatalf("assistant blocks: %+v", asst.Content)
	}
	last := ar.Messages[2]
	if last.Role != "user" || len(last.Content) != 3 || last.Content[0].Type != "tool_result" || last.Content[0].ToolUseID != "tu1" ||
		last.Content[2].Text != "thanks" {
		t.Fatalf("merged user turn: %+v", last)
	}
	if ar.Tools[1].InputSchema == nil || ar.ToolChoice.(map[string]string)["type"] != "any" {
		t.Fatalf("tools: %+v choice %v", ar.Tools, ar.ToolChoice)
	}
}

func TestToAnthropicToolChoiceAndErrors(t *testing.T) {
	base := func(choice string) *api.ChatRequest {
		r := userReq("m", "x")
		r.ToolChoice = json.RawMessage(choice)
		return r
	}
	for in, want := range map[string]string{`"auto"`: "auto", `"none"`: "none",
		`{"type":"function","function":{"name":"f"}}`: "tool"} {
		ar, err := toAnthropic(base(in))
		if err != nil || ar.ToolChoice.(map[string]string)["type"] != want {
			t.Errorf("tool_choice %s: %v %v", in, ar.ToolChoice, err)
		}
	}
	if _, err := toAnthropic(&api.ChatRequest{Messages: []api.Message{{Role: "system", Content: api.TextContent("x")}}}); err == nil {
		t.Fatal("system-only request should fail")
	}
	if _, err := toAnthropic(&api.ChatRequest{Messages: []api.Message{{Role: "wizard"}}}); err == nil {
		t.Fatal("unknown role should fail")
	}
	r := userReq("m", "x")
	r.MaxCompletionTokens = api.Ptr(77)
	if ar, _ := toAnthropic(r); ar.MaxTokens != 77 {
		t.Fatal("max tokens")
	}
}

func TestFinishReason(t *testing.T) {
	for in, want := range map[string]string{"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
		"tool_use": "tool_calls", "refusal": "content_filter"} {
		if got := finishReason(in); got != want {
			t.Errorf("%s -> %s want %s", in, got, want)
		}
	}
}

func anthServer(t *testing.T, h http.HandlerFunc) *Anthropic {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Header.Get("x-api-key") != "k" || r.Header.Get("anthropic-version") != AnthropicVersion {
			t.Errorf("bad request: %s %v", r.URL.Path, r.Header)
		}
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	return NewAnthropic(AnthropicConfig{Name: "anth", BaseURL: srv.URL + "/v1", APIKey: "k"})
}

func TestAnthropicChat(t *testing.T) {
	p := anthServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body anthReq
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.Stream {
			t.Error("non-stream call sent stream=true")
		}
		fmt.Fprint(w, `{"id":"msg_1","model":"claude-x","stop_reason":"tool_use","usage":{"input_tokens":12,"output_tokens":5},
			"content":[{"type":"text","text":"Let me check."},{"type":"tool_use","id":"tu1","name":"get","input":{"a":1}}]}`)
	})
	if p.Name() != "anth" {
		t.Fatal("name")
	}
	resp, err := p.Chat(context.Background(), userReq("claude-x", "hi"))
	if err != nil {
		t.Fatal(err)
	}
	c := resp.Choices[0]
	if resp.ID != "chatcmpl-msg_1" || c.Message.Content.Text() != "Let me check." || c.FinishReason != "tool_calls" ||
		c.Message.ToolCalls[0].Function.Arguments != `{"a":1}` || resp.Usage.TotalTokens != 17 {
		t.Fatalf("resp: %+v", resp)
	}
}

func TestAnthropicChatErrors(t *testing.T) {
	p := anthServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(529)
		fmt.Fprint(w, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	})
	_, err := p.Chat(context.Background(), userReq("m", "hi"))
	var ue *UpstreamError
	if !errors.As(err, &ue) || ue.Status != 529 || ue.Message != "Overloaded" || !Fallbackable(err) {
		t.Fatalf("got %v", err)
	}
	bad := &api.ChatRequest{Messages: []api.Message{{Role: "system", Content: api.TextContent("only")}}}
	if _, err := p.Chat(context.Background(), bad); Fallbackable(err) {
		t.Fatal("untranslatable request must not fall back")
	}
	if _, err := p.Stream(context.Background(), bad); Fallbackable(err) {
		t.Fatal("untranslatable request must not fall back (stream)")
	}
}

const anthStreamBody = `event: message_start
data: {"type":"message_start","message":{"id":"msg_2","model":"claude-x","usage":{"input_tokens":9,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu9","name":"lookup","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"go\"}"}}

event: content_block_delta
data: {"type":"content_block_delta","index":7,"delta":{"type":"input_json_delta","partial_json":"orphan"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":14}}

event: message_stop
data: {"type":"message_stop"}

`

func TestAnthropicStream(t *testing.T) {
	p := anthServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, anthStreamBody)
	})
	s, err := p.Stream(context.Background(), userReq("claude-x", "hi"))
	if err != nil {
		t.Fatal(err)
	}
	got := drain(t, s)
	c := got.Choices[0]
	if c.Message.Content.Text() != "Hi there" || c.FinishReason != "tool_calls" {
		t.Fatalf("choice: %+v", c)
	}
	if len(c.Message.ToolCalls) != 1 || c.Message.ToolCalls[0].ID != "tu9" || c.Message.ToolCalls[0].Function.Arguments != `{"q":"go"}` {
		t.Fatalf("tool calls: %+v", c.Message.ToolCalls)
	}
	if got.Usage.PromptTokens != 9 || got.Usage.CompletionTokens != 14 || got.ID != "chatcmpl-msg_2" {
		t.Fatalf("usage/id: %+v %s", got.Usage, got.ID)
	}
}

func TestAnthropicStreamErrors(t *testing.T) {
	cases := map[string]struct {
		body   string
		status int
	}{
		"overloaded": {"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n", 529},
		"api error":  {"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"x\"}}\n\n", 502},
		"truncated":  {"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\"}}\n\n", 0},
		"bad json":   {"data: {oops\n\n", 0},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			p := anthServer(t, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, tc.body) })
			s, err := p.Stream(context.Background(), userReq("m", "x"))
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			var last error
			for i := 0; i < 5 && last == nil; i++ {
				_, last = s.Recv()
			}
			var ue *UpstreamError
			if !errors.As(last, &ue) || ue.Status != tc.status {
				t.Fatalf("got %v", last)
			}
			if last == io.EOF {
				t.Fatal("EOF is not an error")
			}
		})
	}
}

func TestAnthropicDefaults(t *testing.T) {
	p := NewAnthropic(AnthropicConfig{Name: "a"})
	if !strings.HasPrefix(p.baseURL, "https://api.anthropic.com") || p.client == nil {
		t.Fatal("defaults")
	}
}
