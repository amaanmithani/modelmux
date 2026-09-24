package api

import (
	"encoding/json"
	"testing"
)

func TestContentTextForms(t *testing.T) {
	cases := map[string]string{
		`"hello"`: "hello",
		`[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"b"}]`: "ab",
		`null`: "",
		`42`:   "",
	}
	for in, want := range cases {
		var c Content
		if err := json.Unmarshal([]byte(in), &c); err != nil {
			t.Fatal(err)
		}
		if got := c.Text(); got != want {
			t.Errorf("Text(%s) = %q, want %q", in, got, want)
		}
	}
	if !(Content{}).IsNull() {
		t.Error("zero Content should be null")
	}
}

func TestContentRoundTripPreservesParts(t *testing.T) {
	in := `{"role":"user","content":[{"type":"text","text":"hi"}]}`
	var m Message
	if err := json.Unmarshal([]byte(in), &m); err != nil {
		t.Fatal(err)
	}
	out, _ := json.Marshal(m)
	if string(out) != in {
		t.Fatalf("round trip changed content:\n got %s\nwant %s", out, in)
	}
	empty, _ := json.Marshal(Message{Role: "assistant"})
	if string(empty) != `{"role":"assistant","content":null}` {
		t.Fatalf("null content: %s", empty)
	}
}

func TestStopSequences(t *testing.T) {
	for in, want := range map[string]int{`"x"`: 1, `["a","b"]`: 2, `""`: 0, ``: 0} {
		r := ChatRequest{Stop: json.RawMessage(in)}
		if got := len(r.StopSequences()); got != want {
			t.Errorf("Stop %q: got %d want %d", in, got, want)
		}
	}
}

func TestMaxOutputTokensPrefersNewField(t *testing.T) {
	r := ChatRequest{MaxTokens: Ptr(10)}
	if r.MaxOutputTokens() != 10 {
		t.Fatal("max_tokens ignored")
	}
	r.MaxCompletionTokens = Ptr(20)
	if r.MaxOutputTokens() != 20 {
		t.Fatal("max_completion_tokens should win")
	}
	if (&ChatRequest{}).MaxOutputTokens() != 0 {
		t.Fatal("unset should be 0")
	}
}

func TestCloneIsIndependent(t *testing.T) {
	r := &ChatRequest{Model: "a", Messages: []Message{{Role: "user"}}}
	c := r.Clone()
	c.Model = "b"
	c.Messages[0].Role = "system"
	if r.Model != "a" || r.Messages[0].Role != "user" {
		t.Fatal("clone aliased the original")
	}
}

func TestAccumulatorRebuildsTextAndTools(t *testing.T) {
	var a Accumulator
	a.Add(&ChatChunk{ID: "id1", Model: "m", Created: 5, Choices: []ChunkChoice{{Delta: Delta{Role: "assistant", Content: Ptr("")}}}})
	a.Add(&ChatChunk{Choices: []ChunkChoice{{Delta: Delta{Content: Ptr("Hel")}}}})
	a.Add(&ChatChunk{Choices: []ChunkChoice{{Delta: Delta{Content: Ptr("lo")}}}})
	a.Add(&ChatChunk{Choices: []ChunkChoice{{Delta: Delta{ToolCalls: []ToolCall{{Index: Ptr(0), ID: "t1", Type: "function",
		Function: FunctionCall{Name: "get"}}}}}}})
	a.Add(&ChatChunk{Choices: []ChunkChoice{{Delta: Delta{ToolCalls: []ToolCall{{Index: Ptr(0), Function: FunctionCall{Arguments: `{"a":`}}}}}}})
	a.Add(&ChatChunk{Choices: []ChunkChoice{{Delta: Delta{ToolCalls: []ToolCall{{Index: Ptr(0), Function: FunctionCall{Arguments: `1}`}}}}}}})
	a.Add(&ChatChunk{Choices: []ChunkChoice{{FinishReason: Ptr("tool_calls")}}})
	a.Add(&ChatChunk{Choices: []ChunkChoice{}, Usage: &Usage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7}})
	r := a.Response()
	if r.ID != "id1" || r.Model != "m" || r.Object != "chat.completion" {
		t.Fatalf("header fields: %+v", r)
	}
	c := r.Choices[0]
	if c.Message.Content.Text() != "Hello" || c.FinishReason != "tool_calls" {
		t.Fatalf("choice: %+v", c)
	}
	if len(c.Message.ToolCalls) != 1 || c.Message.ToolCalls[0].Function.Arguments != `{"a":1}` || c.Message.ToolCalls[0].ID != "t1" {
		t.Fatalf("tool calls: %+v", c.Message.ToolCalls)
	}
	if r.Usage.TotalTokens != 7 {
		t.Fatalf("usage: %+v", r.Usage)
	}
}

func TestResponseToChunksRoundTrips(t *testing.T) {
	resp := &ChatResponse{ID: "x", Model: "m", Choices: []Choice{{Message: Message{Role: "assistant", Content: TextContent("hi there"),
		ToolCalls: []ToolCall{{ID: "t", Type: "function", Function: FunctionCall{Name: "f", Arguments: "{}"}}}}, FinishReason: "tool_calls"}},
		Usage: &Usage{TotalTokens: 2}}
	chunks := ResponseToChunks(resp, true)
	if chunks[len(chunks)-1].Usage == nil {
		t.Fatal("usage chunk missing")
	}
	var a Accumulator
	for _, c := range chunks {
		a.Add(c)
	}
	got := a.Response()
	if got.Choices[0].Message.Content.Text() != "hi there" || got.Choices[0].FinishReason != "tool_calls" ||
		got.Choices[0].Message.ToolCalls[0].Function.Name != "f" {
		t.Fatalf("round trip: %+v", got.Choices[0])
	}
	if n := len(ResponseToChunks(resp, false)); n != len(chunks)-1 {
		t.Fatalf("without usage: %d chunks", n)
	}
	empty := ResponseToChunks(&ChatResponse{Choices: []Choice{{}}}, false)
	if *empty[len(empty)-1].Choices[0].FinishReason != "stop" {
		t.Fatal("default finish reason should be stop")
	}
}

func TestNewError(t *testing.T) {
	e := NewError("t", "c", "m")
	if *e.Error.Code != "c" || e.Error.Message != "m" {
		t.Fatal(e)
	}
	if NewError("t", "", "m").Error.Code != nil {
		t.Fatal("empty code should be null")
	}
}
