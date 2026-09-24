// Package stub serves a deterministic OpenAI-compatible chat endpoint.
package stub

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/amaanmithani/modelmux/internal/api"
)

// Handler replies to POST /v1/chat/completions (JSON or SSE) with reply after
// latency. Streaming emits one chunk per word; latency applies before the first.
func Handler(reply string, latency time.Duration) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req api.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":{"message":"bad json"}}`, http.StatusBadRequest)
			return
		}
		if latency > 0 {
			select {
			case <-time.After(latency):
			case <-r.Context().Done():
				return
			}
		}
		words := strings.Fields(reply)
		usage := &api.Usage{PromptTokens: 10, CompletionTokens: len(words), TotalTokens: 10 + len(words)}
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(api.ChatResponse{ID: "chatcmpl-stub", Object: "chat.completion",
				Created: time.Now().Unix(), Model: req.Model, Usage: usage,
				Choices: []api.Choice{{Message: api.Message{Role: "assistant", Content: api.TextContent(reply)}, FinishReason: "stop"}}})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		rc := http.NewResponseController(w)
		send := func(c api.ChatChunk) {
			c.ID, c.Object, c.Model = "chatcmpl-stub", "chat.completion.chunk", req.Model
			b, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", b)
			_ = rc.Flush()
		}
		send(api.ChatChunk{Choices: []api.ChunkChoice{{Delta: api.Delta{Role: "assistant", Content: api.Ptr("")}}}})
		for i, wd := range words {
			if i < len(words)-1 {
				wd += " "
			}
			send(api.ChatChunk{Choices: []api.ChunkChoice{{Delta: api.Delta{Content: api.Ptr(wd)}}}})
		}
		send(api.ChatChunk{Choices: []api.ChunkChoice{{FinishReason: api.Ptr("stop")}}})
		if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
			send(api.ChatChunk{Choices: []api.ChunkChoice{}, Usage: usage})
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		_ = rc.Flush()
	})
	return mux
}
