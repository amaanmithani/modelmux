// Package provider defines the upstream abstraction and its adapters.
package provider

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"

	"github.com/amaanmithani/modelmux/internal/api"
)

// Provider is one upstream model service.
type Provider interface {
	Name() string
	// Chat performs a non-streaming completion. req.Model is the upstream model id.
	Chat(ctx context.Context, req *api.ChatRequest) (*api.ChatResponse, error)
	// Stream starts a streaming completion. It returns once the upstream has
	// accepted the request (HTTP 200); chunks are then read with Recv.
	Stream(ctx context.Context, req *api.ChatRequest) (Stream, error)
}

// Embedder produces embeddings. Used by the semantic cache.
type Embedder interface {
	Embed(ctx context.Context, model string, texts []string) ([][]float32, error)
}

// Stream yields chunks until io.EOF.
type Stream interface {
	Recv() (*api.ChatChunk, error)
	Close() error
}

// UpstreamError is a failed upstream call.
type UpstreamError struct {
	Provider string
	Status   int // 0 for transport errors
	Message  string
	Err      error
}

func (e *UpstreamError) Error() string {
	if e.Status == 0 {
		return fmt.Sprintf("%s: transport error: %v", e.Provider, e.Err)
	}
	return fmt.Sprintf("%s: HTTP %d: %s", e.Provider, e.Status, e.Message)
}

func (e *UpstreamError) Unwrap() error { return e.Err }

// Fallbackable reports whether another target may succeed where this one failed.
// Client-side request errors (400, 422) are not: the same body would fail
// everywhere. Everything provider-specific — rate limits, outages, bad keys,
// unknown model at this provider, timeouts, transport errors — is.
func Fallbackable(err error) bool {
	if err == nil {
		return false
	}
	var ue *UpstreamError
	if errors.As(err, &ue) {
		switch ue.Status {
		case 0:
			return !errors.Is(ue.Err, context.Canceled)
		case http.StatusBadRequest, http.StatusUnprocessableEntity, http.StatusRequestEntityTooLarge:
			return false
		default:
			return ue.Status == 401 || ue.Status == 403 || ue.Status == 404 || ue.Status == 408 ||
				ue.Status == 409 || ue.Status == 425 || ue.Status == 429 || ue.Status >= 500
		}
	}
	if errors.Is(err, context.Canceled) {
		return false
	}
	var ne net.Error
	return errors.Is(err, context.DeadlineExceeded) || errors.As(err, &ne) || errors.Is(err, io.ErrUnexpectedEOF)
}

// readError turns a non-2xx response into an UpstreamError, extracting the
// message from an OpenAI- or Anthropic-shaped error body when present.
func readError(name string, resp *http.Response) error {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	msg := strings.TrimSpace(string(b))
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(b, &env) == nil && env.Error.Message != "" {
		msg = env.Error.Message
	}
	if len(msg) > 500 {
		msg = msg[:500]
	}
	return &UpstreamError{Provider: name, Status: resp.StatusCode, Message: msg}
}

// sseReader reads "data:" payloads from a server-sent-event body.
type sseReader struct {
	sc *bufio.Scanner
}

func newSSEReader(r io.Reader) *sseReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	return &sseReader{sc: sc}
}

// next returns the next event's type and data. For streams that don't name
// events, event is "". Returns io.EOF at end of body.
func (s *sseReader) next() (event string, data []byte, err error) {
	var buf []byte
	have := false
	for s.sc.Scan() {
		line := s.sc.Bytes()
		if len(line) == 0 {
			if have {
				return event, buf, nil
			}
			continue
		}
		if line[0] == ':' {
			continue
		}
		field, value, _ := strings.Cut(string(line), ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event = value
		case "data":
			if have {
				buf = append(buf, '\n')
			}
			buf = append(buf, value...)
			have = true
		}
	}
	if err := s.sc.Err(); err != nil {
		return "", nil, err
	}
	if have {
		return event, buf, nil
	}
	return "", nil, io.EOF
}
