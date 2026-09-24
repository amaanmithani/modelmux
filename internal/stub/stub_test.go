package stub

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHandler(t *testing.T) {
	srv := httptest.NewServer(Handler("a b", time.Millisecond))
	defer srv.Close()
	post := func(body string) string {
		resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var sb strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := resp.Body.Read(buf)
			sb.Write(buf[:n])
			if err != nil {
				break
			}
		}
		return sb.String()
	}
	if out := post(`{"model":"m","messages":[]}`); !strings.Contains(out, `"content":"a b"`) {
		t.Fatal(out)
	}
	out := post(`{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[]}`)
	if !strings.Contains(out, `"content":"a "`) || !strings.Contains(out, `"usage"`) || !strings.HasSuffix(out, "data: [DONE]\n\n") {
		t.Fatal(out)
	}
	if out := post(`{bad`); !strings.Contains(out, "bad json") {
		t.Fatal(out)
	}
}
