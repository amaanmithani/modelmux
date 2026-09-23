// Command modelmux runs the ModelMux LLM gateway.
package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	addr := ":" + envOr("PORT", "8080")
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	log.Printf("modelmux listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
