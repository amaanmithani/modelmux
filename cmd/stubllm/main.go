// Command stubllm is a fixed-latency OpenAI-compatible upstream used by the
// benchmarks, so gateway overhead can be measured without model noise.
package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/amaanmithani/modelmux/internal/stub"
)

func main() {
	addr := flag.String("addr", ":9090", "listen address")
	latency := flag.Duration("latency", 50*time.Millisecond, "fixed response latency")
	reply := flag.String("reply", "The quick brown fox jumps over the lazy dog.", "reply text")
	flag.Parse()
	log.Printf("stubllm on %s latency=%s", *addr, *latency)
	srv := &http.Server{Addr: *addr, Handler: stub.Handler(*reply, *latency), ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
