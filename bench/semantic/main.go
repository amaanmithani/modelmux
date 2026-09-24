// Command semantic measures how the semantic cache's similarity threshold
// trades hit rate for false hits, on labeled Quora question pairs, using the
// same embedding path ModelMux uses at runtime. Writes bench/results/semantic.json.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"strings"

	"github.com/amaanmithani/modelmux/internal/cache"
	"github.com/amaanmithani/modelmux/internal/provider"
)

type pair struct {
	Idx       int    `json:"idx"`
	Q1        string `json:"q1"`
	Q2        string `json:"q2"`
	Duplicate bool   `json:"duplicate"`
}

type row struct {
	Threshold     float64 `json:"threshold"`
	HitRate       float64 `json:"hit_rate_on_duplicates"`
	FalseHitRate  float64 `json:"false_hit_rate_on_non_duplicates"`
	HitPrecision  float64 `json:"precision_of_hits"`
	DuplicateHits int     `json:"duplicate_hits"`
	FalseHits     int     `json:"false_hits"`
}

func main() {
	data := flag.String("data", "bench/data/qqp_sample.jsonl", "pairs (from fetch_qqp.py)")
	base := flag.String("base", "http://localhost:11434/v1", "OpenAI-compatible embeddings endpoint")
	models := flag.String("models", "nomic-embed-text,all-minilm,mxbai-embed-large", "comma-separated embedding models")
	maxFalse := flag.Float64("max-false-hit", 0.01, "false-hit budget used to pick the recommended threshold")
	out := flag.String("out", "bench/results/semantic.json", "output")
	flag.Parse()

	pairs := load(*data)
	emb := provider.NewOpenAI(provider.OpenAIConfig{Name: "embed", BaseURL: *base})
	ctx := context.Background()
	guardOK := make([]bool, len(pairs))
	for i, p := range pairs {
		guardOK[i] = cache.GuardKey(p.Q1) == cache.GuardKey(p.Q2)
	}
	nDup := 0
	for _, p := range pairs {
		if p.Duplicate {
			nDup++
		}
	}
	nNon := len(pairs) - nDup

	var perModel []map[string]any
	for _, model := range strings.Split(*models, ",") {
		sims := embedSims(ctx, emb, model, pairs)
		lat := embedLatency(ctx, emb, model, pairs)
		entry := map[string]any{"model": model, "auc": auc(sims, pairs),
			"embed_latency_ms_p50": lat[len(lat)/2], "embed_latency_ms_p95": lat[len(lat)*95/100]}
		for _, guarded := range []bool{false, true} {
			rows := sweep(sims, pairs, guardOK, guarded, nDup, nNon)
			var rec *row
			for i := range rows {
				if rows[i].FalseHitRate <= *maxFalse {
					rec = &rows[i]
					break
				}
			}
			key := "unguarded"
			if guarded {
				key = "guarded"
			}
			entry[key] = map[string]any{"recommended": rec, "sweep": rows}
			if rec != nil {
				fmt.Fprintf(os.Stderr, "%-18s %-9s τ=%.3f hit=%.3f false=%.3f\n", model, key, rec.Threshold, rec.HitRate, rec.FalseHitRate)
			} else {
				fmt.Fprintf(os.Stderr, "%-18s %-9s no threshold meets the false-hit budget\n", model, key)
			}
		}
		fmt.Fprintf(os.Stderr, "%-18s AUC=%.3f embed p50=%.1fms\n", model, entry["auc"], lat[len(lat)/2])
		perModel = append(perModel, entry)
	}
	res := map[string]any{
		"what": "semantic-cache threshold vs hit rate and false-hit rate, with and without the lexical guard",
		"method": fmt.Sprintf("%d GLUE QQP validation pairs (%d duplicate / %d non-duplicate, seeded sample); "+
			"cosine similarity of embeddings via %s (the runtime path); a hit = similarity >= threshold "+
			"(and, when guarded, cache.GuardKey equal — the same function the cache uses)", len(pairs), nDup, nNon, *base),
		"caveats": []string{
			"QQP labels are crowd-sourced and contain known noise; some 'false hits' are arguably duplicates",
			"non-duplicate QQP pairs are topically related by construction, so false-hit rates are harsher than on random traffic",
			"a false hit returns a cached answer to a different question, so the threshold is chosen for a false-hit budget, not max hit rate",
		},
		"date":           time.Now().Format("2006-01-02"),
		"recommend_rule": fmt.Sprintf("lowest threshold with false-hit rate <= %.3f", *maxFalse),
		"models":         perModel,
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	must(os.WriteFile(*out, append(b, '\n'), 0o644))
}

func embedSims(ctx context.Context, emb *provider.OpenAI, model string, pairs []pair) []float64 {
	sims := make([]float64, len(pairs))
	const batch = 64
	for i := 0; i < len(pairs); i += batch {
		j := min(i+batch, len(pairs))
		var texts []string
		for _, p := range pairs[i:j] {
			texts = append(texts, p.Q1, p.Q2)
		}
		vecs, err := emb.Embed(ctx, model, texts)
		must(err)
		for k := range pairs[i:j] {
			sims[i+k] = cosine(vecs[2*k], vecs[2*k+1])
		}
	}
	return sims
}

// embedLatency times single-text calls, as on the request path.
func embedLatency(ctx context.Context, emb *provider.OpenAI, model string, pairs []pair) []float64 {
	var lat []float64
	for _, p := range pairs[:100] {
		t := time.Now()
		_, err := emb.Embed(ctx, model, []string{p.Q1})
		must(err)
		lat = append(lat, float64(time.Since(t).Microseconds())/1000)
	}
	sort.Float64s(lat)
	return lat
}

func sweep(sims []float64, pairs []pair, guardOK []bool, guarded bool, nDup, nNon int) []row {
	var rows []row
	for t := 0.80; t <= 0.9951; t += 0.005 {
		r := row{Threshold: math.Round(t*1000) / 1000}
		for i, p := range pairs {
			if sims[i] < r.Threshold || (guarded && !guardOK[i]) {
				continue
			}
			if p.Duplicate {
				r.DuplicateHits++
			} else {
				r.FalseHits++
			}
		}
		r.HitRate = float64(r.DuplicateHits) / float64(nDup)
		r.FalseHitRate = float64(r.FalseHits) / float64(nNon)
		if h := r.DuplicateHits + r.FalseHits; h > 0 {
			r.HitPrecision = float64(r.DuplicateHits) / float64(h)
		}
		rows = append(rows, r)
	}
	return rows
}

// auc is the probability a random duplicate pair scores above a random non-duplicate.
func auc(sims []float64, pairs []pair) float64 {
	var d, n []float64
	for i, p := range pairs {
		if p.Duplicate {
			d = append(d, sims[i])
		} else {
			n = append(n, sims[i])
		}
	}
	wins := 0.0
	for _, x := range d {
		for _, y := range n {
			switch {
			case x > y:
				wins++
			case x == y:
				wins += 0.5
			}
		}
	}
	return math.Round(wins/float64(len(d)*len(n))*10000) / 10000
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func load(path string) []pair {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	var out []pair
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var p pair
		must(json.Unmarshal(sc.Bytes(), &p))
		out = append(out, p)
	}
	return out
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
