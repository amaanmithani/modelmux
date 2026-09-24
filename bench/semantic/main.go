// Command semantic measures how the semantic cache's similarity threshold
// trades hit rate for wrong answers, on labeled Quora question pairs, using
// the same embedding path and guard the cache uses at runtime.
//
// The threshold is chosen on a calibration split and every reported number
// comes from a separate held-out split. Two views are reported:
//   - pairwise: does each question pair score above the threshold?
//   - cache simulation: index every held-out q1, query with every q2, take the
//     nearest neighbour as the cache would. This is what a user experiences:
//     a lookup has many candidates, so it has more chances to go wrong than a
//     single pair does.
//
// Writes bench/results/semantic.json.
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
	"strings"
	"time"

	"github.com/amaanmithani/modelmux/internal/cache"
	"github.com/amaanmithani/modelmux/internal/provider"
)

type pair struct {
	Idx       int    `json:"idx"`
	Q1        string `json:"q1"`
	Q2        string `json:"q2"`
	Duplicate bool   `json:"duplicate"`
}

type rate struct {
	K    int     `json:"k"`
	N    int     `json:"n"`
	Rate float64 `json:"rate"`
	Lo95 float64 `json:"ci95_lo"`
	Hi95 float64 `json:"ci95_hi"`
}

// wilson returns the Wilson score interval for k successes in n trials.
func wilson(k, n int) rate {
	if n == 0 {
		return rate{}
	}
	const z = 1.96
	p := float64(k) / float64(n)
	den := 1 + z*z/float64(n)
	c := (p + z*z/(2*float64(n))) / den
	h := z * math.Sqrt(p*(1-p)/float64(n)+z*z/(4*float64(n)*float64(n))) / den
	return rate{K: k, N: n, Rate: round4(p), Lo95: round4(math.Max(0, c-h)), Hi95: round4(math.Min(1, c+h))}
}

func round4(x float64) float64 { return math.Round(x*10000) / 10000 }

type split struct {
	pairs  []pair
	v1, v2 [][]float32 // normalised embeddings of q1 and q2
	sims   []float64
	guard  []bool // GuardKey(q1) == GuardKey(q2)
}

type modeResult struct {
	Threshold        float64 `json:"threshold_chosen_on_calibration"`
	CalibFalseHit    float64 `json:"calibration_false_hit_rate"`
	HeldoutHit       rate    `json:"heldout_pairwise_hit_rate_on_duplicates"`
	HeldoutFalseHit  rate    `json:"heldout_pairwise_false_hit_rate_on_non_duplicates"`
	CacheCorrectHits rate    `json:"cache_sim_correct_answers_for_duplicate_queries"`
	CacheWrong       rate    `json:"cache_sim_wrong_answers_over_all_queries"`
}

func main() {
	calibPath := flag.String("calib", "bench/data/qqp_calib.jsonl", "calibration pairs (fetch_qqp.py)")
	heldPath := flag.String("heldout", "bench/data/qqp_heldout.jsonl", "held-out pairs")
	base := flag.String("base", "http://localhost:11434/v1", "OpenAI-compatible embeddings endpoint")
	models := flag.String("models", "nomic-embed-text,all-minilm,mxbai-embed-large", "comma-separated embedding models")
	budget := flag.Float64("max-false-hit", 0.01, "false-hit budget for choosing the threshold (on calibration)")
	out := flag.String("out", "bench/results/semantic.json", "output")
	flag.Parse()

	emb := provider.NewOpenAI(provider.OpenAIConfig{Name: "embed", BaseURL: *base})
	ctx := context.Background()
	calibPairs, heldPairs := load(*calibPath), load(*heldPath)

	var perModel []map[string]any
	for _, model := range strings.Split(*models, ",") {
		calib := embedSplit(ctx, emb, model, calibPairs)
		held := embedSplit(ctx, emb, model, heldPairs)
		lat := embedLatency(ctx, emb, model, heldPairs)
		entry := map[string]any{
			"model":                model,
			"heldout_auc":          auc(held),
			"embed_latency_ms_p50": lat[len(lat)/2],
			"embed_latency_ms_p95": lat[len(lat)*95/100],
			"top_non_duplicates":   topNonDuplicates(held, 3),
		}
		for _, guarded := range []bool{false, true} {
			key := map[bool]string{false: "unguarded", true: "guarded"}[guarded]
			tau, calibFalse, ok := choose(calib, guarded, *budget)
			if !ok {
				entry[key] = map[string]any{"threshold_chosen_on_calibration": nil,
					"note": "no threshold meets the false-hit budget on the calibration split"}
				fmt.Fprintf(os.Stderr, "%-18s %-9s no threshold qualifies on calibration\n", model, key)
				continue
			}
			r := modeResult{Threshold: tau, CalibFalseHit: round4(calibFalse)}
			r.HeldoutHit, r.HeldoutFalseHit = pairwise(held, guarded, tau)
			r.CacheCorrectHits, r.CacheWrong = cacheSim(held, guarded, tau)
			entry[key] = r
			fmt.Fprintf(os.Stderr, "%-18s %-9s τ=%.3f held-out hit %.3f [%.3f,%.3f] false %.3f [%.3f,%.3f] | cache correct %.3f wrong %.3f\n",
				model, key, tau, r.HeldoutHit.Rate, r.HeldoutHit.Lo95, r.HeldoutHit.Hi95, r.HeldoutFalseHit.Rate,
				r.HeldoutFalseHit.Lo95, r.HeldoutFalseHit.Hi95, r.CacheCorrectHits.Rate, r.CacheWrong.Rate)
		}
		perModel = append(perModel, entry)
	}
	res := map[string]any{
		"what": "semantic-cache threshold vs hits and wrong answers, with and without the lexical guard",
		"method": fmt.Sprintf("GLUE QQP validation, seeded balanced samples: %d calibration pairs choose the threshold "+
			"(lowest τ with pairwise false-hit rate <= %.3f), %d held-out pairs are used for every reported number. "+
			"Embeddings via %s (the runtime path); guard = cache.GuardKey (the runtime function). "+
			"Cache simulation: all held-out q1 in one scope, each q2 looks up its nearest neighbour.",
			len(calibPairs), *budget, len(heldPairs), *base),
		"caveats": []string{
			"QQP labels are crowd-sourced and noisy; some 'wrong answers' are arguably duplicates",
			"the samples are 50% duplicates; real traffic's duplicate rate changes precision",
			"95% intervals are Wilson score intervals on the held-out counts",
		},
		"date":   time.Now().Format("2006-01-02"),
		"models": perModel,
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	must(os.WriteFile(*out, append(b, '\n'), 0o644))
}

func embedSplit(ctx context.Context, emb *provider.OpenAI, model string, pairs []pair) *split {
	s := &split{pairs: pairs}
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
			s.v1 = append(s.v1, normalize(vecs[2*k]))
			s.v2 = append(s.v2, normalize(vecs[2*k+1]))
		}
	}
	for i, p := range pairs {
		s.sims = append(s.sims, dot(s.v1[i], s.v2[i]))
		s.guard = append(s.guard, cache.GuardKey(p.Q1) == cache.GuardKey(p.Q2))
	}
	return s
}

// choose picks the lowest threshold whose pairwise false-hit rate on the
// calibration split is within budget.
func choose(s *split, guarded bool, budget float64) (float64, float64, bool) {
	for t := 0.800; t <= 0.9951; t += 0.005 {
		t = math.Round(t*1000) / 1000
		_, f := pairwise(s, guarded, t)
		if f.Rate <= budget {
			return t, f.Rate, true
		}
	}
	return 0, 0, false
}

func pairwise(s *split, guarded bool, t float64) (hit, falseHit rate) {
	dupHits, dups, falseHits, nons := 0, 0, 0, 0
	for i, p := range s.pairs {
		pass := s.sims[i] >= t && (!guarded || s.guard[i])
		if p.Duplicate {
			dups++
			if pass {
				dupHits++
			}
		} else {
			nons++
			if pass {
				falseHits++
			}
		}
	}
	return wilson(dupHits, dups), wilson(falseHits, nons)
}

// cacheSim puts every q1 in one cache scope and looks up every q2's nearest
// neighbour, as the runtime does. An answer is correct only when the match is
// the query's own pair and that pair is a labeled duplicate.
func cacheSim(s *split, guarded bool, t float64) (correct, wrong rate) {
	keys := make([]string, len(s.pairs))
	for i, p := range s.pairs {
		keys[i] = cache.GuardKey(p.Q1)
	}
	good, dups, bad := 0, 0, 0
	for qi, p := range s.pairs {
		gk := cache.GuardKey(p.Q2)
		best, bestSim := -1, -1.0
		for ei := range s.pairs {
			if guarded && keys[ei] != gk {
				continue
			}
			if sim := dot(s.v2[qi], s.v1[ei]); sim > bestSim {
				best, bestSim = ei, sim
			}
		}
		if p.Duplicate {
			dups++
		}
		if best < 0 || bestSim < t {
			continue
		}
		if best == qi && p.Duplicate {
			good++
		} else {
			bad++
		}
	}
	return wilson(good, dups), wilson(bad, len(s.pairs))
}

func topNonDuplicates(s *split, n int) []map[string]any {
	type cand struct {
		i   int
		sim float64
	}
	var cs []cand
	for i, p := range s.pairs {
		if !p.Duplicate {
			cs = append(cs, cand{i, s.sims[i]})
		}
	}
	sort.Slice(cs, func(a, b int) bool { return cs[a].sim > cs[b].sim })
	var out []map[string]any
	for _, c := range cs[:min(n, len(cs))] {
		out = append(out, map[string]any{"q1": s.pairs[c.i].Q1, "q2": s.pairs[c.i].Q2, "cosine": round4(c.sim),
			"guard_blocks_it": !s.guard[c.i]})
	}
	return out
}

// auc is the probability a random duplicate pair scores above a random non-duplicate.
func auc(s *split) float64 {
	var d, n []float64
	for i, p := range s.pairs {
		if p.Duplicate {
			d = append(d, s.sims[i])
		} else {
			n = append(n, s.sims[i])
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
	return round4(wins / float64(len(d)*len(n)))
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

func normalize(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	inv := 1 / math.Sqrt(n)
	for i, x := range v {
		out[i] = float32(float64(x) * inv)
	}
	return out
}

// dot matches the runtime cache: float32 accumulation over unit vectors.
func dot(a, b []float32) float64 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return float64(s)
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
