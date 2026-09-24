// Constant-arrival-rate load against one endpoint. Run via bench/overhead.sh.
import http from 'k6/http';
import { check } from 'k6';

const RATE = parseInt(__ENV.RATE || '2000');
const DURATION = __ENV.DURATION || '30s';
const TARGET = __ENV.TARGET; // full URL of /v1/chat/completions
const LABEL = __ENV.LABEL || 'run';
const OUT = __ENV.OUT || 'bench/results/overhead.tmp.json';

export const options = {
  discardResponseBodies: true,
  scenarios: {
    load: {
      executor: 'constant-arrival-rate',
      rate: RATE, timeUnit: '1s', duration: DURATION,
      preAllocatedVUs: 600, maxVUs: 2000,
    },
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const body = JSON.stringify({
  model: __ENV.MODEL || 'bench',
  messages: [{ role: 'user', content: 'Say something short.' }],
});
const params = { headers: { 'Content-Type': 'application/json', Authorization: 'Bearer public' } };

export default function () {
  const res = http.post(TARGET, body, params);
  check(res, { 'status 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  const d = data.metrics.http_req_duration.values;
  const out = {
    label: LABEL, target_rps: RATE, duration: DURATION,
    achieved_rps: data.metrics.http_reqs.values.rate,
    requests: data.metrics.http_reqs.values.count,
    failed_rate: data.metrics.http_req_failed.values.rate,
    dropped_iterations: data.metrics.dropped_iterations ? data.metrics.dropped_iterations.values.count : 0,
    latency_ms: { avg: d.avg, p50: d.med, p90: d['p(90)'], p95: d['p(95)'], p99: d['p(99)'], max: d.max },
  };
  return { [OUT]: JSON.stringify(out, null, 2) };
}
