# Flash-Next Disk MTP3 — BetterBench methodology

The completed 2026-10-11 campaign used the released Disk image and MTP3 profile
described in [HARDWARE.md](HARDWARE.md). The
[full HTML report](results.html) retains distributions and interactive charts;
[sanitized results.json](results.json) retains BetterBench's measured results,
resolved configuration and sample gate. Download the HTML file to view it.

## Source and workloads

BetterBench **v0.6.0**, unchanged source commit
`d00ad5ec8098c06584a88ec3468bacd37d5ed098`, corpus **1.0** (29 prompts), corpus
hash `e332dceff176d033`. The only configuration overrides selected concurrency
C1/C2/C4 and prefill targets 2000/8000/16000/32000; all other defaults were
retained in the results configuration.

| Phase | Workload | Warmup | Measured requests |
| --- | --- | --- | --- |
| Single stream | All eight categories; prompts cycle in standard corpus order | Three discarded per category | 20 per category, 160 total |
| Prefill | Approximate targets of 2K, 8K, 16K, 32K tokens; fresh body per request | Two discarded per depth | Eight per depth, 32 total |
| Concurrency | Standard flattened corpus order, client semaphore at C1/C2/C4 | No additional phase warmup | 48 per level, 144 total |

All 336 measured requests succeeded; 368 calls including warmups were retained
locally. One existing healthy server served every phase without restart,
profile transition, additional tuning or a repeated phase. Base and historical
runtimes are outside this campaign.

Category and concurrency requests used temperature **0.7**, top_p **0.95** and
top_k **20**, without a request seed override. Standard corpus generation caps
ranged from 120 to 900 tokens. Prefill used the tool's temperature **0** behavior
and a **16-token** generation cap, retaining top_p/top_k settings.

| Category | Standard output-token caps |
| --- | --- |
| Chat | 160, 200, 400 |
| Code | 300, 600, 800 |
| File editing | 250, 500, 600 |
| JSON | 200, 300, 400, 450 |
| Math | 120, 500, 800 |
| Prose | 180, 400, 500, 900 |
| Reasoning | 150, 500, 600, 900 |
| Summarization | 120, 180, 400 |

## Metric definitions and headline weights

The retained source was checked directly: `betterbench/config.py` defines the
defaults; `betterbench/report.py` computes category medians and `_combined`;
`betterbench/client.py` defines request timing; `betterbench/runner.py` defines
concurrency aggregate throughput.

- **TTFT:** client request send to the first text-bearing reasoning or answer
  stream update, using a monotonic clock.
- **Decode tok/s:** `(completion_tokens - 1) / (last_text_update - first_text_update)`.
  The first token belongs to TTFT; reported category rates are medians.
- **Prompt processing:** each request's actual `prompt_tokens / TTFT_seconds`,
  including network and first-token generation time; the table reports the
  median of these rates, not the ratio of two medians.
- **Aggregate output tok/s:** successful completion-token sum divided by the
  full concurrency-level wall time, including TTFT and client scheduling. It
  is not a sum of per-request decode medians.
- **Stream-update gaps:** elapsed time between text-bearing updates. MTP
  updates can carry multiple tokens, so these are not per-token latencies.

The default combined score weights **code 0.30, reasoning 0.20, prose 0.15,
JSON 0.15, file editing 0.10 and summarization 0.10**. Chat and math are measured
with zero weight. The implementation divides the weighted sum of category
decode medians by the sum of included weights (1.0 for this complete run).
The result is **113.96633676142689 tok/s**, displayed as **114.0 tok/s**.

## Context and cache conditions

BetterBench detected the server's **262144-token context profile**, retaining
its standard 256-token margin. Prompt construction approximates four characters
per token; actual token counts come from server usage. Median prefill depths
were **1556.5, 5934.5, 11842 and 23589.5** tokens. Half-token medians average the
two central observations in an eight-request sample. The largest measured
prompt was **23,689 tokens**; prompts did not approach the context ceiling.

The normal nonce-prefix policy remained enabled, and each prefill request used
a fresh body. This prevented reuse of previous requests' prompt-prefix/KV
cache entries while server APC stayed enabled. Campaign counters recorded
zero prefix-cache hits. The OS page cache was retained and mapped PLE pages
could already be resident: **fresh-prefix is not storage-cold**.

## Sample size

- **126/160 category requests (78.8%)** reached the corpus generation cap.
  Generated tokens include reasoning and potentially incomplete answers.
  This campaign measures performance, not answer quality or completion.
- TTFT tails have only **20 observations per category** or **48 per concurrency
  level**; prefill has **eight per depth**. BetterBench's p99 gate requires 500
  observations. Under-sampled percentiles describe roughly the slowest observed
  requests, not reliable population tails. IQR is p75 minus p25.
- Four first-use Triton JIT warnings appeared during concurrency after standard
  warmups. Their latency spikes remain in the measured results; no extra shape
  warmup or phase repeat removed them. Subsequent runs may avoid those costs.
- An external evidence wrapper journaled results after request timing stopped.
  Its small between-request overhead remains in concurrency-level wall time;
  prompts, sampling, token limits, request timing and scoring were unchanged.

## Public evidence scope

The HTML and JSON are byte-for-byte copies of the campaign's prepared
publication artifacts. Host metadata and the endpoint address were removed;
the endpoint label is `local loopback`. Inspection found no private addresses,
filesystem locations, credentials or other private machine identifiers.
The HTML is self-contained and loads no external resources. These hardware
and methodology records are concise summaries of inspected campaign evidence;
unredacted logs, container inspection and local execution paths are excluded.
