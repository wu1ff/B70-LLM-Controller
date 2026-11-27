# Stock BetterBench 0.6.1 comparison

Corpus 1.0 on the same four Intel Arc Pro B70s. SGLang uses the accepted A0042 image. vLLM 0.33 DFlash2 v6 uses the two retained stock runs.

## Run IDs and sources

Paths beginning with `docs/` are relative to the campaign workspace. Times are as recorded by BetterBench (`−07:00`, America/Phoenix). The vLLM session ID uses a UTC date.

| ID | Engine and run | Recorded start | Session | Metric source |
| --- | --- | --- | --- | --- |
| S1 | SGLang A0042 run 1 | `2026-11-21T21:14:50-0700` | transition | `docs/evidence/0042/transition/stock/summary.json` |
| S2 | SGLang A0042 run 2 | `2026-11-21T21:44:41-0700` | restored after restart and warmup | `docs/evidence/0042/restored/stock/summary.json` |
| V1 | vLLM 0.33 DFlash2 v6 retained run 1 | `2026-11-16T17:59:19-0700` | betterbench-thinking-20261117T005238Z-29fc51/run-1 | `/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v033/release-prep/betterbench-thinking-20261117T005238Z-29fc51/run-1/summary.json` |
| V2 | vLLM 0.33 DFlash2 v6 retained run 2 | `2026-11-16T18:18:41-0700` | betterbench-thinking-20261117T005238Z-29fc51/run-2 | `/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v033/release-prep/betterbench-thinking-20261117T005238Z-29fc51/run-2/summary.json` |

## Full comparison

Each mean is the arithmetic mean of that engine’s two unrounded run values. Display rounding follows the calculation. The reserved phrase **not retained** marks an unavailable scored value or a mean missing a scored run. Exact files, field names, and supporting counts are in [SOURCE-MAP.md](SOURCE-MAP.md).

TTFT is time to first token, which can be reasoning. Decode excludes the first token. Aggregate output includes admission through the last completion. Stream-update gaps can span multiple tokens arriving together. **†** marks recorded tails below BetterBench’s sample gate. All per-token gap-derived rates are suppressed because updates carry multiple tokens. The restored SGLang prose first-answer median is unavailable because only four requests reached an answer, below the stock minimum of five.

| Metric | Category / band | S1 | S2 | SGLang mean | V1 retained | V2 retained | vLLM mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Weighted single-stream** |  |  |  |  |  |  |  |
| Weighted decode (tok/s) | All weighted categories | 199.4704 | 199.9362 | 199.7033 | 178.5671 | 183.8494 | 181.2082 |
| Weighted first token (ms) | All weighted categories | 95.8819 | 95.9883 | 95.9351 | 131.1753 | 129.1942 | 130.1848 |
| Weighted update p50 (ms) | All weighted categories | 18.65 | 18.56 | 18.61 | 20.73 | 20.72 | 20.72 |
| Weighted update p99 (ms) | All weighted categories | 20.19 | 20.17 | 20.18 | 22.04 | 22.00 | 22.02 |
| Weighted low-1% per-token rate (tok/s) | All weighted categories | not retained | not retained | not retained | not retained | not retained | not retained |
| **Single-stream categories** |  |  |  |  |  |  |  |
| Median decode (tok/s) | chat | 157.5747 | 149.9687 | 153.7717 | 129.4776 | 131.3490 | 130.4133 |
| Median decode (tok/s) | code | 201.1343 | 192.7367 | 196.9355 | 180.0908 | 190.2698 | 185.1803 |
| Median decode (tok/s) | file_edit | 233.4970 | 243.1081 | 238.3026 | 194.8864 | 195.6867 | 195.2866 |
| Median decode (tok/s) | json | 264.0008 | 268.9789 | 266.4899 | 244.6519 | 234.5025 | 239.5772 |
| Median decode (tok/s) | math | 266.8501 | 268.2479 | 267.5490 | 238.8751 | 243.6612 | 241.2682 |
| Median decode (tok/s) | prose | 152.2297 | 151.7916 | 152.0106 | 135.6398 | 140.3637 | 138.0017 |
| Median decode (tok/s) | reasoning | 142.6493 | 149.5455 | 146.0974 | 131.3426 | 142.9353 | 137.1390 |
| Median decode (tok/s) | summarization | 248.1596 | 247.7967 | 247.9782 | 217.3896 | 223.8279 | 220.6087 |
| Median first token (ms) | chat | 111.4563 | 113.3657 | 112.4110 | 130.5942 | 127.1123 | 128.8532 |
| Median first token (ms) | code | 89.6900 | 89.9037 | 89.7969 | 130.7167 | 128.7796 | 129.7481 |
| Median first token (ms) | file_edit | 109.8085 | 109.9938 | 109.9011 | 133.0363 | 127.4999 | 130.2681 |
| Median first token (ms) | json | 90.0443 | 90.4496 | 90.2469 | 132.5545 | 125.9093 | 129.2319 |
| Median first token (ms) | math | 88.4252 | 89.3133 | 88.8693 | 134.4613 | 129.3359 | 131.8986 |
| Median first token (ms) | prose | 90.9338 | 91.3919 | 91.1629 | 128.1433 | 127.9057 | 128.0245 |
| Median first token (ms) | reasoning | 99.0351 | 99.3441 | 99.1896 | 132.7951 | 133.3935 | 133.0943 |
| Median first token (ms) | summarization | 110.4031 | 108.7274 | 109.5653 | 129.9298 | 130.5941 | 130.2620 |
| Stream-update gap p50 (ms) | chat | 18.71 | 18.64 | 18.68 | 20.78 | 20.71 | 20.75 |
| Stream-update gap p50 (ms) | code | 18.75 | 18.62 | 18.68 | 20.75 | 20.76 | 20.75 |
| Stream-update gap p50 (ms) | file_edit | 18.57 | 18.52 | 18.54 | 20.79 | 20.76 | 20.78 |
| Stream-update gap p50 (ms) | json | 18.17 | 18.16 | 18.17 | 20.41 | 20.39 | 20.40 |
| Stream-update gap p50 (ms) | math | 18.36 | 18.38 | 18.37 | 20.65 | 20.64 | 20.65 |
| Stream-update gap p50 (ms) | prose | 18.71 | 18.59 | 18.65 | 20.78 | 20.76 | 20.77 |
| Stream-update gap p50 (ms) | reasoning | 18.95 | 18.85 | 18.90 | 20.91 | 20.91 | 20.91 |
| Stream-update gap p50 (ms) | summarization | 18.50 | 18.41 | 18.46 | 20.62 | 20.63 | 20.63 |
| Stream-update gap p99 (ms) | chat | 20.33 | 20.29 | 20.31 | 22.07 | 22.00 | 22.04 |
| Stream-update gap p99 (ms) | code | 20.33 | 20.39 | 20.36 | 21.99 | 21.90 | 21.94 |
| Stream-update gap p99 (ms) | file_edit | 20.22 | 20.09 | 20.16 | 22.07 | 21.86 | 21.96 |
| Stream-update gap p99 (ms) | json | 19.77 | 19.73 | 19.75 | 22.18 | 21.85 | 22.01 |
| Stream-update gap p99 (ms) | math | 20.43 | 20.29 | 20.36 | 22.22 | 22.01 | 22.11 |
| Stream-update gap p99 (ms) | prose | 20.09 | 20.10 | 20.09 | 21.94 | 21.89 | 21.91 |
| Stream-update gap p99 (ms) | reasoning | 20.41 | 20.35 | 20.38 | 22.09 | 22.20 | 22.15 |
| Stream-update gap p99 (ms) | summarization | 20.09 | 20.00 | 20.05 | 22.03 | 22.43 | 22.23 |
| Tokens per stream update | chat | 2.99 | 2.91 | 2.95 | 2.72 | 2.88 | 2.80 |
| Tokens per stream update | code | 3.39 | 3.32 | 3.35 | 3.39 | 3.42 | 3.41 |
| Tokens per stream update | file_edit | 4.12 | 4.13 | 4.12 | 3.91 | 3.90 | 3.90 |
| Tokens per stream update | json | 4.70 | 4.74 | 4.72 | 4.95 | 4.68 | 4.81 |
| Tokens per stream update | math | 4.79 | 4.65 | 4.72 | 4.80 | 4.67 | 4.73 |
| Tokens per stream update | prose | 2.76 | 2.76 | 2.76 | 2.73 | 2.75 | 2.74 |
| Tokens per stream update | reasoning | 2.68 | 2.60 | 2.64 | 2.78 | 2.78 | 2.78 |
| Tokens per stream update | summarization | 4.23 | 4.29 | 4.26 | 4.37 | 4.52 | 4.44 |
| First-token p99 (ms) † | chat | 137.92 | 139.83 | 138.88 | 151.97 | 135.58 | 143.78 |
| First-token p99 (ms) † | code | 126.99 | 125.09 | 126.04 | 137.74 | 133.80 | 135.77 |
| First-token p99 (ms) † | file_edit | 138.23 | 135.33 | 136.78 | 146.45 | 132.27 | 139.36 |
| First-token p99 (ms) † | json | 135.58 | 134.98 | 135.28 | 142.62 | 133.74 | 138.18 |
| First-token p99 (ms) † | math | 136.71 | 115.70 | 126.20 | 197.12 | 175.70 | 186.41 |
| First-token p99 (ms) † | prose | 118.96 | 111.83 | 115.39 | 146.16 | 134.26 | 140.21 |
| First-token p99 (ms) † | reasoning | 132.09 | 144.89 | 138.49 | 151.26 | 152.76 | 152.01 |
| First-token p99 (ms) † | summarization | 136.48 | 133.16 | 134.82 | 137.92 | 149.85 | 143.88 |
| Decode interquartile range (tok/s) | chat | 55.17 | 37.73 | 46.45 | 42.45 | 54.28 | 48.36 |
| Decode interquartile range (tok/s) | code | 63.77 | 81.38 | 72.58 | 61.97 | 67.46 | 64.72 |
| Decode interquartile range (tok/s) | file_edit | 43.75 | 87.83 | 65.79 | 53.80 | 72.37 | 63.09 |
| Decode interquartile range (tok/s) | json | 37.62 | 47.73 | 42.67 | 36.56 | 33.19 | 34.87 |
| Decode interquartile range (tok/s) | math | 36.80 | 37.10 | 36.95 | 48.63 | 41.65 | 45.14 |
| Decode interquartile range (tok/s) | prose | 53.53 | 51.33 | 52.43 | 46.57 | 52.85 | 49.71 |
| Decode interquartile range (tok/s) | reasoning | 55.74 | 54.72 | 55.23 | 40.77 | 51.29 | 46.03 |
| Decode interquartile range (tok/s) | summarization | 51.39 | 60.93 | 56.16 | 39.13 | 32.08 | 35.61 |
| Decode coefficient of variation (ratio) | chat | 0.2088 | 0.2081 | 0.2084 | 0.1897 | 0.2182 | 0.2039 |
| Decode coefficient of variation (ratio) | code | 0.2129 | 0.2390 | 0.2260 | 0.2349 | 0.2187 | 0.2268 |
| Decode coefficient of variation (ratio) | file_edit | 0.1727 | 0.1750 | 0.1739 | 0.1892 | 0.2016 | 0.1954 |
| Decode coefficient of variation (ratio) | json | 0.1255 | 0.1414 | 0.1334 | 0.1441 | 0.1261 | 0.1351 |
| Decode coefficient of variation (ratio) | math | 0.1227 | 0.1055 | 0.1141 | 0.1150 | 0.1157 | 0.1153 |
| Decode coefficient of variation (ratio) | prose | 0.2305 | 0.2163 | 0.2234 | 0.1960 | 0.1927 | 0.1943 |
| Decode coefficient of variation (ratio) | reasoning | 0.2104 | 0.2989 | 0.2546 | 0.2229 | 0.2149 | 0.2189 |
| Decode coefficient of variation (ratio) | summarization | 0.1075 | 0.1522 | 0.1299 | 0.1324 | 0.1098 | 0.1211 |
| Successful single-stream requests | chat | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Successful single-stream requests | code | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Successful single-stream requests | file_edit | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Successful single-stream requests | json | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Successful single-stream requests | math | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Successful single-stream requests | prose | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Successful single-stream requests | reasoning | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Successful single-stream requests | summarization | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Single-stream output tokens | chat | 4,920 | 4,920 | 4,920.0 | 4,920 | 4,920 | 4,920.0 |
| Single-stream output tokens | code | 10,988 | 10,876 | 10,932.0 | 10,565 | 10,871 | 10,718.0 |
| Single-stream output tokens | file_edit | 8,373 | 8,587 | 8,480.0 | 8,471 | 7,922 | 8,196.5 |
| Single-stream output tokens | json | 5,247 | 4,973 | 5,110.0 | 5,141 | 4,921 | 5,031.0 |
| Single-stream output tokens | math | 9,201 | 9,295 | 9,248.0 | 9,224 | 9,378 | 9,301.0 |
| Single-stream output tokens | prose | 9,038 | 9,234 | 9,136.0 | 9,040 | 8,926 | 8,983.0 |
| Single-stream output tokens | reasoning | 10,547 | 10,490 | 10,518.5 | 10,653 | 10,579 | 10,616.0 |
| Single-stream output tokens | summarization | 3,169 | 3,210 | 3,189.5 | 3,197 | 3,057 | 3,127.0 |
| Requests with multi-token updates | chat | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with multi-token updates | code | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with multi-token updates | file_edit | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with multi-token updates | json | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with multi-token updates | math | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with multi-token updates | prose | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with multi-token updates | reasoning | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with multi-token updates | summarization | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | chat | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | code | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | file_edit | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | json | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | math | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | prose | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | reasoning | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| First-token observations | summarization | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Stream-update gap observations | chat | 1,628 | 1,672 | 1,650.0 | 1,789 | 1,689 | 1,739.0 |
| Stream-update gap observations | code | 3,221 | 3,258 | 3,239.5 | 3,096 | 3,155 | 3,125.5 |
| Stream-update gap observations | file_edit | 2,014 | 2,060 | 2,037.0 | 2,149 | 2,010 | 2,079.5 |
| Stream-update gap observations | json | 1,096 | 1,030 | 1,063.0 | 1,019 | 1,032 | 1,025.5 |
| Stream-update gap observations | math | 1,901 | 1,977 | 1,939.0 | 1,903 | 1,990 | 1,946.5 |
| Stream-update gap observations | prose | 3,256 | 3,325 | 3,290.5 | 3,288 | 3,231 | 3,259.5 |
| Stream-update gap observations | reasoning | 3,916 | 4,009 | 3,962.5 | 3,816 | 3,782 | 3,799.0 |
| Stream-update gap observations | summarization | 729 | 728 | 728.5 | 712 | 657 | 684.5 |
| Per-token rate low 1% (tok/s) | chat | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate low 1% (tok/s) | code | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate low 1% (tok/s) | file_edit | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate low 1% (tok/s) | json | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate low 1% (tok/s) | math | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate low 1% (tok/s) | prose | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate low 1% (tok/s) | reasoning | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate low 1% (tok/s) | summarization | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | chat | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | code | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | file_edit | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | json | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | math | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | prose | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | reasoning | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate median (tok/s) | summarization | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | chat | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | code | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | file_edit | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | json | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | math | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | prose | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | reasoning | not retained | not retained | not retained | not retained | not retained | not retained |
| Per-token rate high 99% (tok/s) | summarization | not retained | not retained | not retained | not retained | not retained | not retained |
| Headline weight (%) | chat | 0 | 0 | 0 | 0 | 0 | 0 |
| Headline weight (%) | code | 30 | 30 | 30 | 30 | 30 | 30 |
| Headline weight (%) | file_edit | 10 | 10 | 10 | 10 | 10 | 10 |
| Headline weight (%) | json | 15 | 15 | 15 | 15 | 15 | 15 |
| Headline weight (%) | math | 0 | 0 | 0 | 0 | 0 | 0 |
| Headline weight (%) | prose | 15 | 15 | 15 | 15 | 15 | 15 |
| Headline weight (%) | reasoning | 20 | 20 | 20 | 20 | 20 | 20 |
| Headline weight (%) | summarization | 10 | 10 | 10 | 10 | 10 | 10 |
| **Concurrency** |  |  |  |  |  |  |  |
| Aggregate output (tok/s) | C1 | 186.0441 | 174.3569 | 180.2005 | 164.5913 | 159.7606 | 162.1759 |
| Aggregate output (tok/s) | C2 | 282.5754 | 263.6837 | 273.1295 | 233.5488 | 238.3587 | 235.9538 |
| Aggregate output (tok/s) | C4 | 389.7604 | 368.9610 | 379.3607 | 333.7764 | 340.1015 | 336.9390 |
| Median decode per request (tok/s) | C1 | 250.2044 | 223.6856 | 236.9450 | 195.8966 | 203.9786 | 199.9376 |
| Median decode per request (tok/s) | C2 | 183.3481 | 171.0707 | 177.2094 | 148.6928 | 156.2134 | 152.4531 |
| Median decode per request (tok/s) | C4 | 121.6686 | 120.9802 | 121.3244 | 120.8164 | 109.8827 | 115.3495 |
| Median first token (ms) | C1 | 111.0873 | 110.7226 | 110.9050 | 131.6575 | 129.4813 | 130.5694 |
| Median first token (ms) | C2 | 147.7247 | 147.2531 | 147.4889 | 164.3862 | 164.6319 | 164.5090 |
| Median first token (ms) | C4 | 155.1615 | 157.4557 | 156.3086 | 183.3725 | 181.1450 | 182.2587 |
| First-token p99 (ms) † | C1 | 139.77 | 145.40 | 142.59 | 147.99 | 140.99 | 144.49 |
| First-token p99 (ms) † | C2 | 233.12 | 243.99 | 238.56 | 1,232.60 | 1,142.56 | 1,187.58 |
| First-token p99 (ms) † | C4 | 341.38 | 367.15 | 354.27 | 608.87 | 791.22 | 700.05 |
| Successful concurrency requests | C1 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| Successful concurrency requests | C2 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| Successful concurrency requests | C4 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| Scheduled concurrency requests | C1 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| Scheduled concurrency requests | C2 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| Scheduled concurrency requests | C4 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| First-token observations | C1 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| First-token observations | C2 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| First-token observations | C4 | 48 | 48 | 48.0 | 48 | 48 | 48.0 |
| Timed concurrency window (s) | C1 | 104.26 | 108.39 | 106.32 | 118.58 | 120.19 | 119.38 |
| Timed concurrency window (s) | C2 | 69.78 | 74.65 | 72.22 | 82.99 | 82.22 | 82.60 |
| Timed concurrency window (s) | C4 | 50.20 | 53.09 | 51.64 | 58.09 | 55.95 | 57.02 |
| **Cold prefill** |  |  |  |  |  |  |  |
| Nominal band (tokens) | 2K | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 |
| Actual median input (tokens) | 2K | 1,563.00 | 1,562.00 | 1,562.50 | 1,548.00 | 1,554.50 | 1,551.25 |
| Median first token (ms) | 2K | 284.6616 | 285.7993 | 285.2304 | 337.7442 | 338.4197 | 338.0820 |
| Median prefill throughput (tok/s) | 2K | 5,422.01 | 5,435.84 | 5,428.92 | 4,609.72 | 4,607.69 | 4,608.70 |
| Prefill throughput low 1% (tok/s) † | 2K | 5,256.01 | 5,119.76 | 5,187.88 | 4,577.91 | 4,528.53 | 4,553.22 |
| Prefill throughput high 99% (tok/s) † | 2K | 5,660.43 | 5,822.07 | 5,741.25 | 4,728.83 | 4,758.33 | 4,743.58 |
| Successful prefill observations | 2K | 8 | 8 | 8.0 | 8 | 8 | 8.0 |
| Nominal band (tokens) | 8K | 8,000 | 8,000 | 8,000 | 8,000 | 8,000 | 8,000 |
| Actual median input (tokens) | 8K | 5,968.50 | 5,956.00 | 5,962.25 | 5,953.50 | 5,957.00 | 5,955.25 |
| Median first token (ms) | 8K | 1,006.1216 | 1,007.4439 | 1,006.7827 | 1,259.5870 | 1,259.3976 | 1,259.4923 |
| Median prefill throughput (tok/s) | 8K | 5,932.19 | 5,909.90 | 5,921.04 | 4,731.74 | 4,732.50 | 4,732.12 |
| Prefill throughput low 1% (tok/s) † | 8K | 5,855.17 | 5,759.89 | 5,807.53 | 4,654.67 | 4,684.73 | 4,669.70 |
| Prefill throughput high 99% (tok/s) † | 8K | 5,984.25 | 6,006.63 | 5,995.44 | 4,753.13 | 4,799.21 | 4,776.17 |
| Successful prefill observations | 8K | 8 | 8 | 8.0 | 8 | 8 | 8.0 |
| Nominal band (tokens) | 16K | 16,000 | 16,000 | 16,000 | 16,000 | 16,000 | 16,000 |
| Actual median input (tokens) | 16K | 11,854.50 | 11,853.50 | 11,854.00 | 11,871.00 | 11,863.00 | 11,867.00 |
| Median first token (ms) | 16K | 1,855.8350 | 1,855.0791 | 1,855.4570 | 2,458.5835 | 2,462.3790 | 2,460.4813 |
| Median prefill throughput (tok/s) | 16K | 6,382.69 | 6,376.79 | 6,379.74 | 4,827.04 | 4,801.71 | 4,814.37 |
| Prefill throughput low 1% (tok/s) † | 16K | 6,303.94 | 6,341.51 | 6,322.73 | 4,683.30 | 4,623.97 | 4,653.64 |
| Prefill throughput high 99% (tok/s) † | 16K | 6,416.53 | 6,446.58 | 6,431.55 | 4,845.30 | 4,838.77 | 4,842.03 |
| Successful prefill observations | 16K | 8 | 8 | 8.0 | 8 | 8 | 8.0 |
| Nominal band (tokens) | 32K | 32,000 | 32,000 | 32,000 | 32,000 | 32,000 | 32,000 |
| Actual median input (tokens) | 32K | 23,572.00 | 23,609.00 | 23,590.50 | 23,576.00 | 23,625.00 | 23,600.50 |
| Median first token (ms) | 32K | 3,808.1069 | 3,829.7475 | 3,818.9272 | 5,019.2372 | 5,053.3135 | 5,036.2753 |
| Median prefill throughput (tok/s) | 32K | 6,191.65 | 6,158.63 | 6,175.14 | 4,695.76 | 4,677.96 | 4,686.86 |
| Prefill throughput low 1% (tok/s) † | 32K | 6,167.87 | 6,119.51 | 6,143.69 | 4,675.48 | 4,644.76 | 4,660.12 |
| Prefill throughput high 99% (tok/s) † | 32K | 6,224.89 | 6,195.61 | 6,210.25 | 4,718.47 | 4,702.99 | 4,710.73 |
| Successful prefill observations | 32K | 8 | 8 | 8.0 | 8 | 8 | 8.0 |
| Nominal band (tokens) | 64K | 64,000 | 64,000 | 64,000 | 64,000 | 64,000 | 64,000 |
| Actual median input (tokens) | 64K | 47,100.00 | 47,069.50 | 47,084.75 | 47,082.50 | 47,169.50 | 47,126.00 |
| Median first token (ms) | 64K | 8,109.8768 | 8,077.7153 | 8,093.7960 | 10,888.1490 | 10,943.8228 | 10,915.9859 |
| Median prefill throughput (tok/s) | 64K | 5,810.29 | 5,830.64 | 5,820.46 | 4,327.08 | 4,309.97 | 4,318.52 |
| Prefill throughput low 1% (tok/s) † | 64K | 5,796.40 | 5,775.89 | 5,786.15 | 4,308.62 | 4,272.93 | 4,290.77 |
| Prefill throughput high 99% (tok/s) † | 64K | 5,822.74 | 5,850.83 | 5,836.78 | 4,333.81 | 4,317.64 | 4,325.73 |
| Successful prefill observations | 64K | 8 | 8 | 8.0 | 8 | 8 | 8.0 |
| **Reasoning and answer arrival** |  |  |  |  |  |  |  |
| Requests with a detected reasoning split | chat | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with a detected reasoning split | code | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with a detected reasoning split | file_edit | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with a detected reasoning split | json | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with a detected reasoning split | math | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with a detected reasoning split | prose | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with a detected reasoning split | reasoning | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests with a detected reasoning split | summarization | 20 | 20 | 20.0 | 20 | 20 | 20.0 |
| Requests reaching an answer | chat | 10 | 7 | 8.5 | 6 | 8 | 7.0 |
| Requests reaching an answer | code | 10 | 10 | 10.0 | 11 | 15 | 13.0 |
| Requests reaching an answer | file_edit | 7 | 6 | 6.5 | 7 | 9 | 8.0 |
| Requests reaching an answer | json | 15 | 15 | 15.0 | 15 | 15 | 15.0 |
| Requests reaching an answer | math | 15 | 13 | 14.0 | 14 | 13 | 13.5 |
| Requests reaching an answer | prose | 5 | 4 | 4.5 | 5 | 5 | 5.0 |
| Requests reaching an answer | reasoning | 5 | 5 | 5.0 | 6 | 6 | 6.0 |
| Requests reaching an answer | summarization | 14 | 14 | 14.0 | 16 | 18 | 17.0 |
| Requests without an answer | chat | 10 | 13 | 11.5 | 14 | 12 | 13.0 |
| Requests without an answer | code | 10 | 10 | 10.0 | 9 | 5 | 7.0 |
| Requests without an answer | file_edit | 13 | 14 | 13.5 | 13 | 11 | 12.0 |
| Requests without an answer | json | 5 | 5 | 5.0 | 5 | 5 | 5.0 |
| Requests without an answer | math | 5 | 7 | 6.0 | 6 | 7 | 6.5 |
| Requests without an answer | prose | 15 | 16 | 15.5 | 15 | 15 | 15.0 |
| Requests without an answer | reasoning | 15 | 15 | 15.0 | 14 | 14 | 14.0 |
| Requests without an answer | summarization | 6 | 6 | 6.0 | 4 | 2 | 3.0 |
| Estimated reasoning share (ratio) | chat | 0.7941 | 0.8087 | 0.8014 | 0.9392 | 0.8161 | 0.8776 |
| Estimated reasoning share (ratio) | code | 0.8710 | 0.8928 | 0.8819 | 0.8713 | 0.8033 | 0.8373 |
| Estimated reasoning share (ratio) | file_edit | 0.9764 | 0.9789 | 0.9776 | 0.9770 | 0.9438 | 0.9604 |
| Estimated reasoning share (ratio) | json | 0.5740 | 0.5856 | 0.5798 | 0.5929 | 0.5629 | 0.5779 |
| Estimated reasoning share (ratio) | math | 0.6774 | 0.6767 | 0.6771 | 0.6661 | 0.6751 | 0.6706 |
| Estimated reasoning share (ratio) | prose | 0.9823 | 0.9874 | 0.9849 | 0.9825 | 0.9829 | 0.9827 |
| Estimated reasoning share (ratio) | reasoning | 0.9168 | 0.9705 | 0.9436 | 0.9543 | 0.9622 | 0.9582 |
| Estimated reasoning share (ratio) | summarization | 0.5333 | 0.5735 | 0.5534 | 0.5480 | 0.4845 | 0.5162 |
| Median first answer (ms) | chat | 835.89 | 778.59 | 807.24 | 810.12 | 685.73 | 747.93 |
| Median first answer (ms) | code | 2,272.88 | 1,180.50 | 1,726.69 | 1,761.45 | 1,533.47 | 1,647.46 |
| Median first answer (ms) | file_edit | 623.72 | 696.69 | 660.21 | 770.68 | 764.68 | 767.68 |
| Median first answer (ms) | json | 478.78 | 510.95 | 494.87 | 603.63 | 578.07 | 590.85 |
| Median first answer (ms) | math | 615.08 | 705.24 | 660.16 | 790.69 | 864.33 | 827.51 |
| Median first answer (ms) | prose | 821.82 | not retained | not retained | 1,183.02 | 1,033.51 | 1,108.27 |
| Median first answer (ms) | reasoning | 518.00 | 288.70 | 403.35 | 821.76 | 1,404.21 | 1,112.98 |
| Median first answer (ms) | summarization | 370.18 | 404.02 | 387.10 | 458.23 | 390.86 | 424.54 |
| Capped while reasoning | chat | 10 | 13 | 11.5 | 14 | 12 | 13.0 |
| Capped while reasoning | code | 10 | 10 | 10.0 | 9 | 5 | 7.0 |
| Capped while reasoning | file_edit | 13 | 14 | 13.5 | 13 | 11 | 12.0 |
| Capped while reasoning | json | 5 | 5 | 5.0 | 5 | 5 | 5.0 |
| Capped while reasoning | math | 5 | 7 | 6.0 | 6 | 7 | 6.5 |
| Capped while reasoning | prose | 15 | 16 | 15.5 | 15 | 15 | 15.0 |
| Capped while reasoning | reasoning | 15 | 15 | 15.0 | 14 | 14 | 14.0 |
| Capped while reasoning | summarization | 6 | 6 | 6.0 | 4 | 2 | 3.0 |
| **Completion and run accounting** |  |  |  |  |  |  |  |
| Capped while reasoning | Single-stream | 79 | 86 | 82.5 | 80 | 71 | 75.5 |
| Length-capped requests | Single-stream | 113 | 115 | 114.0 | 112 | 104 | 108.0 |
| Successful measured requests | Single-stream | 160 | 160 | 160.0 | 160 | 160 | 160.0 |
| Successful measured requests | Prefill | 40 | 40 | 40.0 | 40 | 40 | 40.0 |
| Successful measured requests | Concurrency | 144 | 144 | 144.0 | 144 | 144 | 144.0 |
| Successful measured requests | All phases | 344 | 344 | 344.0 | 344 | 344 | 344.0 |
| Discarded warmup requests | All phases | 34 | 34 | 34.0 | 34 | 34 | 34.0 |
| Reported requests with prefix-cache hits | All phases | 0 | 0 | 0.0 | 0 | 0 | 0.0 |
| CLI elapsed (s) | All phases | 781.19 | 796.48 | 788.83 | 937.74 | 932.32 | 935.03 |
