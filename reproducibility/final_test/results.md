# Held-out test results

Results on the 250-example mixed test set. Each retrieval-free baseline is followed by the corresponding system selected exclusively on development data. Delta rows report the selected system minus its baseline. Quality values are means ± standard deviations over seeds 42, 43, and 44. Seconds and tokens are mean costs per answer.

| # | Model | Configuration | SF | ROUGE-L | BERT-F1 | MC-acc | MeanQ | Seconds | Tokens |
|---:|:---|:---|:---:|---:|---:|---:|---:|---:|---:|
| 0a | Qwen no-think | Baseline LLM only | no | 26.00 ± 0.96 | 71.64 ± 0.43 | 76.27 ± 1.22 | 57.97 ± 0.86 | 0.38 | 726 |
| 15a | Qwen no-think | MA-RAG | no | 38.22 ± 0.31 | 77.54 ± 0.28 | 81.60 ± 1.39 | 65.79 ± 0.30 | 2.09 | 3113 |
| Δa | Qwen no-think | Selected − baseline | no | +12.23 ± 0.97 | +5.91 ± 0.50 | +5.33 ± 2.01 | +7.82 ± 1.07 | +1.71 | +2387 |
| 0b | Qwen think | Baseline LLM only | no | 30.07 ± 0.37 | 73.63 ± 0.08 | 86.67 ± 1.22 | 63.46 ± 0.50 | 2.87 | 4051 |
| 6b | Qwen think | rerank top 5 | no | 38.17 ± 0.27 | 77.15 ± 0.14 | 86.93 ± 1.22 | 67.42 ± 0.35 | 3.82 | 5779 |
| Δb | Qwen think | Selected − baseline | no | +8.10 ± 0.27 | +3.52 ± 0.16 | +0.27 ± 0.46 | +3.96 ± 0.19 | +0.96 | +1728 |
| 0c | Llama | Baseline LLM only | no | 11.46 ± 0.35 | 66.25 ± 0.11 | 24.27 ± 3.61 | 33.99 ± 1.30 | 0.39 | 889 |
| 5c | Llama | rerank top 3 | no | 20.21 ± 1.24 | 69.11 ± 1.20 | 42.22 ± 4.75 | 43.85 ± 2.35 | 0.68 | 1779 |
| Δc | Llama | Selected − baseline | no | +8.75 ± 0.90 | +2.87 ± 1.12 | +17.95 ± 2.15 | +9.86 ± 1.30 | +0.29 | +891 |
| 0d | Latxa | Baseline LLM only | no | 10.07 ± 0.27 | 65.64 ± 0.18 | 35.73 ± 0.92 | 37.15 ± 0.37 | 0.24 | 771 |
| 1d | Latxa | E5 top 1 | no | 14.46 ± 0.32 | 67.92 ± 0.30 | 47.40 ± 1.95 | 43.26 ± 0.52 | 0.27 | 1046 |
| Δd | Latxa | Selected − baseline | no | +4.39 ± 0.52 | +2.28 ± 0.47 | +11.66 ± 2.68 | +6.11 ± 0.78 | +0.03 | +275 |
