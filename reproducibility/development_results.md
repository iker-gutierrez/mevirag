# Development results

Final results on the 126-example mixed development set. Quality values are means ± standard deviations over seeds 42, 43, and 44. Seconds and tokens are mean costs per answer. SF indicates whether the self-feedback revision is scored.

Configuration selection was performed separately for each model with the MeanQ–Stability–Token (MST) rule. The ablation-selected rows are Qwen no-think 3a, Qwen think 6b, Llama 5c, and Latxa 1d. The reasoning-pipeline comparison subsequently selected MA-RAG (15a) for Qwen no-think, while the other three ablation references remained selected for held-out testing.

## Ablation study

The eleven configurations cover the retrieval-free baseline, dense-retrieval depth, cross-encoder reranking, three-shot prompting, and retrieval-corpus scope.

### Spanish

| # | Model | Configuration | SF | ROUGE-L | BERT-F1 | MC-acc | MeanQ | Seconds | Tokens | Ablation selected |
|---:|:---|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 0a | Qwen no-think | Baseline LLM only | no | 27.07 ± 0.33 | 72.03 ± 0.30 | 83.07 ± 0.92 | 60.72 ± 0.30 | 0.43 | 734 |  |
| 0a' | Qwen no-think | Baseline LLM only | yes | 26.32 ± 0.41 | 71.54 ± 0.14 | 81.48 ± 0.92 | 59.78 ± 0.49 | 0.99 | 1516 |  |
| 0b | Qwen think | Baseline LLM only | no | 30.30 ± 0.31 | 73.89 ± 0.08 | 92.59 ± 0.92 | 65.59 ± 0.44 | 3.14 | 3960 |  |
| 0b' | Qwen think | Baseline LLM only | yes | 30.08 ± 0.44 | 73.90 ± 0.14 | 92.59 ± 0.92 | 65.52 ± 0.50 | 5.58 | 6456 |  |
| 1a | Qwen no-think | E5 top 1 | no | 37.03 ± 0.20 | 76.69 ± 0.48 | 83.60 ± 2.42 | 65.77 ± 0.91 | 0.45 | 862 |  |
| 1a' | Qwen no-think | E5 top 1 | yes | 36.47 ± 0.46 | 76.36 ± 0.18 | 82.54 ± 2.75 | 65.12 ± 1.05 | 0.70 | 1781 |  |
| 1b | Qwen think | E5 top 1 | no | 37.35 ± 0.68 | 76.85 ± 0.16 | 93.12 ± 1.83 | 69.11 ± 0.81 | 3.37 | 4377 |  |
| 1b' | Qwen think | E5 top 1 | yes | 37.35 ± 0.72 | 76.90 ± 0.22 | 93.12 ± 1.83 | 69.12 ± 0.87 | 6.07 | 7267 |  |
| 2a | Qwen no-think | E5 top 3 | no | 40.09 ± 1.36 | 77.70 ± 0.29 | 83.07 ± 4.85 | 66.95 ± 2.14 | 0.36 | 1364 |  |
| 2a' | Qwen no-think | E5 top 3 | yes | 38.69 ± 0.47 | 77.22 ± 0.24 | 82.01 ± 3.99 | 65.97 ± 1.43 | 0.87 | 2786 |  |
| 2b | Qwen think | E5 top 3 | no | 39.93 ± 0.16 | 77.58 ± 0.10 | 93.12 ± 0.92 | 70.21 ± 0.38 | 3.44 | 4948 |  |
| 2b' | Qwen think | E5 top 3 | yes | 40.11 ± 0.07 | 77.60 ± 0.07 | 93.12 ± 0.92 | 70.28 ± 0.33 | 6.30 | 8408 |  |
| 3a | Qwen no-think | E5 top 5 | no | 41.53 ± 0.43 | 78.15 ± 0.24 | 83.60 ± 2.42 | 67.76 ± 0.97 | 0.65 | 1930 | yes |
| 3a' | Qwen no-think | E5 top 5 | yes | 41.04 ± 1.15 | 77.88 ± 0.24 | 83.07 ± 3.30 | 67.33 ± 1.55 | 1.13 | 3909 |  |
| 3b | Qwen think | E5 top 5 | no | 41.05 ± 0.53 | 77.92 ± 0.23 | 93.12 ± 0.92 | 70.70 ± 0.26 | 3.56 | 5777 |  |
| 3b' | Qwen think | E5 top 5 | yes | 41.24 ± 0.67 | 78.00 ± 0.36 | 93.12 ± 0.92 | 70.79 ± 0.34 | 6.70 | 9885 |  |
| 4a | Qwen no-think | rerank top 1 | no | 37.24 ± 0.64 | 76.72 ± 0.46 | 86.77 ± 0.92 | 66.91 ± 0.25 | 0.84 | 927 |  |
| 4a' | Qwen no-think | rerank top 1 | yes | 36.17 ± 0.16 | 76.34 ± 0.25 | 85.71 ± 1.59 | 66.07 ± 0.56 | 1.18 | 1920 |  |
| 4b | Qwen think | rerank top 1 | no | 38.21 ± 0.74 | 77.17 ± 0.23 | 93.12 ± 0.92 | 69.50 ± 0.60 | 3.51 | 4540 |  |
| 4b' | Qwen think | rerank top 1 | yes | 37.95 ± 0.81 | 77.13 ± 0.25 | 93.12 ± 0.92 | 69.40 ± 0.65 | 7.16 | 7543 |  |
| 5a | Qwen no-think | rerank top 3 | no | 38.21 ± 0.42 | 76.93 ± 0.17 | 84.66 ± 2.42 | 66.60 ± 0.95 | 1.20 | 1479 |  |
| 5a' | Qwen no-think | rerank top 3 | yes | 38.23 ± 0.59 | 76.98 ± 0.24 | 86.24 ± 0.92 | 67.15 ± 0.52 | 1.52 | 3011 |  |
| 5b | Qwen think | rerank top 3 | no | 39.37 ± 0.57 | 77.51 ± 0.34 | 92.59 ± 0.92 | 69.82 ± 0.30 | 3.76 | 5191 |  |
| 5b' | Qwen think | rerank top 3 | yes | 39.16 ± 0.92 | 77.59 ± 0.38 | 92.59 ± 0.92 | 69.78 ± 0.47 | 6.75 | 8803 |  |
| 6a | Qwen no-think | rerank top 5 | no | 39.60 ± 0.91 | 77.93 ± 0.31 | 85.19 ± 2.42 | 67.57 ± 0.82 | 1.14 | 2032 |  |
| 6a' | Qwen no-think | rerank top 5 | yes | 38.53 ± 0.48 | 77.53 ± 0.16 | 85.19 ± 2.42 | 67.08 ± 0.81 | 1.53 | 4122 |  |
| 6b | Qwen think | rerank top 5 | no | 41.41 ± 1.18 | 78.10 ± 0.31 | 93.65 ± 0.00 | 71.05 ± 0.50 | 3.92 | 5753 | yes |
| 6b' | Qwen think | rerank top 5 | yes | 41.22 ± 1.25 | 78.14 ± 0.32 | 93.65 ± 0.00 | 71.01 ± 0.52 | 7.11 | 9920 |  |
| 7a | Qwen no-think | 3-shot, no RAG | no | 26.32 ± 0.74 | 69.61 ± 0.50 | 77.25 ± 2.42 | 57.73 ± 1.18 | 0.94 | 1669 |  |
| 7a' | Qwen no-think | 3-shot, no RAG | yes | 27.53 ± 0.29 | 72.50 ± 0.07 | 79.89 ± 1.83 | 59.98 ± 0.68 | 1.31 | 2412 |  |
| 7b | Qwen think | 3-shot, no RAG | no | 32.78 ± 0.60 | 75.49 ± 0.16 | 93.12 ± 2.42 | 67.13 ± 0.98 | 3.74 | 5405 |  |
| 7b' | Qwen think | 3-shot, no RAG | yes | 32.42 ± 0.93 | 75.34 ± 0.27 | 93.12 ± 2.42 | 66.96 ± 1.11 | 6.32 | 7940 |  |
| 8a | Qwen no-think | 3-shot + E5 top 5 | no | 38.07 ± 2.38 | 74.02 ± 2.80 | 81.48 ± 0.92 | 64.53 ± 1.95 | 0.64 | 2895 |  |
| 8a' | Qwen no-think | 3-shot + E5 top 5 | yes | 41.95 ± 0.57 | 78.29 ± 0.17 | 82.54 | 67.59 ± 0.24 | 1.01 | 4865 |  |
| 8b | Qwen think | 3-shot + rerank top 5 | no | 42.86 ± 0.58 | 79.07 ± 0.14 | 91.53 ± 0.92 | 71.16 ± 0.13 | 4.99 | 7097 |  |
| 8b' | Qwen think | 3-shot + rerank top 5 | yes | 42.69 ± 0.62 | 79.02 ± 0.26 | 91.53 ± 0.92 | 71.08 ± 0.08 | 8.40 | 11316 |  |
| 9a | Qwen no-think | GuiaSalud retrieval, E5 top 5 | no | 40.96 ± 0.62 | 78.09 ± 0.28 | 75.66 ± 0.92 | 64.91 ± 0.42 | 0.80 | 2335 |  |
| 9a' | Qwen no-think | GuiaSalud retrieval, E5 top 5 | yes | 40.86 ± 0.93 | 78.17 ± 0.34 | 78.31 ± 0.92 | 65.78 ± 0.34 | 1.35 | 4720 |  |
| 9b | Qwen think | GuiaSalud retrieval, rerank top 5 | no | 40.69 ± 0.68 | 78.23 ± 0.21 | 87.83 ± 2.42 | 68.92 ± 1.09 | 4.56 | 6355 |  |
| 9b' | Qwen think | GuiaSalud retrieval, rerank top 5 | yes | 41.13 ± 0.64 | 78.41 ± 0.23 | 87.83 ± 2.42 | 69.12 ± 1.08 | 7.87 | 11033 |  |
| 10a | Qwen no-think | CasiMédicos retrieval, E5 top 5 | no | 26.74 ± 0.18 | 72.31 ± 0.13 | 79.89 ± 0.92 | 59.65 ± 0.40 | 0.60 | 2218 |  |
| 10a' | Qwen no-think | CasiMédicos retrieval, E5 top 5 | yes | 26.48 ± 0.57 | 71.85 ± 0.27 | 79.37 ± 2.75 | 59.23 ± 1.19 | 0.99 | 4506 |  |
| 10b | Qwen think | CasiMédicos retrieval, rerank top 5 | no | 30.39 ± 0.52 | 74.01 ± 0.14 | 94.71 ± 0.92 | 66.37 ± 0.52 | 4.20 | 5889 |  |
| 10b' | Qwen think | CasiMédicos retrieval, rerank top 5 | yes | 30.33 ± 0.36 | 74.02 ± 0.12 | 94.71 ± 0.92 | 66.35 ± 0.47 | 7.30 | 10297 |  |

### Basque

| # | Model | Configuration | SF | ROUGE-L | BERT-F1 | MC-acc | MeanQ | Seconds | Tokens | Ablation selected |
|---:|:---|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 0c | Llama | Baseline LLM only | no | 12.30 ± 1.38 | 66.85 ± 0.41 | 30.69 ± 4.85 | 36.61 ± 2.20 | 0.59 | 893 |  |
| 0c' | Llama | Baseline LLM only | yes | 12.20 ± 0.82 | 66.40 ± 0.30 | 33.86 ± 3.30 | 37.49 ± 1.47 | 1.23 | 1993 |  |
| 0d | Latxa | Baseline LLM only | no | 10.06 ± 0.41 | 65.91 ± 0.30 | 37.04 ± 1.83 | 37.67 ± 0.51 | 0.29 | 772 |  |
| 0d' | Latxa | Baseline LLM only | yes | 10.02 ± 0.35 | 65.86 ± 0.21 | 36.51 ± 2.75 | 37.46 ± 0.84 | 0.73 | 1623 |  |
| 1c | Llama | E5 top 1 | no | 20.79 ± 0.65 | 71.92 ± 0.18 | 38.10 ± 3.17 | 43.60 ± 1.31 | 0.40 | 1067 |  |
| 1c' | Llama | E5 top 1 | yes | 19.85 ± 0.24 | 71.22 ± 0.33 | 38.62 ± 0.92 | 43.23 ± 0.26 | 0.66 | 2292 |  |
| 1d | Latxa | E5 top 1 | no | 15.81 ± 0.63 | 68.51 ± 0.23 | 50.54 ± 1.24 | 44.95 ± 0.68 | 0.18 | 1047 | yes |
| 1d' | Latxa | E5 top 1 | yes | 15.04 ± 0.46 | 68.41 ± 0.26 | 48.92 ± 2.02 | 44.12 ± 0.90 | 0.36 | 2165 |  |
| 2c | Llama | E5 top 3 | no | 21.35 ± 2.94 | 69.77 ± 1.81 | 45.39 ± 7.18 | 45.50 ± 3.86 | 0.58 | 1700 |  |
| 2c' | Llama | E5 top 3 | yes | 21.60 ± 2.15 | 71.91 ± 0.37 | 45.20 ± 3.48 | 46.24 ± 1.99 | 0.97 | 3516 |  |
| 2d | Latxa | E5 top 3 | no | 17.01 ± 0.82 | 67.28 ± 0.19 | 50.85 ± 3.97 | 45.05 ± 1.57 | 0.44 | 1738 |  |
| 2d' | Latxa | E5 top 3 | yes | 17.18 ± 0.76 | 67.99 ± 0.57 | 48.55 ± 4.43 | 44.57 ± 1.73 | 0.82 | 3474 |  |
| 3c | Llama | E5 top 5 | no | 20.79 ± 1.22 | 66.43 ± 1.28 | 46.97 ± 7.45 | 44.73 ± 2.91 | 0.80 | 2362 |  |
| 3c' | Llama | E5 top 5 | yes | 22.33 ± 1.18 | 71.11 ± 0.93 | 46.90 ± 3.92 | 46.78 ± 1.80 | 1.51 | 4813 |  |
| 3d | Latxa | E5 top 5 | no | 19.30 ± 0.40 | 67.75 ± 0.86 | 44.33 ± 1.36 | 43.79 ± 0.84 | 1.17 | 2523 |  |
| 3d' | Latxa | E5 top 5 | yes | 17.83 ± 0.95 | 68.67 ± 0.37 | 47.78 ± 2.55 | 44.76 ± 1.06 | 1.91 | 4903 |  |
| 4c | Llama | rerank top 1 | no | 22.64 ± 0.94 | 72.13 ± 0.20 | 39.15 ± 5.57 | 44.64 ± 2.23 | 2.20 | 1123 |  |
| 4c' | Llama | rerank top 1 | yes | 22.44 ± 0.75 | 71.55 ± 0.21 | 40.21 ± 4.85 | 44.73 ± 1.85 | 2.58 | 2400 |  |
| 4d | Latxa | rerank top 1 | no | 15.52 ± 1.16 | 68.15 ± 0.14 | 45.66 ± 3.07 | 43.11 ± 1.39 | 0.52 | 1098 |  |
| 4d' | Latxa | rerank top 1 | yes | 15.14 ± 1.15 | 68.33 ± 0.47 | 45.20 ± 3.48 | 42.89 ± 1.31 | 0.71 | 2271 |  |
| 5c | Llama | rerank top 3 | no | 23.78 ± 0.43 | 70.08 ± 0.81 | 48.52 ± 1.86 | 47.46 ± 0.62 | 1.45 | 1795 | yes |
| 5c' | Llama | rerank top 3 | yes | 24.21 ± 0.26 | 72.30 ± 0.32 | 45.75 ± 1.19 | 47.42 ± 0.47 | 2.01 | 3719 |  |
| 5d | Latxa | rerank top 3 | no | 17.40 ± 0.85 | 67.33 ± 0.60 | 49.67 ± 5.83 | 44.80 ± 2.29 | 0.78 | 1809 |  |
| 5d' | Latxa | rerank top 3 | yes | 16.50 ± 1.03 | 68.22 ± 0.79 | 50.17 ± 4.02 | 44.96 ± 1.74 | 1.05 | 3615 |  |
| 6c | Llama | rerank top 5 | no | 24.15 ± 0.86 | 67.44 ± 0.61 | 51.40 ± 6.73 | 47.67 ± 2.20 | 1.36 | 2449 |  |
| 6c' | Llama | rerank top 5 | yes | 25.00 ± 1.00 | 72.25 ± 0.23 | 46.47 ± 3.33 | 47.91 ± 1.03 | 2.07 | 5011 |  |
| 6d | Latxa | rerank top 5 | no | 18.94 ± 0.99 | 67.87 ± 0.55 | 46.95 ± 3.77 | 44.59 ± 1.33 | 1.52 | 2669 |  |
| 6d' | Latxa | rerank top 5 | yes | 17.12 ± 1.16 | 68.59 ± 0.19 | 46.05 ± 5.10 | 43.92 ± 1.25 | 2.10 | 5130 |  |
| 7c | Llama | 3-shot, no RAG | no | 11.20 ± 0.18 | 61.19 ± 0.39 | 30.99 ± 3.18 | 34.46 ± 1.22 | 1.21 | 2149 |  |
| 7c' | Llama | 3-shot, no RAG | yes | 12.47 ± 0.59 | 66.90 ± 0.50 | 32.28 ± 0.92 | 37.21 ± 0.61 | 1.97 | 3135 |  |
| 7d | Latxa | 3-shot, no RAG | no | 6.44 ± 1.08 | 51.41 ± 1.53 | 34.03 ± 9.70 | 30.62 ± 2.88 | 0.60 | 1927 |  |
| 7d' | Latxa | 3-shot, no RAG | yes | 9.66 ± 0.76 | 65.61 ± 0.17 | 34.25 ± 10.86 | 36.51 ± 3.91 | 1.04 | 2733 |  |
| 8c | Llama | 3-shot + rerank top 3 | no | 15.04 ± 1.82 | 61.46 ± 1.45 | 40.62 ± 7.45 | 39.04 ± 2.82 | 2.15 | 3501 |  |
| 8c' | Llama | 3-shot + rerank top 3 | yes | 20.48 ± 2.07 | 70.29 ± 0.84 | 44.40 ± 3.63 | 45.06 ± 1.86 | 2.81 | 5455 |  |
| 8d | Latxa | 3-shot + E5 top 1 | no | 16.74 ± 0.51 | 65.08 ± 0.90 | 46.03 ± 4.20 | 42.62 ± 1.36 | 0.53 | 2210 |  |
| 8d' | Latxa | 3-shot + E5 top 1 | yes | 16.48 ± 1.05 | 69.42 ± 0.53 | 49.21 ± 5.72 | 45.04 ± 2.29 | 0.90 | 3306 |  |
| 9c | Llama | GuiaSalud retrieval, rerank top 3 | no | 19.89 ± 0.90 | 70.29 ± 0.17 | 20.98 ± 1.82 | 37.05 ± 0.81 | 0.86 | 1840 |  |
| 9c' | Llama | GuiaSalud retrieval, rerank top 3 | yes | 19.32 ± 1.19 | 70.67 ± 0.21 | 20.63 ± 3.17 | 36.88 ± 1.50 | 1.44 | 3838 |  |
| 9d | Latxa | GuiaSalud retrieval, E5 top 1 | no | 15.47 ± 0.57 | 65.89 ± 0.32 | 34.21 ± 1.00 | 38.53 ± 0.60 | 0.44 | 1048 |  |
| 9d' | Latxa | GuiaSalud retrieval, E5 top 1 | yes | 15.17 ± 0.65 | 67.73 ± 0.30 | 31.62 ± 2.07 | 38.17 ± 0.85 | 0.64 | 2167 |  |
| 10c | Llama | CasiMédicos retrieval, rerank top 3 | no | 15.02 ± 1.93 | 66.86 ± 0.48 | 45.40 ± 5.54 | 42.43 ± 2.48 | 1.33 | 2040 |  |
| 10c' | Llama | CasiMédicos retrieval, rerank top 3 | yes | 15.58 ± 1.51 | 67.92 ± 0.17 | 42.52 ± 5.27 | 42.01 ± 2.29 | 1.98 | 4119 |  |
| 10d | Latxa | CasiMédicos retrieval, E5 top 1 | no | 10.76 ± 0.32 | 65.63 ± 0.51 | 52.42 ± 1.65 | 42.94 ± 0.41 | 0.45 | 1137 |  |
| 10d' | Latxa | CasiMédicos retrieval, E5 top 1 | yes | 10.68 ± 0.24 | 66.07 ± 0.11 | 52.13 ± 1.23 | 42.96 ± 0.46 | 0.65 | 2335 |  |

## Reasoning pipelines

Each table compares the model-specific single-pass MeviRAG reference with five inference-only reasoning-pipeline variants. Calls is the mean number of LLM generations per answer.

### Qwen no-think

| # | Pipeline | SF | ROUGE-L | BERT-F1 | MC-acc | MeanQ | Seconds | Tokens | Calls | Selected for test |
|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 3a | Qwen no-think, E5 top 5 | no | 41.53 ± 0.43 | 78.15 ± 0.24 | 83.60 ± 2.42 | 67.76 ± 0.97 | 0.65 | 1930 | 1.0 |  |
| 11a | MedCoT-RAG (our best retrieval) | no | 39.88 ± 0.40 | 77.62 ± 0.13 | 84.13 ± 1.59 | 67.21 ± 0.48 | 1.00 | 2387 | 1.0 |  |
| 12a | MedCoT-RAG (causal top 5) | no | 39.66 ± 2.14 | 77.19 ± 1.00 | 86.24 ± 3.99 | 67.70 ± 2.31 | 0.95 | 2443 | 1.0 |  |
| 13a | RAR² (parallel scaling) | no | 37.76 ± 1.31 | 76.92 ± 0.38 | 84.13 ± 2.75 | 66.27 ± 1.36 | 3.41 | 8562 | 4.0 |  |
| 14a | RAR² (iterative scaling) | no | 35.56 ± 1.05 | 76.05 ± 0.33 | 84.13 ± 3.17 | 65.25 ± 1.16 | 3.87 | 9181 | 3.0 |  |
| 15a | MA-RAG | no | 41.92 ± 1.51 | 78.27 ± 0.42 | 87.30 ± 0.00 | 69.16 ± 0.64 | 2.15 | 3213 | 3.7 | yes |

### Qwen think

| # | Pipeline | SF | ROUGE-L | BERT-F1 | MC-acc | MeanQ | Seconds | Tokens | Calls | Selected for test |
|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 6b | Qwen think, rerank top 5 | no | 41.41 ± 1.18 | 78.10 ± 0.31 | 93.65 ± 0.00 | 71.05 ± 0.50 | 3.92 | 5753 | 1.0 | yes |
| 11b | MedCoT-RAG (our best retrieval) | no | 40.95 ± 1.22 | 77.51 ± 0.32 | 92.59 ± 2.42 | 70.35 ± 1.30 | 4.62 | 6177 | 1.0 |  |
| 12b | MedCoT-RAG (causal top 5) | no | 41.38 ± 1.60 | 77.48 ± 0.50 | 94.18 ± 3.99 | 71.01 ± 1.99 | 4.34 | 6169 | 1.0 |  |
| 13b | RAR² (parallel scaling) | no | 20.85 ± 1.34 | 67.79 ± 0.24 | 66.67 ± 1.59 | 51.77 ± 0.96 | 13.07 | 24694 | 4.0 |  |
| 14b | RAR² (iterative scaling) | no | 31.23 ± 1.76 | 73.83 ± 0.53 | 80.95 ± 4.20 | 62.00 ± 1.80 | 13.63 | 21362 | 3.0 |  |
| 15b | MA-RAG | no | 41.54 ± 0.62 | 78.34 ± 0.05 | 92.06 ± 0.00 | 70.65 ± 0.19 | 12.86 | 16510 | 3.3 |  |

### Llama

| # | Pipeline | SF | ROUGE-L | BERT-F1 | MC-acc | MeanQ | Seconds | Tokens | Calls | Selected for test |
|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 5c | Llama, rerank top 3 | no | 23.78 ± 0.43 | 70.08 ± 0.81 | 48.52 ± 1.86 | 47.46 ± 0.62 | 1.45 | 1795 | 1.0 | yes |
| 11c | MedCoT-RAG (our best retrieval) | no | 17.32 ± 0.46 | 68.87 ± 0.14 | 45.50 ± 6.61 | 43.90 ± 2.29 | 0.99 | 2135 | 1.0 |  |
| 12c | MedCoT-RAG (causal top 5) | no | 16.56 ± 1.76 | 68.78 ± 0.43 | 44.44 ± 8.84 | 43.26 ± 3.47 | 0.83 | 2641 | 1.0 |  |
| 13c | RAR² (parallel scaling) | no | 11.73 ± 1.19 | 62.03 ± 1.89 | 33.89 ± 2.43 | 35.88 ± 1.78 | 6.54 | 4350 | 4.0 |  |
| 14c | RAR² (iterative scaling) | no | 13.22 ± 1.84 | 64.87 ± 1.41 | 40.00 ± 2.56 | 39.37 ± 1.31 | 8.52 | 5237 | 3.0 |  |
| 15c | MA-RAG | no | 17.39 ± 1.56 | 66.81 ± 1.52 | 44.47 ± 3.05 | 42.89 ± 0.13 | 7.14 | 3937 | 4.4 |  |

### Latxa

| # | Pipeline | SF | ROUGE-L | BERT-F1 | MC-acc | MeanQ | Seconds | Tokens | Calls | Selected for test |
|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1d | Latxa, E5 top 1 | no | 15.81 ± 0.63 | 68.51 ± 0.23 | 50.54 ± 1.24 | 44.95 ± 0.68 | 0.18 | 1047 | 1.0 | yes |
| 11d | MedCoT-RAG (our best retrieval) | no | 14.19 ± 1.39 | 67.31 ± 0.28 | 51.32 ± 2.42 | 44.28 ± 1.24 | 1.03 | 1530 | 1.0 |  |
| 12d | MedCoT-RAG (causal top 5) | no | 15.21 ± 0.50 | 68.21 ± 0.19 | 49.74 ± 4.58 | 44.38 ± 1.48 | 1.98 | 2805 | 1.0 |  |
| 13d | RAR² (parallel scaling) | no | 10.53 ± 0.78 | 62.62 ± 0.76 | 44.97 ± 4.85 | 39.37 ± 2.01 | 6.10 | 5518 | 4.0 |  |
| 14d | RAR² (iterative scaling) | no | 12.43 ± 1.24 | 65.04 ± 0.97 | 49.21 ± 7.27 | 42.22 ± 3.03 | 9.58 | 5153 | 3.0 |  |
| 15d | MA-RAG | no | 12.36 ± 0.43 | 66.74 ± 0.31 | 44.68 ± 0.41 | 41.26 ± 0.14 | 0.82 | 1322 | 3.4 |  |
