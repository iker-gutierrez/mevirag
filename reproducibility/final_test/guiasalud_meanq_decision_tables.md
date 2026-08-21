
### Qwen3.5-9B (no-think)

| Rank | Experiment | ROUGE-L | BERT-F1 | MC-acc | **MeanQ** |
|---:|---|---:|---:|---:|---:|
| **1** | **domain: GuiaSalud only (SF)** | 34.68 | 75.16 | 80.42 | **63.42** |
| 2 | domain: GuiaSalud only | 33.75 | 75.04 | 80.95 | 63.25 |
| 3 | 3-shot + rerank top 5 (SF) | 28.78 | 72.85 | 82.54 | 61.39 |
| 4 | rerank top 5 (SF) | 27.11 | 71.65 | 85.19 | 61.31 |
| 5 | Baseline LLM only | 27.30 | 71.96 | 84.13 | 61.13 |
| 6 | rerank top 1 (SF) | 27.23 | 71.73 | 84.13 | 61.03 |
| 7 | domain: CasiMedicos only (SF) | 26.90 | 71.50 | 84.66 | 61.02 |
| 8 | rerank top 3 | 27.55 | 72.06 | 83.07 | 60.89 |
| 9 | Baseline LLM only (SF) | 26.99 | 71.57 | 83.60 | 60.72 |
| 10 | rerank top 1 | 27.27 | 71.90 | 82.54 | 60.57 |
| 11 | 3-shot, no RAG (SF) | 27.61 | 72.43 | 81.48 | 60.51 |
| 12 | rerank top 5 | 26.99 | 71.92 | 82.01 | 60.31 |
| 13 | e5 top 1 (SF) | 27.05 | 71.63 | 82.01 | 60.23 |
| 14 | rerank top 3 (SF) | 26.70 | 71.56 | 82.01 | 60.09 |
| 15 | domain: CasiMedicos only | 26.78 | 71.72 | 80.95 | 59.82 |
| 16 | e5 top 5 (SF) | 26.45 | 71.45 | 80.95 | 59.62 |
| 17 | 3-shot, no RAG | 27.61 | 71.46 | 79.37 | 59.48 |
| 18 | e5 top 5 | 26.50 | 71.68 | 79.89 | 59.36 |
| 19 | e5 top 1 | 26.62 | 71.62 | 79.37 | 59.20 |
| 20 | e5 top 3 (SF) | 26.19 | 71.37 | 79.37 | 58.97 |
| 21 | 3-shot + rerank top 5 | 27.33 | 69.38 | 78.31 | 58.34 |
| 22 | e5 top 3 | 26.14 | 71.44 | 77.25 | 58.28 |

**Chosen config: domain: GuiaSalud only (SF)** (MeanQ 63.42), the highest MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) over 3 seeds on either the no-self-feedback or self-feedback prediction, whichever scores higher for that row (candidates labeled "(SF)" are the self-feedback variant). MeanQ is used instead of any single metric so a config is only chosen if it does well on lexical, semantic, and decision correctness together.


### Qwen3.5-9B (think)

| Rank | Experiment | ROUGE-L | BERT-F1 | MC-acc | **MeanQ** |
|---:|---|---:|---:|---:|---:|
| **1** | **3-shot + rerank top 5** | 32.21 | 75.18 | 92.59 | **66.66** |
| 2 | rerank top 3 (SF) | 29.92 | 73.23 | 95.77 | 66.31 |
| 3 | 3-shot, no RAG | 32.47 | 75.39 | 91.01 | 66.29 |
| 4 | rerank top 3 | 29.94 | 73.10 | 95.77 | 66.27 |
| 5 | Baseline LLM only (SF) | 30.61 | 73.83 | 94.18 | 66.20 |
| 6 | domain: GuiaSalud only | 35.84 | 75.86 | 85.71 | 65.80 |
| 7 | Baseline LLM only | 30.30 | 73.60 | 93.12 | 65.67 |
| 8 | e5 top 3 | 29.48 | 72.94 | 93.65 | 65.36 |
| 9 | e5 top 3 (SF) | 29.28 | 72.92 | 93.65 | 65.28 |
| 10 | domain: CasiMedicos only | 29.41 | 73.13 | 93.12 | 65.22 |
| 11 | e5 top 5 | 29.14 | 72.93 | 93.12 | 65.06 |
| 12 | e5 top 5 (SF) | 29.02 | 72.87 | 92.59 | 64.83 |
| 13 | rerank top 5 | 28.96 | 72.86 | 92.59 | 64.80 |
| 14 | rerank top 5 (SF) | 28.90 | 72.88 | 92.59 | 64.79 |
| 15 | rerank top 1 | 29.15 | 72.92 | 92.06 | 64.71 |
| 16 | rerank top 1 (SF) | 29.08 | 72.91 | 92.06 | 64.68 |
| 17 | e5 top 1 | 28.98 | 72.90 | 91.53 | 64.47 |
| 18 | e5 top 1 (SF) | 28.88 | 72.86 | 91.53 | 64.42 |

**Chosen config: 3-shot + rerank top 5** (MeanQ 66.66), the highest MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) over 3 seeds on either the no-self-feedback or self-feedback prediction, whichever scores higher for that row (candidates labeled "(SF)" are the self-feedback variant). MeanQ is used instead of any single metric so a config is only chosen if it does well on lexical, semantic, and decision correctness together.


### Llama-3.1-8B-Instruct

| Rank | Experiment | ROUGE-L | BERT-F1 | MC-acc | **MeanQ** |
|---:|---|---:|---:|---:|---:|
| **1** | **Baseline LLM only** | 4.05 | 64.78 | 24.87 | **31.23** |
| 2 | Baseline LLM only (SF) | 4.31 | 63.00 | 25.40 | 30.90 |
| 3 | domain: GuiaSalud only | 6.29 | 65.87 | 20.11 | 30.76 |
| 4 | 3-shot, no RAG (SF) | 5.34 | 65.20 | 21.27 | 30.60 |
| 5 | rerank top 1 (SF) | 5.50 | 65.30 | 20.63 | 30.48 |
| 6 | e5 top 5 | 5.01 | 65.73 | 20.63 | 30.46 |
| 7 | rerank top 1 | 5.27 | 65.43 | 20.63 | 30.44 |
| 8 | e5 top 5 (SF) | 4.98 | 65.28 | 20.63 | 30.30 |
| 9 | 3-shot + rerank top 5 | 5.91 | 65.69 | 18.93 | 30.18 |
| 10 | e5 top 1 | 4.58 | 65.28 | 20.63 | 30.17 |
| 11 | e5 top 3 | 5.19 | 66.02 | 19.05 | 30.08 |
| 12 | 3-shot, no RAG | 5.73 | 65.12 | 19.02 | 29.96 |
| 13 | domain: GuiaSalud only (SF) | 6.01 | 65.61 | 17.99 | 29.87 |
| 14 | domain: CasiMedicos only | 5.08 | 65.27 | 19.05 | 29.80 |
| 15 | rerank top 3 | 5.60 | 66.30 | 17.46 | 29.79 |
| 16 | e5 top 1 (SF) | 4.80 | 64.82 | 19.58 | 29.73 |
| 17 | 3-shot + rerank top 5 (SF) | 5.71 | 66.25 | 15.87 | 29.28 |
| 18 | domain: CasiMedicos only (SF) | 5.35 | 65.86 | 16.40 | 29.20 |
| 19 | rerank top 3 (SF) | 5.42 | 65.55 | 15.87 | 28.95 |
| 20 | e5 top 3 (SF) | 4.78 | 65.95 | 14.29 | 28.34 |
| 21 | rerank top 5 | 4.18 | 64.86 | 15.87 | 28.30 |
| 22 | rerank top 5 (SF) | 4.73 | 64.28 | 15.34 | 28.12 |

**Chosen config: Baseline LLM only** (MeanQ 31.23), the highest MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) over 3 seeds on either the no-self-feedback or self-feedback prediction, whichever scores higher for that row (candidates labeled "(SF)" are the self-feedback variant). MeanQ is used instead of any single metric so a config is only chosen if it does well on lexical, semantic, and decision correctness together.


### Latxa-Llama-3.1-8B-Instruct

| Rank | Experiment | ROUGE-L | BERT-F1 | MC-acc | **MeanQ** |
|---:|---|---:|---:|---:|---:|
| **1** | **3-shot, no RAG (SF)** | 5.54 | 63.75 | 25.40 | **31.56** |
| 2 | Baseline LLM only (SF) | 4.89 | 63.50 | 24.87 | 31.08 |
| 3 | 3-shot + rerank top 5 (SF) | 4.93 | 65.17 | 19.58 | 29.89 |
| 4 | rerank top 5 (SF) | 3.36 | 64.53 | 21.69 | 29.86 |
| 5 | domain: CasiMedicos only (SF) | 4.93 | 64.23 | 20.11 | 29.76 |
| 6 | domain: CasiMedicos only | 5.19 | 64.34 | 19.58 | 29.70 |
| 7 | rerank top 5 | 4.63 | 63.71 | 20.42 | 29.58 |
| 8 | e5 top 1 (SF) | 3.22 | 62.47 | 22.22 | 29.31 |
| 9 | e5 top 5 (SF) | 4.35 | 64.29 | 19.05 | 29.23 |
| 10 | e5 top 5 | 4.14 | 64.32 | 19.05 | 29.17 |
| 11 | e5 top 3 | 4.63 | 64.68 | 17.75 | 29.02 |
| 12 | e5 top 3 (SF) | 4.86 | 64.85 | 16.40 | 28.70 |
| 13 | rerank top 3 (SF) | 3.94 | 64.43 | 15.87 | 28.08 |
| 14 | rerank top 1 (SF) | 4.82 | 64.93 | 13.76 | 27.84 |
| 15 | rerank top 1 | 4.82 | 63.07 | 14.70 | 27.53 |
| 16 | domain: GuiaSalud only (SF) | 4.94 | 63.27 | 13.23 | 27.15 |
| 17 | domain: GuiaSalud only | 3.97 | 64.18 | 11.11 | 26.42 |
| 18 | Baseline LLM only | 4.09 | 48.71 | 25.93 | 26.24 |
| 19 | 3-shot + rerank top 5 | 4.57 | 52.15 | 21.16 | 25.96 |
| 20 | rerank top 3 | 3.84 | 59.23 | 14.29 | 25.79 |
| 21 | e5 top 1 | 2.92 | 48.18 | 22.46 | 24.52 |
| 22 | 3-shot, no RAG | 2.95 | 41.85 | 28.57 | 24.46 |

**Chosen config: 3-shot, no RAG (SF)** (MeanQ 31.56), the highest MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) over 3 seeds on either the no-self-feedback or self-feedback prediction, whichever scores higher for that row (candidates labeled "(SF)" are the self-feedback variant). MeanQ is used instead of any single metric so a config is only chosen if it does well on lexical, semantic, and decision correctness together.
