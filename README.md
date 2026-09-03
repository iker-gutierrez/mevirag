# MeviRAG: evidence-grounded RAG for the medical domain

Retrieval-augmented generation (RAG) pipeline for clinical question answering in Spanish and Basque.
The system is evaluated on two tasks: open-answer clinical QA (GuiaSalud) and multiple-choice medical exam QA (CasiMédicos-Exp), across four core generator configurations (Qwen3.5-9B in non-thinking and thinking mode, Llama-3.1-8B-Instruct, and its Basque-adapted counterpart Latxa-Llama-3.1-8B-Instruct) and an eleven-condition ablation grid varying retrieval depth, cross-encoder reranking, few-shot prompting, self-feedback, and domain restriction, plus five inference-only reasoning-pipeline variants drawn from recent literature.

Full system implementation: retrieval, generation, self-feedback, reasoning pipelines, and evaluation.

## Status

- **Spanish and Basque dev ablations**: complete, all eleven conditions, three seeds each, decided by MeanQ (mean of ROUGE-L, BERT-F1, MC-accuracy, see `scripts/meanq.py`). Retrieval uses the full corpus while excluding each query's own gold instance at retrieval time, preventing answer leakage without discarding the remaining corpus.
- **Reasoning-pipeline comparison**: complete for both languages, including a second Basque backbone (Llama) alongside Latxa, and both Qwen3.5-9B modes.
- **Test set evaluation**: complete. Each model's own dev-set MeanQ-best configuration was frozen and run once against the held-out test split.

## Key findings

- Retrieval helps every model in both languages substantially. Reranking and few-shot prompting each help only narrowly and inconsistently, and are actively harmful for the weakest models in each language. Self-feedback is neutral for three of the four models but a genuine, if modest, gain for the Basque-adapted model specifically.
- Of the five reasoning-pipeline variants, two show a real gain over the single-pass RAG ablation winner, and only for one of the two Spanish configurations tested, at several times the inference cost. Every pipeline underperforms the ablation winner in Basque, and for the other Spanish configuration.
- A persistent 20-26 point MeanQ gap between the best achievable Spanish and Basque configurations survives every technique tested on dev and remains substantial on the held-out test set (21.94-24.16 points across the directly comparable systems).
- Basque language adaptation (Latxa vs. Llama) does not raise overall single-pass MeanQ above the non-adapted model's, but it does raise multiple-choice accuracy specifically, and it is the only technique tested for which the Basque-adapted model shows a genuine advantage: a positive self-feedback gain that does not extend to multi-step reasoning.

## Test results

Each model's own best dev-set configuration was frozen and run once against the held-out test split. MeanQ is the mean of ROUGE-L, BERT-F1, and MC-accuracy.

| Model | Baseline (LLM only) | Best RAG config | Best RAG MeanQ | $\Delta$ |
|---|---|---|---|---|
| Qwen3.5-9B (no-think) | 57.97±0.86 | MA-RAG | 65.79±0.30 | +7.82±1.07 |
| Qwen3.5-9B (think) | 63.46±0.50 | rerank top 5 | 67.42±0.35 | +3.96±0.19 |
| Llama-3.1-8B-Instruct | 33.99±1.30 | rerank top 3 | 43.85±2.35 | +9.86±1.30 |
| Latxa-Llama-3.1-8B-Instruct | 37.15±0.37 | retrieve top 1 | 43.26±0.52 | +6.11±0.78 |

For Qwen no-think, MA-RAG was the only one of the five reasoning-pipeline variants to beat its own model's RAG ablation winner on dev, so it is the frozen test-set configuration. The other three models carry forward their own single-pass ablation winner instead. Full per-condition results, including cost (seconds/tokens per sample) and self-feedback deltas, are in the ablation reports linked below.

## Repository layout

- `data/raw/`: original datasets, kept out of git.
- `data/interim/`: temporary converted files.
- `data/processed/`: normalized JSONL/CSV splits (Spanish and Basque).
- `src/medical_rag_thesis/`: reusable experiment code (retrieval, generation, evaluation, reasoning pipelines).
- `scripts/`: command-line entry points for data prep, experiments, staged ablation, and result-table/report generation.
- `slurm/`: Slurm job scripts, including the staged-ablation launchers (`slurm/staged_*.sh`).
- `configs/experiments/`: per-run experiment configs (retrieval depth, reranking, few-shot, self-feedback, and reasoning pipeline). The final held-out test configurations are also copied, with their predictions and evaluations, to `reproducibility/final_test/`.
- `experiments/runs/`: generated predictions and run artifacts (gitignored).
- `reports/metrics/`: ablation result tables and summaries.
- `docs/`: current prompt reference (`prompts.md`), supervisor meeting notes, reading list, bibliography notes.

The manuscript itself (LaTeX source and compiled PDF) is kept outside this repository and is not tracked in git.

## Ablation results

- [Spanish dev ablation results](reports/metrics/es_dev_ablation_results.md)
- [Basque dev ablation results](reports/metrics/eu_dev_ablation_results.md)

## Quick start

Install the package in editable mode:

```bash
python -m pip install -e .
```

### Data preparation

GuiaSalud (Spanish and Basque) is built and published by the separate [guiasalud](https://github.com/iker-gutierrez/guiasalud) repository, which owns the fixed train/dev/test split and the Basque translation. Download it from the [Hugging Face dataset](https://huggingface.co/datasets/ikergf/guiasalud) into `data/processed/guiasalud` (Spanish) and `data/processed/guiasalud_eu` (Basque).

Import CasiMédicos-Exp from Hugging Face (`HiTZ/casimedicos-exp`):

```bash
python scripts/import_casimedicos_exp.py \
  --raw-dir data/raw/casimedicos_exp \
  --output-dir data/processed/casimedicos
```

Create the mixed dataset (GuiaSalud + CasiMédicos-Exp):

```bash
python scripts/build_guiasalud_casimedicos.py
```

### Retrieval index

The retrieval corpus is the full corpus (train, dev and test together), with each query's own gold instance excluded at query time rather than by corpus-splitting:

```bash
python scripts/build_retrieval_index.py \
  --input data/processed/guiasalud_casimedicos/all.jsonl \
  --output-dir models/retrieval/guiasalud_casimedicos_full_multilingual_e5_large
```

### Generation experiment

```bash
python scripts/run_generation_experiment.py \
  --input data/processed/guiasalud_casimedicos/dev.jsonl \
  --output experiments/runs/qwen9b_rerank5_3shot_noSF/predictions.jsonl \
  --model Qwen/Qwen3.5-9B \
  --experiment-name qwen9b_rerank5_3shot_noSF \
  --retrieval-index models/retrieval/guiasalud_casimedicos_full_multilingual_e5_large \
  --retrieval-top-k 15 \
  --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --reranker-top-k 5 \
  --few-shot-file data/processed/guiasalud_casimedicos/train.jsonl \
  --few-shot-k 3
```

In practice, most experiments are launched from a JSON config in `configs/experiments/` via `scripts/run_generation_from_config.py`, which translates config fields into the CLI flags above. See any file under `configs/experiments/` for the full field list.

### Evaluation

```bash
python scripts/evaluate_predictions.py \
  --predictions experiments/runs/qwen9b_rerank5_3shot_noSF/predictions.jsonl \
  --output reports/metrics/qwen9b_rerank5_3shot_noSF.json \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es
```

### Reasoning pipelines

```bash
python scripts/run_reasoning_pipeline.py \
  --config configs/experiments/1530_qwen35_9b_no_think_structured_cot_meanq_best_extractive_mixed_dev.json
```

## Slurm runs

The staged ablation grid (retrieval depth -> reranking -> few-shot -> domain restriction) is launched per model/language via the `slurm/staged_*.sh` scripts, e.g.:

```bash
sbatch slurm/staged_qwen35_9b_no_think_A.sh
sbatch slurm/staged_latxa_A.sh
```

Retrieval indices are rebuilt with:

```bash
sbatch slurm/rebuild_full_corpus_indices.sh
```

Slurm logs go to `experiments/slurm_logs/`.
