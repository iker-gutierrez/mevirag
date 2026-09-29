# MeviRAG: evidence-grounded RAG for the medical domain

This repository contains the completed implementation and evaluation of MeviRAG, a retrieval-augmented generation (RAG) pipeline for clinical question answering in Spanish and Basque.
The system is evaluated on two tasks: open-answer clinical QA (GuiaSalud) and multiple-choice medical exam QA (CasiMédicos-Exp), across four core generator configurations (Qwen3.5-9B in non-thinking and thinking mode, Llama-3.1-8B-Instruct, and its Basque-adapted counterpart Latxa-Llama-3.1-8B-Instruct) and an eleven-condition ablation grid varying retrieval depth, cross-encoder reranking, few-shot prompting, self-feedback, and retrieval-corpus scope, plus five inference-only reasoning-pipeline variants drawn from recent literature.

It includes retrieval, generation, self-feedback, reasoning pipelines, evaluation, and the final reproducibility artefacts. Retrieval uses the full corpus while excluding each query's own gold instance at query time.

## Key findings

- **Retrieval and ablations.** Retrieval helps every model substantially. Reranking is model-specific, while few-shot prompting and corpus restriction do not provide consistent gains. Self-feedback is close to neutral for both Qwen settings and improves both Basque models on average, but the final MST-selected Basque configurations do not use it.
- **Reasoning pipelines.** Of the five reasoning-pipeline variants, only MA-RAG improves over its model's selected single-pass MeviRAG reference, and only for Qwen no-think. Every reasoning pipeline underperforms the selected reference for Qwen think, Llama, and Latxa while generally increasing inference cost.
- **Spanish--Basque gap.** A persistent 20-26 point MeanQ gap between the best achievable Spanish and Basque configurations survives every technique tested on dev and remains substantial on the held-out test set (21.94-24.16 points across the directly comparable systems).
- **Basque adaptation.** Basque-specific continued pre-training provides targeted benefits for Latxa, particularly in retrieval-free generation, self-feedback, and multiple-choice accuracy, but it does not produce a uniform advantage over Llama once retrieval is used.

## Test results

Each model's development-selected configuration was frozen and evaluated on the held-out test split. MeanQ is the mean of ROUGE-L, BERT-F1, and MC-accuracy.

| Model | Baseline (LLM only) | Selected system | Selected MeanQ | $\Delta$ |
|---|---|---|---|---|
| Qwen3.5-9B (no-think) | 57.97±0.86 | MA-RAG | 65.79±0.30 | +7.82±1.07 |
| Qwen3.5-9B (think) | 63.46±0.50 | rerank top 5 | 67.42±0.35 | +3.96±0.19 |
| Llama-3.1-8B-Instruct | 33.99±1.30 | rerank top 3 | 43.85±2.35 | +9.86±1.30 |
| Latxa-Llama-3.1-8B-Instruct | 37.15±0.37 | retrieve top 1 | 43.26±0.52 | +6.11±0.78 |

For Qwen no-think, MA-RAG was the only one of the five reasoning-pipeline variants to beat its own model's RAG ablation winner on dev, so it is the frozen test-set configuration. The other three models carry forward their own single-pass ablation winner instead. The complete [development](reproducibility/development_results.md) and [held-out test](reproducibility/final_test/results.md) reports include the per-condition quality and cost results.

## Repository layout

```text
.
├── configs/
│   ├── data/                  # Dataset and split configuration
│   └── experiments/           # Retrieval, generation, and reasoning configurations
├── data/
│   ├── raw/                   # Original datasets (not tracked)
│   ├── interim/               # Temporary converted data (not tracked)
│   └── processed/             # Normalized Spanish and Basque splits (not tracked)
├── docs/                      # Prompt and evaluation documentation
├── reproducibility/
│   ├── development_results.md # Final development results
│   └── final_test/            # Test configs, predictions, evaluations, and results
├── scripts/                   # Data, experiment, evaluation, and reporting commands
├── slurm/                     # Slurm workflows for the reported experiments
├── src/mevirag/               # Reusable MeviRAG implementation
├── tests/                     # Automated tests
├── experiments/runs/          # Generated run artifacts (not tracked)
└── reports/metrics/           # Generated metric outputs (not tracked)
```

## Reproducibility artefacts

The complete development ablation and reasoning-pipeline results are available
in [`reproducibility/development_results.md`](reproducibility/development_results.md).
The exact final-test configurations, predictions, evaluation outputs, and
consolidated results are available under
[`reproducibility/final_test/`](reproducibility/final_test/).

## Quick start

Install the package in editable mode:

```bash
python -m pip install -e .
```

The final GPU experiments use vLLM. Install a vLLM release compatible with
the local CUDA and PyTorch environment before running the published configs or
reasoning pipelines.

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
python scripts/evaluate_predictions_by_source.py \
  --predictions experiments/runs/qwen9b_rerank5_3shot_noSF/predictions.jsonl \
  --output reports/metrics/qwen9b_rerank5_3shot_noSF.json \
  --semantic-model '' \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es
```

### Reasoning pipelines

```bash
python scripts/run_reasoning_pipeline.py \
  --config configs/experiments/16000_qwen35_9b_no_think_structured_cot_e5_topk5_extractive_guiasalud_dev_costaware.json
```

## Slurm runs

The staged ablation grid is launched separately for Spanish and Basque. Each
evaluation stage produces the selections required by dependent later stages.
For example:

```bash
sbatch slurm/spanish_ablation_generation_stageA.sh
sbatch slurm/basque_ablation_generation_stageA.sh
```

The full mixed Spanish retrieval index can be rebuilt with:

```bash
python scripts/build_retrieval_index.py \
  --input data/processed/guiasalud_casimedicos/all.jsonl \
  --output-dir models/retrieval/guiasalud_casimedicos_full_multilingual_e5_large
```

Slurm logs go to `experiments/slurm_logs/`.

## Citation

If you use MeviRAG or this repository, please cite the Master's thesis that
introduces the system and reports its evaluation:

```bibtex
@mastersthesis{gutierrezfandino2026mevirag,
  author = {Gutierrez Fandiño, Iker},
  title  = {{GuiaSalud dataset and MeviRAG: Towards evidence-grounded medical QA in Spanish and Basque}},
  school = {University of the Basque Country (EHU)},
  year   = {2026},
  type   = {Master's thesis}
}
```

## License

This repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

## Contact

For questions, contact Iker Gutierrez Fandiño at
[ikergutierrezfandino@gmail.com](mailto:ikergutierrezfandino@gmail.com).
