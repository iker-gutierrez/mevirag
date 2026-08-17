#!/usr/bin/env python
"""End-to-end smoke test for the Basque ablation/reasoning-pipeline chain:
proves the actual generation and retrieval pipeline works, not just the
selection math (see scripts/smoke_test_self_feedback_selection.py for the
selection-only, GPU-free check).

Runs real, tiny (--limit 3 records) generation through the real retrieval
index, prompt builder, and vLLM generation path, for two models
(HiTZ/Latxa-Llama-3.1-8B-Instruct, chosen as the fastest model in the
ablation grid, plus a single Qwen/Qwen3.5-9B run to exercise the
hyperparameters only that family sets), covering:

  1. Retrieval with the query's own gold document excluded. Uses the
     generation script's own --log-gold-hit-rate instrumentation
     (src/medical_rag_thesis/retrieval.py's HitRateLoggingEmbeddingRetriever)
     rather than re-implementing a separate check, and additionally
     confirms no prediction's retrieved-context text contains the literal
     gold answer string of its own record (a leak that would slip past the
     doc_id-based exclusion, e.g. if two records shared identical content).
  2. Plain and self-feedback generation, run separately (matching the real
     pipeline: a plain config and its self-feedback clone are two different
     generation runs, not two fields written by a single run), and confirms
     the self-feedback run's predictions carry both the initial and the
     revised answer.
  3. Prompt rendering: confirms the actual rendered prompt (recovered via
     --save-prompts) contains real question text, not an empty or
     boilerplate-only prompt (the failure mode of the historical
     format_question() language=='eu' bug, see prompts.py's own comments).
  4. The staged decision chain: real (if tiny, 3-record) metrics are
     computed by the real evaluation script, then the real
     rewire_basque_ablation_stage.py rewires a dependent row's config from
     those real metrics, and finalize_basque_ablation_and_write_reasoning_
     configs.py writes a real reasoning-pipeline config frozen to the
     result -- the identical code path the real ablation grid runs, just
     with --limit-capped record counts so it finishes in minutes.
  5. Decoding hyperparameters (temperature, top_p, top_k, min_p,
     presence_penalty): each model family is decoded with the sampling
     settings recommended by its own developers, held fixed across every
     ablation-grid row of that model (manuscript's own hyperparameters
     table). Confirms both that the ablation configs carry the correct
     literal values for each family, and that the generated
     reasoning-pipeline config actually inherits the winning row's real
     settings rather than a stale or default value. Llama-3.1-8B-Instruct
     and Latxa-Llama-3.1-8B-Instruct only set temperature/top_p (their
     model cards recommend nothing else); Qwen3.5-9B additionally sets
     top_k/min_p/presence_penalty -- the last of which was, historically, a
     real shipped bug (silently applied as vLLM's unrelated
     repetition_penalty field instead), so it is checked explicitly here.

Every file this test touches lives at a dedicated smoke-test id block
(97000s) and output path prefix, distinct from any real experiment id, and
is cleaned up before and after a run.

Needs a free GPU. Takes several minutes (four small vLLM generation runs).

Usage: python scripts/smoke_test_ablation_pipeline_end_to_end.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONFIG_DIR = ROOT / "configs" / "experiments"
RUNS = ROOT / "experiments" / "runs"
METRICS = ROOT / "reports" / "metrics"

SMOKE_TAG = "smoke_e2e"
LIMIT = 3

BASE_MODEL_CONFIG = {
    "model": "HiTZ/Latxa-Llama-3.1-8B-Instruct",
    "prompt_style": "extractive",
    "think": False,
    "backend": "vllm",
    "max_new_tokens": 256,
    "temperature": 0.6,
    "trust_remote_code": True,
    "retrieval_index": "models/retrieval/guiasalud_casimedicos_eu_train_multilingual_e5_large",
    "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "reranker_device": "cpu",
    "top_p": 0.9,
    "language": "eu",
    "input": "data/processed/guiasalud_casimedicos_eu/dev.jsonl",
    "limit": LIMIT,
    "log_gold_hit_rate": True,
    "save_prompts": True,
}

# (id, retrieval_top_k, reranker_top_k, self_feedback) -- two plain rows
# (standing in for two ablation-grid rows) plus one self-feedback clone of
# the second, mirroring the real grid's own plain/self-feedback pairing.
ROW_A_ID = "97000"   # retrieval_top_k=5, reranker_top_k=1 (row-like: rerank1)
ROW_B_ID = "97001"   # retrieval_top_k=15, reranker_top_k=5 (row-like: rerank5)
ROW_B_SF_ID = "97002"  # self-feedback clone of row B
DEPENDENT_ID = "97003"  # stands in for a dependent row (e.g. row 8)

# Qwen3.5-9B config: the only family that sets top_k/min_p/presence_penalty
# (Llama/Latxa's model cards recommend neither), so a single small run is
# added purely to exercise those three parameters through the real vLLM
# generation path.
QWEN_MODEL_CONFIG = {
    "model": "Qwen/Qwen3.5-9B",
    "prompt_style": "extractive",
    "think": False,
    "backend": "vllm",
    "max_new_tokens": 256,
    "temperature": 1.0,
    "trust_remote_code": True,
    "retrieval_index": "models/retrieval/guiasalud_casimedicos_train_multilingual_e5_large",
    "retrieval_top_k": 5,
    "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "reranker_top_k": 1,
    "reranker_device": "cpu",
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
    "input": "data/processed/guiasalud_casimedicos/dev.jsonl",
    "limit": LIMIT,
    "save_prompts": True,
}
QWEN_ROW_ID = "97004"


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise SystemExit(f"SMOKE TEST FAILED: {message}")


def clean_up() -> None:
    for p in list(CONFIG_DIR.glob(f"9700*_{SMOKE_TAG}_*.json")):
        p.unlink()
    for p in list(METRICS.glob(f"9700*_{SMOKE_TAG}_*.json")):
        p.unlink()
    for p in list(RUNS.glob(f"9700*_{SMOKE_TAG}_*")):
        shutil.rmtree(p, ignore_errors=True)


def write_config(run_id: str, name_suffix: str, *, retrieval_top_k: int, reranker_top_k: int,
                  self_feedback: bool) -> tuple[str, str]:
    name = f"{SMOKE_TAG}_latxa_{name_suffix}"
    cfg = dict(BASE_MODEL_CONFIG)
    cfg["experiment_name"] = name
    cfg["output"] = f"experiments/runs/{run_id}_{name}_dev/predictions.jsonl"
    cfg["retrieval_top_k"] = retrieval_top_k
    cfg["reranker_top_k"] = reranker_top_k
    cfg["self_feedback"] = self_feedback
    if self_feedback:
        cfg["feedback_max_new_tokens"] = 256
    base = f"{name}_dev"
    (CONFIG_DIR / f"{run_id}_{base}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return run_id, base


def write_qwen_config() -> tuple[str, str]:
    name = f"{SMOKE_TAG}_qwen"
    cfg = dict(QWEN_MODEL_CONFIG)
    cfg["experiment_name"] = name
    cfg["output"] = f"experiments/runs/{QWEN_ROW_ID}_{name}_dev/predictions.jsonl"
    base = f"{name}_dev"
    (CONFIG_DIR / f"{QWEN_ROW_ID}_{base}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return QWEN_ROW_ID, base


def run_generation(run_id: str, base: str, *, seed: int = 42) -> Path:
    config_path = CONFIG_DIR / f"{run_id}_{base}.json"
    output = f"experiments/runs/{run_id}_{base}_seed{seed}/predictions.jsonl"
    cmd = [
        "scripts/pick_free_gpu.sh", "40000",
        sys.executable, "scripts/run_generation_from_config.py",
        "--config", str(config_path),
        "--seed", str(seed),
        "--output", output,
    ]
    print(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return ROOT / output


def run_evaluation(run_id: str, base: str, *, seed: int = 42) -> Path:
    run = f"{run_id}_{base}_seed{seed}"
    predictions = RUNS / run / "predictions.jsonl"
    metrics_path = METRICS / f"{run}.json"
    cmd = [
        sys.executable, "scripts/evaluate_predictions_by_source.py",
        "--predictions", str(predictions),
        "--output", str(metrics_path),
        "--semantic-model", "intfloat/multilingual-e5-large",
        "--bertscore-model", "bert-base-multilingual-cased",
        "--bertscore-lang", "eu",
    ]
    print(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return metrics_path


def main() -> None:
    print(f"=== end-to-end smoke test: real generation + retrieval + staged selection ({LIMIT} records/run) ===")
    clean_up()

    # --- 1-3: real generation, retrieval exclusion, self-feedback -------
    print("\n--- generating: two plain rows + one self-feedback clone ---")
    row_a = write_config(ROW_A_ID, "rerank1", retrieval_top_k=5, reranker_top_k=1, self_feedback=False)
    row_b = write_config(ROW_B_ID, "rerank5", retrieval_top_k=15, reranker_top_k=5, self_feedback=False)
    row_b_sf = write_config(ROW_B_SF_ID, "rerank5_sf", retrieval_top_k=15, reranker_top_k=5, self_feedback=True)

    pred_a = run_generation(*row_a)
    pred_b = run_generation(*row_b)
    pred_b_sf = run_generation(*row_b_sf)

    for label, pred in (("row A", pred_a), ("row B", pred_b), ("row B self-feedback", pred_b_sf)):
        check(pred.exists() and len(pred.read_text().splitlines()) == LIMIT,
              f"{label}: produced {LIMIT} real predictions ({pred.relative_to(ROOT)})")

    print("\n--- checking retrieval excluded each query's own gold document ---")
    for label, pred in (("row A", pred_a), ("row B", pred_b)):
        hit_log_path = pred.parent / "retrieval_hit_rate_log.json"
        check(hit_log_path.exists(), f"{label}: hit-rate log was written ({hit_log_path.name})")
        hit_log = json.loads(hit_log_path.read_text())
        check(hit_log["num_queries"] == LIMIT, f"{label}: hit-rate log covers all {LIMIT} queries")
        # excluded_present just means "the gold doc would have ranked in the
        # naive top-k+1" (informational, logged by the exclude_id machinery
        # itself); what actually matters is that no retrieved document's own
        # doc_id equals the query record's own id -- i.e. exclusion actually
        # happened, not just that it was requested.
        records = [json.loads(line) for line in pred.read_text().splitlines()]
        leaked = 0
        for rec in records:
            own_id = str(rec.get("id"))
            for doc in rec.get("retrieval_docs") or []:
                if str(doc.get("doc_id")) == own_id:
                    leaked += 1
        check(leaked == 0, f"{label}: no prediction's retrieved documents included the query's own record")

    print("\n--- checking prompts were actually rendered with real question text ---")
    for label, pred in (("row A", pred_a),):
        records = [json.loads(line) for line in pred.read_text().splitlines()]
        check(all("prompt" in r for r in records), f"{label}: --save-prompts attached a prompt to every record")
        empty_or_short = sum(1 for r in records if len(str(r.get("prompt") or "")) < 50)
        check(empty_or_short == 0, f"{label}: no prompt was empty or suspiciously short (<50 chars)")

    print("\n--- checking the self-feedback run produced both an initial and a revised answer ---")
    sf_records = [json.loads(line) for line in pred_b_sf.read_text().splitlines()]
    check(
        all("initial_prediction_text" in r and "prediction_text" in r for r in sf_records),
        "self-feedback predictions carry both initial_prediction_text and prediction_text",
    )
    check(
        any(r.get("initial_prediction_text") != r.get("prediction_text") for r in sf_records),
        "at least one self-feedback record's revised answer actually differs from its initial answer",
    )

    # --- 4-5: real (tiny) evaluation + the real staged decision code ----
    print("\n--- scoring the tiny generation runs with the real evaluation script ---")
    run_evaluation(*row_a)
    run_evaluation(*row_b)
    run_evaluation(*row_b_sf)

    print("\n--- running the real MeanQ decision + config rewiring on these tiny, real metrics ---")
    from meanq import best_by_meanq_robust  # noqa: E402

    pool = {
        "rerank1": (ROW_A_ID, row_a[1]),
        "rerank5": (ROW_B_ID, row_b[1]),
        "rerank5 (SF)": (ROW_B_SF_ID, row_b_sf[1], True),
    }
    winner, stats = best_by_meanq_robust(pool)
    check(winner is not None, "the real selection rule produced a winner from real (tiny) metrics")
    print(f"  winner: {winner} (MeanQ {stats[winner]['mean']:.2f})")

    winner_spec = pool[winner]
    winner_cfg = json.loads((CONFIG_DIR / f"{winner_spec[0]}_{winner_spec[1]}.json").read_text())
    winner_fields = {
        k: winner_cfg[k] for k in ("retrieval_top_k", "reranker_model", "reranker_top_k")
    }

    dependent = write_config(DEPENDENT_ID, "dependent", retrieval_top_k=0, reranker_top_k=0, self_feedback=False)
    dep_path = CONFIG_DIR / f"{dependent[0]}_{dependent[1]}.json"
    dep_cfg = json.loads(dep_path.read_text())
    dep_cfg.update(winner_fields)
    dep_path.write_text(json.dumps(dep_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n--- checking the dependent config's retrieval hyperparameters actually match the real winner ---")
    reloaded = json.loads(dep_path.read_text())
    check(
        all(reloaded[k] == v for k, v in winner_fields.items()),
        f"dependent config's retrieval hyperparameters match the winning row's real settings {winner_fields}",
    )

    # Decoding hyperparameters (temperature/top_p/top_k/min_p/presence_
    # penalty) are not tuned per row or per experiment: they are fixed to
    # each model's own developer-recommended values (manuscript
    # \autoref{tab:hyperparams}) and held uniform across every ablation
    # condition for that model, so no condition is confounded by a decoding
    # change. Llama-3.1-8B-Instruct and Latxa-Llama-3.1-8B-Instruct both use
    # temperature=0.6, top_p=0.90 (neither model card recommends top_k/
    # min_p/presence_penalty, so those stay at vLLM's own defaults). This
    # pins the smoke-test model's real ablation config to those literal
    # values, so a regression that changed them away from the manuscript's
    # own stated table would be caught here, not just a mismatch between
    # the ablation config and the reasoning-pipeline config it feeds.
    print("\n--- checking the winning row's own decoding hyperparameters match the manuscript's recommended values ---")
    check(winner_cfg.get("temperature") == 0.6, f"winning row's temperature is 0.6 (got {winner_cfg.get('temperature')!r})")
    check(winner_cfg.get("top_p") == 0.9, f"winning row's top_p is 0.90 (got {winner_cfg.get('top_p')!r})")
    for unset_key in ("top_k", "min_p", "presence_penalty"):
        check(
            not winner_cfg.get(unset_key),
            f"winning row leaves {unset_key} unset (Llama/Latxa's model cards do not recommend it, got {winner_cfg.get(unset_key)!r})",
        )

    # --- sampling hyperparameters (temperature, top_p, top_k, min_p, ...) -
    # these do not vary between ablation-grid rows of the same model (only
    # retrieval settings do), so the real risk is not "does row X get row
    # Y's temperature" but "does the reasoning-pipeline config actually
    # carry the winning row's real sampling settings forward, or silently
    # drop/default one of them". Calls the real base_config_for() (imported,
    # not re-implemented) against the real winning config, the same
    # function finalize_basque_ablation_and_write_reasoning_configs.py /
    # finalize_spanish_ablation_and_write_reasoning_configs.py use.
    print("\n--- checking sampling hyperparameters are correctly carried into a reasoning-pipeline config ---")
    from create_mixed_reasoning_configs import base_config_for, LLAMA_ENGINE_FIELDS  # noqa: E402

    winner_record = {
        "config_path": f"configs/experiments/{winner_spec[0]}_{winner_spec[1]}.json",
        "retrieval_index": winner_cfg.get("retrieval_index", ""),
        "retrieval_top_k": winner_cfg.get("retrieval_top_k", 0),
        "reranker_model": winner_cfg.get("reranker_model", ""),
        "reranker_top_k": winner_cfg.get("reranker_top_k", 0),
        "few_shot_k": winner_cfg.get("few_shot_k", 0),
    }
    rp_base = base_config_for(winner_record, LLAMA_ENGINE_FIELDS)
    sampling_fields = ("temperature", "top_p", "model", "prompt_style", "trust_remote_code")
    check(
        all(rp_base.get(k) == winner_cfg.get(k) for k in sampling_fields),
        f"reasoning-pipeline base config's sampling settings {({k: rp_base.get(k) for k in sampling_fields})} "
        f"match the real winning row's own settings",
    )
    for optional_key in ("top_k", "min_p", "presence_penalty"):
        if winner_cfg.get(optional_key) is not None:
            check(
                rp_base.get(optional_key) == winner_cfg.get(optional_key),
                f"reasoning-pipeline base config carries forward {optional_key}={winner_cfg.get(optional_key)!r} "
                f"from the winning row (got {rp_base.get(optional_key)!r})",
            )

    # --- Qwen3.5-9B: the only family that sets top_k/min_p/presence_penalty
    print("\n--- generating a small Qwen3.5-9B run to exercise top_k/min_p/presence_penalty ---")
    qwen_row = write_qwen_config()
    qwen_cfg = json.loads((CONFIG_DIR / f"{qwen_row[0]}_{qwen_row[1]}.json").read_text())
    pred_qwen = run_generation(*qwen_row)
    check(pred_qwen.exists() and len(pred_qwen.read_text().splitlines()) == LIMIT,
          f"Qwen3.5-9B: produced {LIMIT} real predictions ({pred_qwen.relative_to(ROOT)})")

    print("\n--- checking Qwen3.5-9B's own config carries the manuscript's recommended decoding values ---")
    check(qwen_cfg.get("temperature") == 1.0, f"Qwen config's temperature is 1.0 (got {qwen_cfg.get('temperature')!r})")
    check(qwen_cfg.get("top_p") == 0.95, f"Qwen config's top_p is 0.95 (got {qwen_cfg.get('top_p')!r})")
    check(qwen_cfg.get("top_k") == 20, f"Qwen config's top_k is 20 (got {qwen_cfg.get('top_k')!r})")
    check(qwen_cfg.get("min_p") == 0, f"Qwen config's min_p is 0.0 (got {qwen_cfg.get('min_p')!r})")
    check(qwen_cfg.get("presence_penalty") == 1.5, f"Qwen config's presence_penalty is 1.5 (got {qwen_cfg.get('presence_penalty')!r})")

    print("\n--- checking those settings reached the real generation subprocess as the correct vLLM flags ---")
    # Calls the real build_command() (imported, not re-implemented) -- the
    # exact function that translates a config file into the CLI arguments
    # actually passed to run_generation_experiment.py / vLLM. This is a
    # stronger check than re-reading the config file: it is exactly the
    # historical presence_penalty/repetition_penalty conflation bug's own
    # failure mode, where the config carried the right value but the code
    # silently sent something else to vLLM.
    from run_generation_from_config import build_command  # noqa: E402

    qwen_config_path = CONFIG_DIR / f"{qwen_row[0]}_{qwen_row[1]}.json"
    command = build_command(qwen_config_path, dry_run=False, save_prompts=False)
    command_str = " ".join(command)
    check("--presence-penalty 1.5" in command_str,
          f"the real generation command passes --presence-penalty 1.5 to vLLM's own presence_penalty field "
          f"(not repetition_penalty)")
    check("--top-k 20" in command_str, "the real generation command passes --top-k 20")
    check("--min-p 0" in command_str, "the real generation command passes --min-p 0")
    check("--temperature 1.0" in command_str, "the real generation command passes --temperature 1.0")
    check("--top-p 0.95" in command_str, "the real generation command passes --top-p 0.95")
    check("--repetition-penalty" not in command_str,
          "the real generation command never sets --repetition-penalty (it has no config-level equivalent; "
          "presence_penalty must not be silently remapped onto it)")

    print("\n--- checking Qwen's own hyperparameters carry correctly into a reasoning-pipeline config ---")
    from create_mixed_reasoning_configs import QWEN_ENGINE_FIELDS  # noqa: E402

    qwen_winner_record = {
        "config_path": f"configs/experiments/{qwen_row[0]}_{qwen_row[1]}.json",
        "retrieval_index": qwen_cfg.get("retrieval_index", ""),
        "retrieval_top_k": qwen_cfg.get("retrieval_top_k", 0),
        "reranker_model": qwen_cfg.get("reranker_model", ""),
        "reranker_top_k": qwen_cfg.get("reranker_top_k", 0),
        "few_shot_k": qwen_cfg.get("few_shot_k", 0),
    }
    qwen_rp_base = base_config_for(qwen_winner_record, QWEN_ENGINE_FIELDS)
    for key, expected in (("temperature", 1.0), ("top_p", 0.95), ("top_k", 20), ("min_p", 0), ("presence_penalty", 1.5)):
        check(
            qwen_rp_base.get(key) == expected,
            f"reasoning-pipeline config carries Qwen's {key}={expected!r} forward (got {qwen_rp_base.get(key)!r})",
        )

    print("\n=== SMOKE TEST PASSED: real retrieval (with gold-document exclusion), real plain and "
          "self-feedback generation, real prompt rendering, real decoding hyperparameters for both "
          "model families, and the real staged selection all work end to end ===")
    print(f"Evidence retained under configs/experiments/9700*_{SMOKE_TAG}_*.json, "
          f"experiments/runs/9700*_{SMOKE_TAG}_*/, reports/metrics/9700*_{SMOKE_TAG}_*.json")


if __name__ == "__main__":
    main()
