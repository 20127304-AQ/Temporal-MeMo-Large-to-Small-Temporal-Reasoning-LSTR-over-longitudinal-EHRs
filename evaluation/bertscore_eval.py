"""
evaluation/bertscore_eval.py
===============================
Modal GPU job: computes BERTScore-F1 (semantic similarity, ``roberta-large``
backbone via the ``bert-score`` package) for any baseline's Phase-3 output
files. Complements the purely lexical metrics in ``text_metrics.py`` with a
semantic-similarity signal that tolerates paraphrasing.

Usage:
    modal run evaluation/bertscore_eval.py --results_glob "shared/baseline_g_artifacts/phase3_results_*.jsonl"

Input:  one or more JSONL files matching ``results_glob``, each record with
        at least ``final_answer`` and ``ground_truth_answer``.
Output: ``<out_dir>/per_query_text_metrics.jsonl`` (adds ``bertscore_f1`` to
        every record, alongside the deterministic text_metrics.py fields)
        and ``<out_dir>/local_metrics_summary.json`` (aggregate means).
"""
import glob
import json
import os

import modal

app = modal.App("temporal-memo-bertscore-eval")

ml_general_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch torchvision numpy transformers datasets tiktoken "
        "tqdm matplotlib pandas bert-score"
    )
    .add_local_python_source("evaluation")
)


@app.function(image=ml_general_image, gpu="a10g", timeout=1800)
def compute_metrics(records: list) -> dict:
    """Compute the 5 deterministic text_metrics.py metrics + BERTScore-F1
    for a list of already-loaded result records. Returns
    ``(per_query_list, aggregate_dict)``."""
    from evaluation.text_metrics import compute_all_text_metrics

    preds = [r.get("final_answer", "") or "" for r in records]
    gts = [r.get("ground_truth_answer", "") or "" for r in records]

    per_query = []
    for r, pred, gt in zip(records, preds, gts):
        m = compute_all_text_metrics(pred, gt)
        per_query.append({
            "person_id": r.get("person_id"), "question": r.get("question"),
            "ground_truth_answer": gt, "final_answer": pred,
            "reasoning_type": r.get("reasoning_type"), "difficulty": r.get("difficulty"),
            **m,
        })

    print("[INFO] Computing BERTScore-F1 with bert_score (roberta-large)...")
    from bert_score import score as bertscore_fn
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, F1 = bertscore_fn(preds, gts, lang="en", device=device, verbose=True)
    for i, f1v in enumerate(F1.tolist()):
        per_query[i]["bertscore_f1"] = f1v

    agg = {key: sum(q[key] for q in per_query) / len(per_query)
           for key in ("exact_match", "precision", "recall", "token_f1", "rouge_l", "bertscore_f1")}
    agg["n_queries"] = len(per_query)

    print("[RESULT] Aggregate local (non-judge) metrics:")
    print(json.dumps(agg, indent=2))
    return {"per_query": per_query, "aggregate": agg}


@app.local_entrypoint()
def main(results_glob: str = "shared/baseline_g_artifacts/phase3_results_*.jsonl",
          out_dir: str = "evaluation_results"):
    records = []
    for fp in sorted(glob.glob(results_glob)):
        with open(fp) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    print(f"[INFO] Loaded {len(records)} records from {results_glob}")
    assert records, "No result records found -- check --results_glob."

    result = compute_metrics.remote(records)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_query_text_metrics.jsonl"), "w") as f:
        for q in result["per_query"]:
            f.write(json.dumps(q) + "\n")
    with open(os.path.join(out_dir, "local_metrics_summary.json"), "w") as f:
        json.dump(result["aggregate"], f, indent=2)
    print(f"Saved results to {out_dir}/")
