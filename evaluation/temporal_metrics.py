"""
evaluation/temporal_metrics.py
=================================
Computes the two headline temporal-reasoning metrics used to compare every
baseline (D through H) in this project:

  1. **Temporal MAE (days)** — Mean Absolute Error between the dates/
     durations the model's answer implies and the dates/durations in the
     ground truth, in days. Lower is better. Answers absolute-date accuracy:
     "did the model get WHEN right?"

  2. **Kendall's Tau** — rank correlation between the chronological order of
     events in the ground truth vs. the chronological order implied by the
     model's answer. Higher (closer to 1.0) is better. Answers relative-
     ordering accuracy: "did the model get the SEQUENCE right, even if
     individual dates are imprecise?"

These two metrics deliberately measure DIFFERENT failure modes, and this
project's results show they can diverge sharply (e.g. a small parametric
model can nail relative ordering with terrible absolute dates, while a
RAG-heavy system can nail dates but scramble the sequence) — see the
top-level README's results discussion.

**Pipeline.**
  1. Run ``llm_judge.py`` first to get, for every query, a symmetric
     extraction of structured events (``{event, date, duration_days}``) from
     BOTH the reference and candidate answers.
  2. **Event matching**: greedy token-overlap (Token-F1 >= 0.15, via
     ``text_metrics.precision_recall_f1`` on the ``event`` description) pairs
     each reference event to its best unmatched candidate event.
  3. **Temporal MAE**: pool ``|Δdate|`` (days) and ``|Δduration|`` (days)
     across all matched pairs project-wide, take the mean.
  4. **Kendall's Tau**: for each query with >= 2 matched, DATED event pairs,
     compute tau-a (concordant - discordant) / total_pairs on the ordinal
     date ranks; average across all queries with a computable tau. A custom
     O(n²) implementation is used (no scipy dependency) since per-query n is
     always small (a handful of events).

Usage:
    python evaluation/temporal_metrics.py \\
        --judge_results evaluation_results/llm_judge_raw.jsonl \\
        --local_metrics evaluation_results/local_metrics_summary.json \\
        --out_path evaluation_results/phase4_final_metrics.json
"""
import argparse
import json
import statistics as stats
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.text_metrics import precision_recall_f1  # noqa: E402


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_date(s):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def event_similarity(e1: dict, e2: dict) -> float:
    _, _, f1 = precision_recall_f1(e1.get("event", "") or "", e2.get("event", "") or "")
    return f1


def match_events(ref_events, cand_events, sim_threshold: float = 0.15):
    """Greedy matching: for each reference event (in order), pick the best
    unmatched candidate event by token-F1 text similarity of the event
    description, if above ``sim_threshold``."""
    pairs, used_cand = [], set()
    for re_ev in ref_events:
        best_j, best_sim = None, 0.0
        for ci, ca_ev in enumerate(cand_events):
            if ci in used_cand:
                continue
            sim = event_similarity(re_ev, ca_ev)
            if sim > best_sim:
                best_sim, best_j = sim, ci
        if best_j is not None and best_sim >= sim_threshold:
            used_cand.add(best_j)
            pairs.append((re_ev, cand_events[best_j], best_sim))
    return pairs


def kendall_tau(x, y):
    """Simple O(n^2) Kendall's tau-a implementation (no scipy dependency
    needed; per-query n is always small)."""
    n = len(x)
    if n < 2:
        return None
    concordant = discordant = ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            prod = (x[i] - x[j]) * (y[i] - y[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                ties += 1
    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs if total_pairs else None


def aggregate(judge_records: list, local_agg: dict = None) -> dict:
    """Compute Temporal MAE, Kendall's Tau, and (if local_agg is given) merge
    in the deterministic text metrics + judge accuracy for the final
    10-metric report used throughout this project."""
    judge_scores, judge_accurate, parse_errors = [], [], 0
    for r in judge_records:
        jr = r["judge_result"]
        if jr.get("_parse_error") or jr.get("score") is None:
            parse_errors += 1
            continue
        judge_scores.append(jr["score"])
        judge_accurate.append(1 if jr.get("candidate_accurate") else 0)

    avg_judge_score = sum(judge_scores) / len(judge_scores) if judge_scores else None
    judge_accuracy_pct = 100.0 * sum(judge_accurate) / len(judge_accurate) if judge_accurate else None

    date_diffs, duration_diffs, per_query_taus = [], [], []
    n_queries_with_matches = n_queries_with_tau = 0
    per_query_temporal = []

    for r in judge_records:
        jr = r["judge_result"]
        if jr.get("_parse_error"):
            continue
        ref_events = jr.get("reference_events", []) or []
        cand_events = jr.get("candidate_events", []) or []
        pairs = match_events(ref_events, cand_events)

        q_date_diffs, q_duration_diffs = [], []
        matched_dates_ref, matched_dates_cand = [], []

        for ref_ev, cand_ev, _sim in pairs:
            rd, cd = parse_date(ref_ev.get("date")), parse_date(cand_ev.get("date"))
            if rd is not None and cd is not None:
                diff_days = abs((rd - cd).days)
                q_date_diffs.append(diff_days)
                date_diffs.append(diff_days)
                matched_dates_ref.append(rd)
                matched_dates_cand.append(cd)

            r_dur, c_dur = ref_ev.get("duration_days"), cand_ev.get("duration_days")
            if isinstance(r_dur, (int, float)) and isinstance(c_dur, (int, float)):
                d = abs(r_dur - c_dur)
                q_duration_diffs.append(d)
                duration_diffs.append(d)

        tau = None
        if len(matched_dates_ref) >= 2:
            ref_ord = [d.toordinal() for d in matched_dates_ref]
            cand_ord = [d.toordinal() for d in matched_dates_cand]
            tau = kendall_tau(ref_ord, cand_ord)
            if tau is not None:
                per_query_taus.append(tau)
                n_queries_with_tau += 1

        if pairs:
            n_queries_with_matches += 1

        per_query_temporal.append({
            "person_id": r.get("person_id"), "question": r.get("question"),
            "n_ref_events": len(ref_events), "n_cand_events": len(cand_events),
            "n_matched_pairs": len(pairs), "n_dated_matched_pairs": len(matched_dates_ref),
            "date_diffs_days": q_date_diffs, "duration_diffs_days": q_duration_diffs, "kendall_tau": tau,
        })

    combined_mae_values = date_diffs + duration_diffs
    date_mae = sum(date_diffs) / len(date_diffs) if date_diffs else None
    duration_mae = sum(duration_diffs) / len(duration_diffs) if duration_diffs else None
    combined_mae = sum(combined_mae_values) / len(combined_mae_values) if combined_mae_values else None
    kendall_tau_mean = sum(per_query_taus) / len(per_query_taus) if per_query_taus else None
    kendall_tau_median = stats.median(per_query_taus) if per_query_taus else None

    final_metrics = {
        "n_queries": len(judge_records),
        "avg_judge_score_0_3": avg_judge_score,
        "llm_judge_accuracy_pct": judge_accuracy_pct,
        "temporal_mae_days": combined_mae,
        "kendall_tau_mean": kendall_tau_mean,
    }
    if local_agg:
        final_metrics.update({k: local_agg.get(k) for k in
                               ("exact_match", "precision", "recall", "token_f1", "rouge_l", "bertscore_f1")})

    diagnostics = {
        "judge_parse_errors": parse_errors,
        "date_mae_days": date_mae, "duration_mae_days": duration_mae, "combined_mae_days": combined_mae,
        "n_date_diff_pairs": len(date_diffs), "n_duration_diff_pairs": len(duration_diffs),
        "kendall_tau_median": kendall_tau_median,
        "n_queries_with_matched_events": n_queries_with_matches,
        "n_queries_with_computable_tau": n_queries_with_tau,
    }

    return {"final_metrics": final_metrics, "diagnostics": diagnostics, "per_query_temporal": per_query_temporal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_results", type=str, required=True,
                     help="Path to llm_judge.py output JSONL (llm_judge_raw.jsonl).")
    ap.add_argument("--local_metrics", type=str, default=None,
                     help="Optional path to bertscore_eval.py's local_metrics_summary.json "
                          "(to merge EM/Precision/Recall/F1/ROUGE-L/BERTScore into the final report).")
    ap.add_argument("--out_path", type=str, default="evaluation_results/phase4_final_metrics.json")
    args = ap.parse_args()

    judge_records = load_jsonl(args.judge_results)
    local_agg = None
    if args.local_metrics:
        with open(args.local_metrics) as f:
            local_agg = json.load(f)

    result = aggregate(judge_records, local_agg)
    print("=== FINAL METRICS ===")
    print(json.dumps(result["final_metrics"], indent=2))
    print("=== DIAGNOSTICS ===")
    print(json.dumps(result["diagnostics"], indent=2))

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
