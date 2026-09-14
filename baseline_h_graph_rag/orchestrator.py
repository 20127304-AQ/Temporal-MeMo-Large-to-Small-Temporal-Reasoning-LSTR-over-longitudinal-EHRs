"""
Baseline H — Local orchestrator for the Adaptive SLM ReAct pipeline.

Drives ``SLMReActEngine`` (Modal, ``meta-llama/Llama-3.2-3B-Instruct``) over
all benchmark queries for the canonical pilot cohort, one Modal
``.remote()`` call per patient (keeps the GPU-hosted model loaded across
that patient's entire query set to avoid repeated cold starts), and writes
incrementally to ``{out_dir}/phase3_results_{pid}.jsonl`` so partial
progress survives any mid-run failure.
"""
import argparse
import json
import os
import time

CANONICAL_PATIENT_IDS = [
    8855233, 8858035, 8860166, 8860822, 8911710,
    8922374, 8945136, 8955898, 8979075, 9070451,
    9077267, 9088069, 9102132, 9126037, 9354504,
    9989372, 11235719, 11617093, 12015052, 12355020,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spreadsheet", type=str, default="shared/spreadsheet.xlsx")
    ap.add_argument("--patient_ids", type=str, default=None)
    ap.add_argument("--limit_per_patient", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="shared/workflow_H_artifacts")
    args = ap.parse_args()

    import pandas as pd
    import modal

    all_q = pd.read_excel(args.spreadsheet)
    SLMReActEngine = modal.Cls.from_name("baseline-h-slm-react", "SLMReActEngine")
    svc = SLMReActEngine()

    patient_ids = [int(x) for x in args.patient_ids.split(",")] if args.patient_ids else \
        sorted(pid for pid in all_q["person_id"].unique().tolist() if pid in CANONICAL_PATIENT_IDS)

    os.makedirs(args.out_dir, exist_ok=True)
    worklist = []
    for pid in patient_ids:
        queries = all_q[all_q["person_id"] == pid].to_dict("records")
        if args.limit_per_patient:
            queries = queries[: args.limit_per_patient]
        if queries:
            worklist.append((pid, queries))
    total_queries = sum(len(qs) for _, qs in worklist)
    print(f"Total queries to process: {total_queries} across {len(worklist)} patients")

    overall_t0 = time.time()
    summary, n_errors = {}, 0

    for pid, queries in worklist:
        print(f"\n=== Patient {pid}: {len(queries)} queries ===")
        t0 = time.time()
        try:
            results = svc.run_patient_queries.remote(pid, queries)
        except Exception as e:
            print(f"  !! ERROR on patient {pid}: {e}")
            n_errors += len(queries)
            summary[str(pid)] = {"n_queries": len(queries), "n_errors": len(queries), "error": str(e)}
            continue

        out_path = os.path.join(args.out_dir, f"phase3_results_{pid}.jsonl")
        with open(out_path, "w") as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        elapsed = time.time() - t0

        valid = [r for r in results if r.get("n_iterations") is not None]
        n_err_pid = len(results) - len(valid)
        n_errors += n_err_pid
        summary[str(pid)] = {
            "n_queries": len(results), "n_errors": n_err_pid, "elapsed_sec": round(elapsed, 1),
            "avg_iterations": round(sum(r["n_iterations"] for r in valid) / len(valid), 2) if valid else 0,
            "avg_evidence_chunks": round(sum(r["final_evidence_pool_size"] for r in valid) / len(valid), 2) if valid else 0,
            "halt_reason_counts": {k: sum(1 for r in valid if r.get("halt_reason") == k)
                                    for k in ("sufficient", "duplicate_strategy", "max_hops")},
        }
        print(f"Patient {pid} done in {elapsed:.1f}s -> {out_path} ({len(valid)}/{len(results)} ok)")

    total_elapsed = time.time() - overall_t0
    summary["_overall"] = {"total_elapsed_sec": round(total_elapsed, 1), "n_patients": len(worklist),
                            "n_queries": total_queries, "n_errors": n_errors}
    with open(os.path.join(args.out_dir, "phase3_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nALL DONE in {total_elapsed:.1f}s. Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
