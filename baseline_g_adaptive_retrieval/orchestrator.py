"""
Baseline G — Phase 3: Adaptive Evidence Retrieval + Executive Synthesis
==========================================================================

The full inference-time pipeline for Baseline G, tying together:
  - **Executive** (``claude-sonnet-4-5-20250929`` via the Anthropic API):
    decomposes the query, runs the sufficiency gate, and synthesizes the
    final grounded answer.
  - **SLM Router** (``meta-llama/Llama-3.2-1B-Instruct`` + that patient's
    Phase-2 retrieval-policy LoRA adapter, served on Modal): at each hop,
    generates a *retrieval cue* (a search string tuned to this patient's
    chart vocabulary) rather than answering the question directly.
  - **Episodic Search** (``common.retrieval_index.PatientHybridIndex``):
    BM25 top-5 + MedCPT top-5 hybrid search against the patient's episodic
    index, merged/deduped into a running evidence pool.

**Per-query loop:**
  1. **Decompose & Specify** (Claude) → an ordered list of evidence
     requirements.
  2. For up to ``MAX_HOPS`` hops:
       a. **SLM-Guided Retrieval** (1B + adapter) → a retrieval cue string.
       b. **Episodic Search** (hybrid BM25+MedCPT) → new evidence merged into
          the pool.
       c. **Sufficiency check** (Claude) → SUFFICIENT (stop) or INSUFFICIENT
          + a refined next requirement (continue).
  3. **Executive Synthesis** (Claude) → final answer + cited chunk IDs.

Output: ``{out_dir}/phase3_results_{pid}.jsonl`` (written incrementally, one
line per completed query, so partial progress survives any mid-run
failure).

Note on hop caps: this project's experiments used both a strict safety cap
of ``MAX_HOPS = 3`` and an extended "bounded sufficiency" variant with
``MAX_HOPS = 15``. Both are supported here via the ``--max_hops`` flag; see
the top-level README for the (important, non-obvious) empirical finding that
raising the cap from 3 to 15 barely changed Kendall's Tau, indicating the
retrieval-depth bottleneck is not really about *how much* evidence is
gathered.

Requires an ``ANTHROPIC_API_KEY`` in the project's ``.env``, and requires the
Modal-hosted SLM router service to be deployed first (``modal deploy
slm_router_service.py`` — a thin Modal ``@app.cls`` wrapping the Phase-2
adapters; not included verbatim here to keep this repo's core logic
readable, but structurally identical to ``train_adapters.py``'s model
loading code plus a ``generate_cue`` method).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.retrieval_index import PatientHybridIndex  # noqa: E402

TOP_K_EACH = 5
EXECUTIVE_MODEL = "claude-sonnet-4-5-20250929"

CANONICAL_PATIENT_IDS = [
    8855233, 8858035, 8860166, 8860822, 8911710,
    8922374, 8945136, 8955898, 8979075, 9070451,
    9077267, 9088069, 9102132, 9126037, 9354504,
    9989372, 11235719, 11617093, 12015052, 12355020,
]


def _get_anthropic_client():
    from dotenv import load_dotenv
    import anthropic

    project_root = os.getenv("PROJECT_ROOT", ".")
    load_dotenv(os.path.join(project_root, ".env"), override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in .env. Add it before running Baseline G.")
    return anthropic.Anthropic(api_key=api_key)


def decompose(client, question: str) -> dict:
    resp = client.messages.create(
        model=EXECUTIVE_MODEL, max_tokens=400, temperature=0.0,
        system="Break this clinical question into an ORDERED JSON list of specific evidence "
               "requirements needed to answer it fully. Respond with ONLY a JSON object: "
               '{"requirements": ["<requirement 1>", "<requirement 2>", ...]}',
        messages=[{"role": "user", "content": question}],
    )
    raw = resp.content[0].text.strip()
    try:
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end + 1])
        requirements = parsed.get("requirements") or [question]
    except Exception:
        requirements = [question]
    return {"requirements": requirements, "raw": raw}


def sufficiency_check(client, question: str, requirements: list, evidence_text: str) -> dict:
    resp = client.messages.create(
        model=EXECUTIVE_MODEL, max_tokens=300, temperature=0.0,
        system="You are a strict clinical evidence auditor. Given the question, the evidence "
               "requirements, and the evidence gathered so far, decide SUFFICIENT or "
               "INSUFFICIENT. Respond with ONLY a JSON object: "
               '{"verdict": "SUFFICIENT" or "INSUFFICIENT", "reasoning": "<one sentence>", '
               '"next_requirement": "<the next specific thing to search for, or empty string>"}',
        messages=[{"role": "user", "content": f"QUESTION: {question}\nREQUIREMENTS: {requirements}\n\n"
                                                f"EVIDENCE SO FAR:\n{evidence_text}"}],
    )
    raw = resp.content[0].text.strip()
    try:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end + 1])
    except Exception:
        return {"verdict": "SUFFICIENT", "reasoning": "parse error, stopping early", "next_requirement": ""}


def synthesize(client, question: str, evidence_text: str) -> dict:
    resp = client.messages.create(
        model=EXECUTIVE_MODEL, max_tokens=500, temperature=0.0,
        system="Answer the clinical question STRICTLY using the evidence chunks below. Cite "
               "every claim with the [chunk_id: ...] tag it came from. If evidence is "
               "incomplete, say so.",
        messages=[{"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_text}"}],
    )
    text = resp.content[0].text.strip()
    import re
    cited = list(dict.fromkeys(re.findall(r"\[chunk_id:\s*([^\]]+)\]", text)))
    return {"answer": text, "cited_chunk_ids": cited}


def process_one_query(claude_client, slm_svc, idx: PatientHybridIndex, person_id: int, q: dict, max_hops: int):
    t0 = time.time()
    question = q["question"]

    decomposed = decompose(claude_client, question)
    requirements = decomposed["requirements"]

    evidence_pool: dict = {}
    hops = []
    current_requirement = requirements[0]

    for hop_idx in range(1, max_hops + 1):
        # SLM router: generate a retrieval cue tuned to this patient's chart
        # vocabulary. `slm_svc` is a deployed Modal service exposing
        # `generate_cue.remote(person_id, requirement) -> {"cue", "search_text",
        # "query_embedding", "parse_error"}`, built on top of the Phase-2 adapter.
        slm_result = slm_svc.generate_cue.remote(person_id, current_requirement)
        hits = idx.hybrid_search(slm_result["search_text"], slm_result["query_embedding"], top_k_each=TOP_K_EACH)

        new_chunk_ids = []
        for h in hits:
            if h["chunk_id"] not in evidence_pool:
                evidence_pool[h["chunk_id"]] = h
                new_chunk_ids.append(h["chunk_id"])

        evidence_text = "\n\n".join(f"[chunk_id: {cid}] {c['text']}" for cid, c in evidence_pool.items())
        suff = sufficiency_check(claude_client, question, requirements, evidence_text)

        hops.append({
            "hop": hop_idx, "requirement": current_requirement, "slm_cue": slm_result["cue"],
            "new_chunk_ids_retrieved": new_chunk_ids, "total_evidence_pool_size": len(evidence_pool),
            "sufficiency_verdict": suff["verdict"], "sufficiency_reasoning": suff.get("reasoning"),
        })

        if suff["verdict"] == "SUFFICIENT" or hop_idx == max_hops or not suff.get("next_requirement"):
            break
        current_requirement = suff["next_requirement"]

    evidence_text = "\n\n".join(f"[chunk_id: {cid}] {c['text']}" for cid, c in evidence_pool.items())
    synth = synthesize(claude_client, question, evidence_text)

    return {
        "person_id": person_id, "question": question, "ground_truth_answer": q.get("answer"),
        "reasoning_type": q.get("reasoning_type"), "difficulty": q.get("difficulty"),
        "is_temporal": bool(q.get("is_temporal")),
        "final_answer": synth["answer"], "cited_chunk_ids": synth["cited_chunk_ids"],
        "adaptive_evidence_notes": {
            "decompose_requirements": requirements, "hops": hops, "n_hops_used": len(hops),
            "final_evidence_pool_chunk_ids": list(evidence_pool.keys()),
            "final_evidence_pool_size": len(evidence_pool),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, default="shared/hybrid_indices")
    ap.add_argument("--spreadsheet", type=str, default="shared/spreadsheet.xlsx")
    ap.add_argument("--patient_ids", type=str, default=None)
    ap.add_argument("--limit_per_patient", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="baseline_g_results")
    ap.add_argument("--max_hops", type=int, default=3, help="3 = original safety cap; 15 = extended bounded-sufficiency variant.")
    ap.add_argument("--workers", type=int, default=1, help="Concurrent queries via ThreadPoolExecutor (network-bound).")
    ap.add_argument("--slm_app_name", type=str, default="baseline-g-slm-router")
    args = ap.parse_args()

    import pandas as pd
    import modal

    all_q = pd.read_excel(args.spreadsheet)
    claude_client = _get_anthropic_client()
    SLMRouter = modal.Cls.from_name(args.slm_app_name, "SLMGuidedRetrieval")
    slm_svc = SLMRouter()

    patient_ids = [int(x) for x in args.patient_ids.split(",")] if args.patient_ids else \
        sorted(pid for pid in all_q["person_id"].unique().tolist() if pid in CANONICAL_PATIENT_IDS)

    os.makedirs(args.out_dir, exist_ok=True)
    idx_map, flat_tasks = {}, []
    for pid in patient_ids:
        sub = all_q[all_q["person_id"] == pid]
        queries = sub.to_dict("records")
        if args.limit_per_patient:
            queries = queries[: args.limit_per_patient]
        if not queries:
            continue
        try:
            idx_map[pid] = PatientHybridIndex(pid, base_dir=args.index_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] patient {pid}: index not found ({e})")
            continue
        for qi, q in enumerate(queries):
            flat_tasks.append((pid, qi, q, len(queries)))

    out_handles, out_locks = {}, {}
    for pid in idx_map:
        out_handles[pid] = open(os.path.join(args.out_dir, f"phase3_results_{pid}.jsonl"), "w")
        out_locks[pid] = threading.Lock()

    print(f"Processing {len(flat_tasks)} queries across {len(idx_map)} patients, {args.workers} worker(s), max_hops={args.max_hops}")
    progress_lock = threading.Lock()
    global_idx = 0

    def _run(task):
        pid, qi, q, n_for_pid = task
        try:
            return pid, qi, process_one_query(claude_client, slm_svc, idx_map[pid], pid, q, args.max_hops)
        except Exception as e:
            traceback.print_exc()
            return pid, qi, {"person_id": pid, "question": q["question"], "error": str(e), "final_answer": None}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_run, t) for t in flat_tasks]
        for fut in as_completed(futures):
            pid, qi, record = fut.result()
            with out_locks[pid]:
                out_handles[pid].write(json.dumps(record, ensure_ascii=False) + "\n")
                out_handles[pid].flush()
            with progress_lock:
                global_idx += 1
                print(f"[{global_idx}/{len(flat_tasks)}] patient {pid} query {qi+1}: "
                      f"{record.get('question', '')[:70]}")

    for f in out_handles.values():
        f.close()
    print("ALL DONE.")


if __name__ == "__main__":
    main()
