"""
Baseline E: Pure Episodic RAG (No Parametric Memory)
======================================================

**Architecture.** Standard Retrieval-Augmented Generation with NO PLoRA /
parametric memory component at all. The Executive model (Claude) answers
strictly from chunks retrieved out of the hybrid BM25 + MedCPT episodic
index built by ``build_episodic_index.py``. This isolates the "Memory
Model Value-Add" question: how much does adding a parametric memory adapter
(Baseline F, G) actually buy you over retrieval alone?

**Pipeline (per query).**
  1. **Decompose.** The Executive (Claude) turns the benchmark question into
     a short temporal search plan (an initial retrieval query string).
  2. **Initial retrieval.** Hybrid BM25 + MedCPT search (top-5 each) against
     the patient's episodic index.
  3. **Sufficiency gate.** The Executive judges whether the retrieved
     evidence pool is SUFFICIENT to answer the question.
  4. **Iterative retrieval.** If INSUFFICIENT, the Executive reformulates
     the search query and retrieves more evidence. This repeats until
     SUFFICIENT or a hard cap of 15 total retrieved chunks is hit (whichever
     comes first) -- the cap exists purely as a safety valve against
     runaway loops, not as a target to reach.
  5. **Synthesis.** The Executive writes the final answer, citing evidence
     using the injected ``[Date: ... | Section: ...]`` temporal metadata,
     using ONLY the retrieved episodic evidence (explicitly instructed NOT
     to use any outside/parametric knowledge).

This is the "Retrieval Only" control against which Baseline F (RAG + PLoRA)
and Baseline G (Adaptive Retrieval + Policy-Distilled PLoRA) are compared.

Requires an `ANTHROPIC_API_KEY` in the project's `.env` file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.retrieval_index import PatientHybridIndex  # noqa: E402

MAX_TOTAL_CHUNKS = 15
TOP_K_EACH = 5
EXECUTIVE_MODEL = "claude-sonnet-4-5-20250929"


def _get_anthropic_client():
    """Load ANTHROPIC_API_KEY from the project's .env (never from the raw
    process environment -- see repository README for the API-key policy)."""
    from dotenv import load_dotenv
    import anthropic

    project_root = os.getenv("PROJECT_ROOT", ".")
    load_dotenv(os.path.join(project_root, ".env"), override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found in .env. Add it via your environment "
            "variables panel / .env file before running Baseline E."
        )
    return anthropic.Anthropic(api_key=api_key)


def decompose_query(client, question: str) -> str:
    """Ask the Executive for an initial lexical/semantic search string."""
    resp = client.messages.create(
        model=EXECUTIVE_MODEL,
        max_tokens=200,
        temperature=0.0,
        system="You turn a clinical question into a short, specific search query "
               "(keywords + concepts) suitable for a hybrid lexical+semantic search "
               "engine over a patient's chart. Respond with ONLY the search query text, "
               "no explanation.",
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text.strip()


def sufficiency_check(client, question: str, evidence_text: str) -> dict:
    """Ask the Executive whether the current evidence pool suffices."""
    resp = client.messages.create(
        model=EXECUTIVE_MODEL,
        max_tokens=300,
        temperature=0.0,
        system="You are a strict evidence auditor. Given a clinical question and the "
               "evidence retrieved so far, decide if it is SUFFICIENT to fully answer "
               "the question. Respond with ONLY a JSON object: "
               '{"verdict": "SUFFICIENT" or "INSUFFICIENT", "next_query": "<a refined '
               'search query to run next, or empty string if SUFFICIENT>"}',
        messages=[{"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_text}"}],
    )
    text = resp.content[0].text.strip()
    try:
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])
    except Exception:
        return {"verdict": "SUFFICIENT", "next_query": ""}  # fail safe: stop looping


def synthesize_answer(client, question: str, evidence_text: str) -> str:
    resp = client.messages.create(
        model=EXECUTIVE_MODEL,
        max_tokens=500,
        temperature=0.0,
        system="Answer the clinical question STRICTLY using the evidence chunks below. "
               "Do NOT use any outside medical knowledge. Cite every factual claim with "
               "the exact [Date: ... | Section: ...] tag from the evidence it came from. "
               "If the evidence is incomplete, say so explicitly.",
        messages=[{"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_text}"}],
    )
    return resp.content[0].text.strip()


def answer_one_query(client, index: PatientHybridIndex, question: str) -> dict:
    t0 = time.time()
    search_query = decompose_query(client, question)
    evidence_pool: dict = {}
    hops = []

    for hop_idx in range(1, 4):  # a handful of reformulation rounds, bounded by MAX_TOTAL_CHUNKS below
        hits = index.hybrid_search(search_query, query_embedding=None, top_k_each=TOP_K_EACH)
        new_ids = []
        for h in hits:
            cid = h["chunk_id"]
            if cid not in evidence_pool and len(evidence_pool) < MAX_TOTAL_CHUNKS:
                evidence_pool[cid] = h
                new_ids.append(cid)

        evidence_text = "\n\n".join(f"{c['text']}" for c in evidence_pool.values())
        suff = sufficiency_check(client, question, evidence_text)
        hops.append({"hop": hop_idx, "search_query": search_query,
                     "new_chunk_ids": new_ids, "pool_size": len(evidence_pool),
                     "verdict": suff.get("verdict")})

        if suff.get("verdict") == "SUFFICIENT" or len(evidence_pool) >= MAX_TOTAL_CHUNKS or not suff.get("next_query"):
            break
        search_query = suff["next_query"]

    evidence_text = "\n\n".join(f"{c['text']}" for c in evidence_pool.values())
    final_answer = synthesize_answer(client, question, evidence_text)

    return {
        "final_answer": final_answer,
        "cited_chunk_ids": list(evidence_pool.keys()),
        "n_hops": len(hops),
        "hops": hops,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, default="shared/hybrid_indices",
                     help="Directory containing per-patient episodic indices (see build_episodic_index.py).")
    ap.add_argument("--spreadsheet", type=str, default="shared/spreadsheet.xlsx")
    ap.add_argument("--patient_ids", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="baseline_e_results")
    args = ap.parse_args()

    from common.data_utils import CANONICAL_PATIENT_IDS, load_benchmark_queries

    client = _get_anthropic_client()
    patient_ids = [int(x) for x in args.patient_ids.split(",")] if args.patient_ids else CANONICAL_PATIENT_IDS

    os.makedirs(args.out_dir, exist_ok=True)
    for pid in patient_ids:
        queries = load_benchmark_queries(args.spreadsheet, person_id=pid)
        if not queries:
            continue
        try:
            index = PatientHybridIndex(pid, base_dir=args.index_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] patient {pid}: index not found ({e}). Run build_episodic_index.py first.")
            continue

        out_path = os.path.join(args.out_dir, f"phase3_results_{pid}.jsonl")
        with open(out_path, "w") as fout:
            for qi, q in enumerate(queries):
                result = answer_one_query(client, index, q["question"])
                record = {
                    "person_id": pid,
                    "question": q["question"],
                    "ground_truth_answer": q.get("answer"),
                    "reasoning_type": q.get("reasoning_type"),
                    "difficulty": q.get("difficulty"),
                    "is_temporal": bool(q.get("is_temporal")),
                    **result,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[patient {pid}] {qi + 1}/{len(queries)} done ({result['elapsed_sec']}s, "
                      f"{result['n_hops']} hop(s))")


if __name__ == "__main__":
    main()
