"""
Baseline F: Standard Hybrid RAG (Dual-Memory — Episodic RAG + Semantic PLoRA)
================================================================================

**Architecture.** The "Dual-Memory" design: every query is answered using
BOTH the episodic RAG index (Stream A — exact facts/dates, retrieved) and a
lightweight per-patient LoRA adapter (Stream B — semantic trends,
parametric, trained by ``train_trend_adapter.py``). The Executive model
(Claude) resolves the two streams via a strict **epistemic hierarchy**:

    - Trust the RAG evidence for anything factual/episodic: exact dates,
      medication names/doses, lab values, discrete events.
    - Trust the PLoRA-generated trend narrative for anything about
      trajectory/sequencing: "was this improving or worsening", "what
      pattern emerged over time", relative event ordering.
    - If the two streams conflict on a factual claim, RAG wins (it is
      grounded in retrievable source text; the adapter is not).

This is Baseline F in this project's final nomenclature (the 8B-adapter
"Hybrid Dual-Memory" configuration; see ``baseline_ef_rag/README.md`` for how
this differs from Baseline E, which has no parametric stream at all, and
from Baseline G, which additionally makes retrieval itself adaptive/agentic
rather than a single-pass lookup).

**Pipeline (per query).**
  1. **Episodic retrieval** (Stream A): hybrid BM25 + MedCPT top-5 search
     against the patient's episodic index (single pass, no iterative
     sufficiency loop — that adaptive behavior is reserved for Baseline G).
  2. **Semantic trend generation** (Stream B): the patient's Llama-3.1-8B +
     LoRA adapter is prompted with the question and asked to state, in its
     own words, the relevant trend/trajectory it has internalized.
  3. **Epistemic-hierarchy synthesis** (Claude Executive): combines both
     streams, explicit about which stream informed which part of the
     answer, citing RAG evidence with its ``[Date: ... | Section: ...]`` tag.

Requires an `ANTHROPIC_API_KEY` in the project's `.env` file, and a trained
adapter for the target patient under ``/data/baseline_f_adapters/patient_{id}/``
(see ``train_trend_adapter.py``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.retrieval_index import PatientHybridIndex  # noqa: E402

TOP_K_EACH = 5
EXECUTIVE_MODEL = "claude-sonnet-4-5-20250929"
MEMORY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYNTHESIS_SYSTEM_PROMPT = """You are a clinical documentation assistant working with TWO independent \
information sources about a patient:

1. EPISODIC EVIDENCE (retrieved chart excerpts, ground truth for exact facts/dates):
{episodic_evidence}

2. SEMANTIC TREND (this patient's own trend-summarization model's account of their trajectory, \
useful for relative sequencing/trend claims, but NOT grounded in retrievable source text):
{semantic_trend}

STRICT EPISTEMIC HIERARCHY:
- For exact dates, medication names/doses, lab values, and discrete events: trust ONLY the \
  episodic evidence. Cite it with its [Date: ... | Section: ...] tag.
- For trajectory/trend claims (improving/worsening, general pattern over time): you may use \
  the semantic trend account, clearly flagged as such.
- If the two sources conflict on a factual claim, the episodic evidence wins; note the \
  discrepancy if it's material.
"""


def _get_anthropic_client():
    from dotenv import load_dotenv
    import anthropic

    project_root = os.getenv("PROJECT_ROOT", ".")
    load_dotenv(os.path.join(project_root, ".env"), override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in .env. Add it before running Baseline F.")
    return anthropic.Anthropic(api_key=api_key)


class TrendAdapterModel:
    """Thin wrapper around a single patient's Llama-3.1-8B + LoRA adapter,
    loaded lazily so this module can be imported without GPU/transformers
    dependencies present (e.g. for unit testing the orchestration logic)."""

    def __init__(self, adapter_dir: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForCausalLM.from_pretrained(MEMORY_MODEL, torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        if hasattr(self.model, "to"):
            import torch as _torch
            self.model = self.model.to("cuda" if _torch.cuda.is_available() else "cpu")

    def generate_trend(self, question: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": "You are this patient's internalized trend model. "
                                           "Given a question, briefly state the relevant trend "
                                           "or trajectory you recall, in 1-3 sentences. Do not "
                                           "invent exact dates or numbers."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=150, do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def answer_one_query(client, index: PatientHybridIndex, trend_model, question: str) -> dict:
    t0 = time.time()

    # Stream A: single-pass hybrid retrieval (no adaptive/iterative loop -- that's Baseline G).
    hits = index.hybrid_search(question, query_embedding=None, top_k_each=TOP_K_EACH)
    episodic_evidence = "\n\n".join(h["text"] for h in hits) or "(no episodic evidence retrieved)"

    # Stream B: patient-specific semantic trend, from the LoRA adapter.
    semantic_trend = trend_model.generate_trend(question) if trend_model is not None else "(no trend adapter loaded)"

    system_prompt = SYNTHESIS_SYSTEM_PROMPT.format(
        episodic_evidence=episodic_evidence, semantic_trend=semantic_trend
    )
    resp = client.messages.create(
        model=EXECUTIVE_MODEL, max_tokens=500, temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    final_answer = resp.content[0].text.strip()

    return {
        "final_answer": final_answer,
        "cited_chunk_ids": [h["chunk_id"] for h in hits],
        "semantic_trend_used": semantic_trend,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, default="shared/hybrid_indices")
    ap.add_argument("--adapter_dir", type=str, default="shared/baseline_f_adapters")
    ap.add_argument("--spreadsheet", type=str, default="shared/spreadsheet.xlsx")
    ap.add_argument("--patient_ids", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="baseline_f_results")
    ap.add_argument("--skip_adapter", action="store_true",
                     help="Run RAG-only (no trend adapter) -- useful for a quick smoke test "
                          "without a GPU / trained adapter available.")
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
            print(f"[SKIP] patient {pid}: episodic index not found ({e}).")
            continue

        trend_model = None
        adapter_path = os.path.join(args.adapter_dir, f"patient_{pid}")
        if not args.skip_adapter and os.path.isdir(adapter_path):
            trend_model = TrendAdapterModel(adapter_path)
        elif not args.skip_adapter:
            print(f"[WARN] patient {pid}: no adapter found at {adapter_path}, "
                  f"proceeding with RAG-only synthesis for this patient.")

        out_path = os.path.join(args.out_dir, f"phase3_results_{pid}.jsonl")
        with open(out_path, "w") as fout:
            for qi, q in enumerate(queries):
                result = answer_one_query(client, index, trend_model, q["question"])
                record = {
                    "person_id": pid, "question": q["question"],
                    "ground_truth_answer": q.get("answer"),
                    "reasoning_type": q.get("reasoning_type"), "difficulty": q.get("difficulty"),
                    "is_temporal": bool(q.get("is_temporal")), **result,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[patient {pid}] {qi + 1}/{len(queries)} done ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
