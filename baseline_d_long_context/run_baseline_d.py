"""
Baseline D: Small Model Long-Context Prompting
================================================

**Architecture.** The simplest, weakest baseline in the project, and the
control that everything else (PLoRA memory, RAG, adaptive retrieval,
Graph-RAG) is trying to beat: feed a small model the patient's ENTIRE raw
visit history, concatenated as plain text, directly in the prompt -- no
chunking, no retrieval, no memory adapters, no summarization. Whatever the
model can attend to within its native context window is all it gets.

**Purpose.** Establishes the "performance floor" for context-window overload
and the well-documented "Lost in the Middle" phenomenon (Liu et al., 2023):
small models degrade sharply at recalling/ordering facts buried in the
middle of a very long context, even when that context technically fits
within the advertised context length.

**Models evaluated.**
  - ``meta-llama/Llama-3.2-1B-Instruct``  (128K advertised context)
  - ``meta-llama/Llama-3.1-8B-Instruct``  (128K advertised context)

Both are served locally via vLLM on a single Modal GPU for efficient batched
generation across all 289 benchmark queries.

**Pipeline.**
  1. Load the 20-patient canonical cohort (``common.data_utils``).
  2. For each of the 289 benchmark queries, build a prompt containing the
     FULL raw visit history of that query's patient (truncated only if it
     exceeds the model's max context length -- and we log every truncation,
     since silently working around context overload would defeat the point
     of this baseline).
  3. Generate an answer with each model (temperature=0.0, deterministic).
  4. Write results to ``baseline_d_results_{model_tag}.jsonl`` in the exact
     schema consumed by ``evaluation/`` (``person_id, question,
     ground_truth_answer, final_answer, reasoning_type, difficulty``), so the
     same Phase-4 evaluation pipeline used for every other baseline in this
     repo (Temporal MAE, Kendall's Tau, text-overlap metrics, LLM judge) can
     be applied without modification.

Run with:
    modal run baseline_d_long_context/run_baseline_d.py --model 1b
    modal run baseline_d_long_context/run_baseline_d.py --model 8b
"""
import os

import modal

app = modal.App("baseline-d-long-context")

# ---------------------------------------------------------------------------
# Modal image: vLLM for efficient batched local inference of both model
# sizes. Pinned vLLM version matches the one validated elsewhere in this
# project's Modal scripts to avoid guided-decoding / API surface drift.
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system vllm==0.6.6.post1 pandas openpyxl")
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", ""),
        "HF_HOME": "/root/.cache/huggingface",
    })
    .add_local_python_source("common")
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("temporal-memo-data", create_if_missing=True)

MODEL_REGISTRY = {
    "1b": {"name": "meta-llama/Llama-3.2-1B-Instruct", "max_model_len": 131072},
    "8b": {"name": "meta-llama/Llama-3.1-8B-Instruct", "max_model_len": 131072},
}

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. You will be shown a "
    "patient's COMPLETE longitudinal clinical record (every visit note, in "
    "chronological order). Answer the question strictly using information "
    "contained in this record. If the record does not contain enough "
    "information to answer confidently, say so explicitly. Be precise about "
    "dates and the order of events."
)

USER_TEMPLATE = """PATIENT COMPLETE VISIT HISTORY ({n_visits} visits, chronological order):

{full_history}

QUESTION: {question}

Answer the question above using ONLY the visit history provided."""


@app.function(
    image=vllm_image,
    gpu="H100",
    timeout=3 * 3600,
    volumes={"/data": data_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_long_context_baseline(model_key: str = "1b", debug: bool = False, debug_n_queries: int = 5):
    """Serve one model with vLLM and answer all 289 benchmark queries using
    each query's patient's full raw visit history as context.

    Args:
        model_key: "1b" or "8b" (see MODEL_REGISTRY).
        debug: if True, only process a handful of queries for a smoke test.
        debug_n_queries: number of queries to use in debug mode.
    """
    import sys
    sys.path.insert(0, "/root")
    import json
    import time

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from common.data_utils import (
        CANONICAL_PATIENT_IDS,
        load_canonical_cohort,
        format_full_patient_history,
    )

    cfg = MODEL_REGISTRY[model_key]
    model_name = cfg["name"]

    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} via vLLM "
          f"(max_model_len={cfg['max_model_len']}) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {time.time() - t0:.1f}s")

    # --- Load cohort + benchmark queries ---
    cohort = load_canonical_cohort("/data/patient_sequences.jsonl", CANONICAL_PATIENT_IDS)
    cohort_by_id = {p["person_id"]: p for p in cohort}

    all_queries = pd.read_excel("/data/spreadsheet.xlsx")
    all_queries = all_queries[all_queries["person_id"].isin(CANONICAL_PATIENT_IDS)]
    queries = all_queries.to_dict("records")
    if debug:
        queries = queries[:debug_n_queries]
    print(f"[INFO] {len(queries)} queries to process.")

    # --- Pre-compute (and cache) each patient's full-history text once, so
    #     we don't re-serialize the same multi-thousand-visit patient for
    #     every one of their ~15 queries. ---
    # A conservative character budget approximating ~4 chars/token, leaving
    # headroom for the system prompt, question, and generation tokens within
    # the model's max_model_len.
    approx_chars_budget = int(cfg["max_model_len"] * 3.5)
    history_cache = {}
    n_truncated = 0
    for pid, p in cohort_by_id.items():
        full_text = format_full_patient_history(p, max_chars=approx_chars_budget)
        if full_text.endswith("[TRUNCATED: context window exceeded]"):
            n_truncated += 1
        history_cache[pid] = (full_text, len(p.get("visits", [])))
    print(f"[INFO] {n_truncated}/{len(cohort_by_id)} patients required truncation to fit "
          f"the {model_name} context window -- this IS the expected 'Lost in the Middle' "
          f"failure mode Baseline D is designed to surface, not a bug.")

    # --- Build all prompts up front, then generate in one batched vLLM call
    #     for maximum GPU throughput. ---
    prompts = []
    for q in queries:
        pid = q["person_id"]
        full_history, n_visits = history_cache[pid]
        user_msg = USER_TEMPLATE.format(n_visits=n_visits, full_history=full_history, question=q["question"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    print(f"[{time.strftime('%H:%M:%S')}] Generating {len(prompts)} answers ...")
    t_gen = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t_gen
    print(f"[{time.strftime('%H:%M:%S')}] Generation done in {gen_time:.1f}s "
          f"({gen_time / max(len(prompts), 1):.2f}s/query)")

    # --- Write results in the shared evaluation schema ---
    out_name = f"baseline_d_results_{model_key}.jsonl"
    out_path = f"/data/{out_name}"
    with open(out_path, "w") as f:
        for q, output in zip(queries, outputs):
            record = {
                "person_id": q["person_id"],
                "question": q["question"],
                "ground_truth_answer": q.get("answer"),
                "reasoning_type": q.get("reasoning_type"),
                "difficulty": q.get("difficulty"),
                "is_temporal": bool(q.get("is_temporal")),
                "final_answer": output.outputs[0].text.strip(),
                "model": model_name,
                "context_was_truncated": history_cache[q["person_id"]][0].endswith(
                    "[TRUNCATED: context window exceeded]"
                ),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    data_volume.commit()
    print(f"[DONE] Wrote {len(queries)} results to {out_path}")
    return {"model": model_name, "n_queries": len(queries), "n_patients_truncated": n_truncated,
            "gen_time_sec": round(gen_time, 1)}


@app.local_entrypoint()
def main(model: str = "1b", debug: bool = False):
    assert model in MODEL_REGISTRY, f"model must be one of {list(MODEL_REGISTRY)}"
    result = run_long_context_baseline.remote(model_key=model, debug=debug)
    print("DONE:", result)
