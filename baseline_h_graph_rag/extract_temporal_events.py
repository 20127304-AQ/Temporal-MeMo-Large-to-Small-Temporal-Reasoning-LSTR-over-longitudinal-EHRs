"""
Baseline H — Phase 1, Step B: Temporal Event Extraction (Qwen2.5-32B via vLLM)
=================================================================================

For each windowed extraction job built by ``build_windows.py``, prompts a
frozen ``Qwen/Qwen2.5-32B-Instruct`` to identify clinical EVENTS in that
window and, for each one, output:
  - ``chunk_uuid``: the UUID pointer (from Step A) to the source chunk this
    event was grounded in (never raw text — see ``build_step_a_indices.py``
    docstring for why).
  - ``event_type``: one of a 10-value controlled vocabulary (diagnosis,
    disposition, encounter, imaging, lab_result, medication, other,
    procedure, triage, vital_sign).
  - ``time_start`` / ``time_end``: an ISO-ish timestamp interval for the
    event (may be a point-in-time event, i.e. ``time_start == time_end``).

**Deliberately NOT asked of the LLM: Allen's interval relations.** Asking a
32B model to directly output pairwise interval logic (before/after/overlaps/
etc.) for every event pair is both expensive and unreliable. Instead, the
LLM's job is scoped to the much easier, more reliable sub-task of extracting
*(chunk_uuid, event_type, time interval)* triples; the actual Allen relation
between any two events is then computed **deterministically** from those
intervals in ``build_temporal_graph.py``. This decomposition — LLM for
fuzzy extraction, code for exact logic — is a recurring design pattern in
this baseline and is what keeps the resulting graph auditable.

Output: shared/workflow_H_indices/step_b_raw_results.json
  {"n_windows": int, "n_parse_fail": int,
   "results": [{"person_id", "window_idx", "parsed": {"events": [...]}}]}
"""
import os

import modal

app = modal.App("baseline-h-extract-temporal-events")

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

EVENT_TYPE_VOCAB = [
    "diagnosis", "disposition", "encounter", "imaging", "lab_result",
    "medication", "other", "procedure", "triage", "vital_sign",
]

SYSTEM_PROMPT = f"""You are a clinical timeline extraction engine. You will be shown a window \
of a patient's structural chart chunks, each tagged with its UUID and a [Date: ... | Section: ...] \
prefix. Identify discrete clinical EVENTS in this window (diagnoses, medication changes, \
procedures, encounters, lab results, imaging, vital signs, triage, disposition changes).

For each event, output a JSON object with EXACTLY these fields:
  - "chunk_uuid": the UUID of the chunk this event's evidence comes from (copy verbatim from \
    the chunk's uuid, never invent one).
  - "event_type": EXACTLY one of {EVENT_TYPE_VOCAB}.
  - "time_start": ISO-ish timestamp ("YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD") when the event begins.
  - "time_end": timestamp when the event ends (same as time_start for point-in-time events).

Output STRICT JSON ONLY: {{"events": [ ... ]}}, no prose, no markdown fences."""

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system vllm==0.6.6.post1")
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("temporal-memo-data", create_if_missing=True)


def build_window_prompt(window: dict) -> str:
    lines = [f"[uuid: {c['uuid']}] [Date: {c['date']} | Section: {c['section']}] {c['text']}"
             for c in window["chunks"]]
    return (f"Patient {window['person_id']}, window {window['window_idx']} "
            f"(visits {window['visit_start']}-{window['visit_end']}):\n\n" + "\n\n".join(lines))


def extract_json_object(text: str):
    import json
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


@app.function(
    image=vllm_image, gpu="H100", timeout=3 * 3600,
    volumes={"/data": data_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def extract(debug: bool = False, debug_n: int = 10):
    import json
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    windows = []
    with open("/data/workflow_H_indices/graph_extraction_jobs.jsonl") as f:
        for line in f:
            if line.strip():
                windows.append(json.loads(line))
    if debug:
        windows = windows[:debug_n]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(model=MODEL_NAME, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_window_prompt(w)}],
            tokenize=False, add_generation_prompt=True,
        )
        for w in windows
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Generating extractions for {len(prompts)} windows ...")
    outputs = llm.generate(prompts, sampling_params)

    results, n_parse_fail = [], 0
    for w, out in zip(windows, outputs):
        parsed = extract_json_object(out.outputs[0].text)
        if parsed is None:
            n_parse_fail += 1
        results.append({"person_id": w["person_id"], "window_idx": w["window_idx"], "parsed": parsed})

    out_payload = {"n_windows": len(windows), "n_parse_fail": n_parse_fail, "results": results}
    with open("/data/workflow_H_indices/step_b_raw_results.json", "w") as f:
        json.dump(out_payload, f)
    data_volume.commit()

    print(f"[DONE] {len(windows)} windows, {n_parse_fail} parse failures "
          f"({100 * (1 - n_parse_fail / max(len(windows), 1)):.1f}% valid JSON)")
    return {"n_windows": len(windows), "n_parse_fail": n_parse_fail}


@app.local_entrypoint()
def main(debug: bool = False):
    print("DONE:", extract.remote(debug=debug))
