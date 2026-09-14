"""
Baseline G — Phase 1 (Stream B): Retrieval-Policy Distillation
=================================================================

Uses a frozen ``Qwen/Qwen2.5-32B-Instruct`` (served via vLLM on a single
Modal GPU) as a "retrieval engineer". For each 30-visit overlapping chunk of
each pilot patient, the model identifies key clinical events and emits a
JSON retrieval policy per event:

    [Sub_query, Patient_Specific_Keywords, Synonyms,
     Target_Clinical_Sections, Approximate_Date_Ranges]

This is the artifact that gets distilled (Phase 2) into a small per-patient
LoRA adapter, teaching a 1B model to predict *how to search this specific
patient's chart* rather than teaching it clinical facts directly — the core
idea behind "Retrieval-Policy Distillation" as opposed to knowledge
distillation.

Input:  /data/pilot_artifacts/chunks_{person_id}.json (from
        ``build_cohort_and_chunks.py``)
Output: /data/baseline_g_reflections/policy_{person_id}.jsonl
        /data/baseline_g_reflections/generation_summary.json
"""
import os

import modal

app = modal.App("baseline-g-retrieval-policy")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system vllm==0.6.6.post1")
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

data_volume = modal.Volume.from_name("temporal-memo-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

CANONICAL_PATIENT_IDS = [
    8855233, 8858035, 8860166, 8860822, 8911710,
    8922374, 8945136, 8955898, 8979075, 9070451,
    9077267, 9088069, 9102132, 9126037, 9354504,
    9989372, 11235719, 11617093, 12015052, 12355020,
]

SYSTEM_PROMPT = """You are an expert clinical retrieval engineer. You will be shown a chunk \
of a single patient's longitudinal clinical record, consisting of a consecutive sequence of \
timestamped visit notes. Your job is NOT to answer clinical questions. Your job is to design a \
RETRIEVAL POLICY that will later help a hybrid BM25 + semantic search system efficiently locate \
evidence for key clinical events, so a downstream QA system can answer questions about this \
patient's history without re-reading the entire chart.

Steps:
1. Identify the KEY CLINICAL EVENTS in this chunk (e.g. admissions, new diagnoses, medication \
   starts/changes/stops, procedures, significant symptom changes/complications, falls, abnormal \
   labs/vitals, care-plan or disposition changes). Focus on events that are informative and \
   specific to this patient, not generic boilerplate.
2. For EACH key clinical event, output one JSON object with exactly these fields:
   - "Sub_query": a natural-language question that a retrieval system could use to find evidence \
     about this specific event.
   - "Patient_Specific_Keywords": list of exact, specific terms/phrases from THIS patient's chart \
     that a lexical (BM25) search should query for.
   - "Synonyms": list of alternate phrasings, abbreviations, or clinically related terms.
   - "Target_Clinical_Sections": list of the chart section / note types where evidence would \
     most likely appear.
   - "Approximate_Date_Ranges": the approximate date range (format "YYYY-MM-DD to YYYY-MM-DD") \
     during which this event and its related evidence would appear.

Output STRICT JSON ONLY: a JSON array of these objects, no prose, no markdown fences, no comments. \
If there are no clear key clinical events in this chunk, output an empty JSON array []."""

USER_TEMPLATE = """Patient ID: {person_id}
Chunk index: {chunk_index} (visits {start}-{end}, dates {date_start} to {date_end})

Below are the {n_visits} consecutive visit notes in this chunk, each prefixed with its timestamp.

{visits_text}

Now identify the key clinical events in this chunk and output the JSON array of retrieval-policy \
objects as instructed."""


def build_chunk_prompt(person_id, chunk, max_chars_per_visit=1500, max_total_chars=26000):
    visits = chunk["visits"]
    lines, total = [], 0
    for v in visits:
        text = (v.get("text") or "").strip()
        if len(text) > max_chars_per_visit:
            text = text[:max_chars_per_visit] + " [...truncated]"
        line = f"[{v.get('visit_datetime', 'unknown_date')}] {text}"
        if total + len(line) > max_total_chars:
            lines.append("[...remaining visits in this chunk truncated for length...]")
            break
        lines.append(line)
        total += len(line)
    return USER_TEMPLATE.format(
        person_id=person_id, chunk_index=chunk["chunk_index"], start=chunk["start"], end=chunk["end"],
        date_start=chunk["date_start"], date_end=chunk["date_end"], n_visits=chunk["n_visits"],
        visits_text="\n\n".join(lines),
    )


def extract_json_array(text):
    import json
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


@app.function(
    image=vllm_image, gpu="H100", timeout=7200,
    volumes={"/data": data_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(debug: bool = False, debug_n_patients: int = 1, debug_n_chunks: int = 2):
    import json
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    patient_ids = CANONICAL_PATIENT_IDS[:debug_n_patients] if debug else CANONICAL_PATIENT_IDS

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(model=MODEL_NAME, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.95, tensor_parallel_size=1)
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {time.time()-t0:.1f}s")

    out_dir = "/data/baseline_g_reflections"
    os.makedirs(out_dir, exist_ok=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1536)
    summary = {"patients": {}, "total_chunks": 0, "total_events": 0, "parse_failures": 0}

    for pid in patient_ids:
        with open(f"/data/pilot_artifacts/chunks_{pid}.json") as f:
            chunks = json.load(f)
        if debug:
            chunks = chunks[:debug_n_chunks]

        prompts = []
        for chunk in chunks:
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_chunk_prompt(pid, chunk)}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        t_start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - t_start

        n_events, n_fail = 0, 0
        with open(f"{out_dir}/policy_{pid}.jsonl", "w") as fout:
            for chunk, output in zip(chunks, outputs):
                parsed = extract_json_array(output.outputs[0].text)
                if parsed is None or not isinstance(parsed, list):
                    n_fail += 1
                    continue
                for event in parsed:
                    if not isinstance(event, dict):
                        continue
                    fout.write(json.dumps({
                        "person_id": pid, "chunk_index": chunk["chunk_index"],
                        "chunk_start_visit": chunk["start"], "chunk_end_visit": chunk["end"],
                        "chunk_date_start": chunk["date_start"], "chunk_date_end": chunk["date_end"],
                        "Sub_query": event.get("Sub_query"),
                        "Patient_Specific_Keywords": event.get("Patient_Specific_Keywords"),
                        "Synonyms": event.get("Synonyms"),
                        "Target_Clinical_Sections": event.get("Target_Clinical_Sections"),
                        "Approximate_Date_Ranges": event.get("Approximate_Date_Ranges"),
                    }) + "\n")
                    n_events += 1

        summary["patients"][str(pid)] = {"n_chunks": len(chunks), "n_events": n_events,
                                          "n_parse_failures": n_fail, "gen_time_sec": round(gen_time, 1)}
        summary["total_chunks"] += len(chunks)
        summary["total_events"] += n_events
        summary["parse_failures"] += n_fail
        print(f"Patient {pid}: {n_events} events, {n_fail} parse failures, {gen_time:.1f}s")

    with open(f"{out_dir}/generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    data_volume.commit()
    print(f"ALL DONE. Summary: {json.dumps(summary, indent=2)}")
    return summary


@app.local_entrypoint()
def main(debug: bool = False):
    print("DONE:", generate.remote(debug=debug))
