"""
Baseline G — Phase 2a: Build per-patient SFT (prompt, completion) pairs.

Reconstructs the exact 30-visit/stride-25 chunks Phase 1 used, rebuilds
Phase 1's exact system/user prompt per chunk, and pairs each chunk's prompt
with the JSON array of retrieval-policy events generated for that
chunk_index (from ``generate_retrieval_policies.py``) as the target
completion — this (prompt, completion) pair is what the Phase-2 LoRA
adapter is trained to reproduce, patient by patient.

Output: /data/baseline_g_sft_data/sft_{person_id}.jsonl
Each line: {"person_id":..., "chunk_index":..., "prompt": "...", "completion": "..."}
"""
import json
import os

from generate_retrieval_policies import SYSTEM_PROMPT, build_chunk_prompt, CANONICAL_PATIENT_IDS

COHORT_PATH = "/data/pilot_artifacts/cohort_20_timelines.json"
POLICY_DIR = "/data/baseline_g_reflections"
OUT_DIR = "/data/baseline_g_sft_data"
CHUNK_SIZE, STRIDE = 30, 25


def build_chunks(visits, chunk_size=CHUNK_SIZE, stride=STRIDE):
    n = len(visits)
    chunks, start, idx = [], 0, 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_visits = visits[start:end]
        chunks.append({
            "chunk_index": idx, "start": start, "end": end, "n_visits": len(chunk_visits),
            "date_start": chunk_visits[0].get("visit_datetime") if chunk_visits else None,
            "date_end": chunk_visits[-1].get("visit_datetime") if chunk_visits else None,
            "visits": chunk_visits,
        })
        idx += 1
        if end == n:
            break
        start += stride
    return chunks


def main():
    with open(COHORT_PATH) as f:
        cohort = {p["person_id"]: p for p in json.load(f)}

    os.makedirs(OUT_DIR, exist_ok=True)
    for pid in CANONICAL_PATIENT_IDS:
        if pid not in cohort:
            continue
        chunks = build_chunks(cohort[pid].get("visits", []))

        policy_path = f"{POLICY_DIR}/policy_{pid}.jsonl"
        if not os.path.exists(policy_path):
            print(f"[SKIP] {pid}: no policy file found, run generate_retrieval_policies.py first.")
            continue

        # Group retrieval-policy events by their source chunk_index.
        events_by_chunk = {}
        with open(policy_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                events_by_chunk.setdefault(rec["chunk_index"], []).append({
                    "Sub_query": rec["Sub_query"],
                    "Patient_Specific_Keywords": rec["Patient_Specific_Keywords"],
                    "Synonyms": rec["Synonyms"],
                    "Target_Clinical_Sections": rec["Target_Clinical_Sections"],
                    "Approximate_Date_Ranges": rec["Approximate_Date_Ranges"],
                })

        examples = []
        for chunk in chunks:
            events = events_by_chunk.get(chunk["chunk_index"], [])
            prompt_text = SYSTEM_PROMPT + "\n\n" + build_chunk_prompt(pid, chunk)
            completion_text = json.dumps(events, ensure_ascii=False)
            examples.append({
                "person_id": pid, "chunk_index": chunk["chunk_index"],
                "prompt": prompt_text, "completion": completion_text,
            })

        with open(f"{OUT_DIR}/sft_{pid}.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"[OK] patient {pid}: {len(examples)} SFT examples written")


if __name__ == "__main__":
    main()
