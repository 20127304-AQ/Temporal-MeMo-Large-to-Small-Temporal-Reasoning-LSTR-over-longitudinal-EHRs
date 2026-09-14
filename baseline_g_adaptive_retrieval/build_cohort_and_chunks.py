"""
Baseline G — Phase 1a: Build the canonical pilot cohort + 30-visit chunks.

Reproduces the canonical 20-patient pilot cohort from the raw
``patient_sequences.jsonl`` corpus and builds 30-visit overlapping chunks
(stride 25, i.e. a 5-visit overlap) per patient — the exact same chunking
convention used across every baseline in this repository
(``common.data_utils.build_overlapping_chunks``).

We select the canonical person_ids directly (rather than re-deriving them
via RNG on every run, which risks implementation drift) to guarantee
alignment with every other baseline's pilot cohort.

Outputs (written to ``/data/pilot_artifacts/`` on the Modal volume):
  - cohort_20_timelines.json : the 20 canonical patients' full visit records
  - chunks_summary.json      : per-patient chunk counts + totals
  - chunks_{person_id}.json  : list of 30-visit chunks for each patient
"""
import os

import modal

app = modal.App("baseline-g-build-cohort")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system pandas openpyxl")
    .add_local_python_source("common")
)

volume = modal.Volume.from_name("temporal-memo-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": volume}, timeout=600, cpu=2, memory=8192)
def build():
    import json

    from common.data_utils import CANONICAL_PATIENT_IDS, load_canonical_cohort, build_overlapping_chunks

    out_dir = "/data/pilot_artifacts"
    os.makedirs(out_dir, exist_ok=True)

    cohort = load_canonical_cohort("/data/patient_sequences.jsonl", CANONICAL_PATIENT_IDS)
    with open(f"{out_dir}/cohort_20_timelines.json", "w") as f:
        json.dump(cohort, f)

    summary = {"chunk_size": 30, "stride": 25, "patients": {}}
    total_chunks = 0
    for p in cohort:
        pid = p["person_id"]
        chunks = build_overlapping_chunks(p.get("visits", []))
        with open(f"{out_dir}/chunks_{pid}.json", "w") as f:
            json.dump(chunks, f)
        total_chunks += len(chunks)
        summary["patients"][str(pid)] = {"n_visits": len(p.get("visits", [])), "n_chunks": len(chunks)}
        print(f"  patient {pid}: {len(p.get('visits', []))} visits -> {len(chunks)} chunks")

    summary["total_chunks"] = total_chunks
    summary["canonical_patient_ids"] = CANONICAL_PATIENT_IDS
    with open(f"{out_dir}/chunks_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main():
    print("DONE:", build.remote())
