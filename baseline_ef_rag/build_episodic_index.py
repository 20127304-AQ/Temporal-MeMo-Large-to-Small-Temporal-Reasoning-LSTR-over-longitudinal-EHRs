"""
Episodic Index Builder — shared preparation step for Baselines E, F, and G
============================================================================

**Purpose.** Builds the "Episodic Memory" retrieval index that Baselines E
(Pure Episodic RAG), F (Standard Hybrid RAG), and G (Adaptive Evidence
Retrieval) all search over. This is the "Stream A" preparation step of the
project's Dual-Memory architecture: raw visit notes are converted into
retrieval-ready structural chunks with injected temporal metadata, then
indexed with both a lexical (BM25) and a semantic (MedCPT) index.

**Pipeline.**
  1. **Structural chunking.** Each visit's raw note text is split into
     clinically meaningful sections using a controlled-vocabulary header
     detector (Chief Complaint, History of Present Illness, Assessment,
     Plan, Medications, Vitals, Discharge Summary, etc.). Notes without any
     detectable headers are kept as a single "Progress Note" chunk. This
     mirrors real clinical documentation structure far more faithfully than
     fixed-length text windows, and gives the retriever chunks that are
     already topically coherent.
  2. **Temporal metadata injection.** Every chunk's text is prefixed with
     ``[Date: YYYY-MM-DD | Section: <clinical_header>]`` so that (a) BM25
     can lexically match on dates/sections when they appear in a query, and
     (b) the injected string is directly parseable back out of a synthesized
     answer for citation-checking (see ``evaluation/temporal_metrics.py`` and
     the ``[Date: ... | Section: ...]`` citation format used by
     Baseline H's synthesis step).
  3. **BM25 tokenization.** Every chunk's (metadata-prefixed) text is
     tokenized once and cached, so the lexical index never needs to
     retokenize at query time.
  4. **MedCPT embedding.** Every chunk's raw text is embedded with
     ``ncbi/MedCPT-Article-Encoder`` (768-dim, PubMed-pretrained bi-encoder
     — chosen because general-purpose embedding models under-perform on
     clinical terminology). Run on a Modal GPU for throughput; CPU also
     works, just much slower.

**Output** (identical schema consumed by ``common.retrieval_index.PatientHybridIndex``):

    <output_dir>/<person_id>/
    ├── chunks.jsonl            # {chunk_id, person_id, visit_idx, visit_datetime,
    │                           #  date, section, text, raw_text}
    ├── bm25_tokenized.json     # list[list[str]], index-aligned with chunks.jsonl
    ├── medcpt_embeddings.npy   # float32 (n_chunks, 768)
    └── index_meta.json         # {person_id, num_chunks, num_visits, medcpt_model, ...}
"""
import os
import re

import modal

app = modal.App("baseline-ef-build-episodic-index")

ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch transformers numpy pandas openpyxl tqdm"
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("common")
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("temporal-memo-data", create_if_missing=True)

MEDCPT_MODEL = "ncbi/MedCPT-Article-Encoder"

# Controlled vocabulary of clinical section headers we actively look for at
# the start of a line (case-insensitive), roughly ordered by how commonly
# they appear in longitudinal outpatient/inpatient notes. Extend this list if
# your own corpus uses different documentation conventions.
SECTION_HEADERS = [
    "Chief Complaint", "History of Present Illness", "Past Medical History",
    "Medications", "Allergies", "Social History", "Family History",
    "Review of Systems", "Physical Exam", "Vitals", "Vital Signs",
    "Assessment", "Assessment and Plan", "Plan", "Impression",
    "Discharge Summary", "Discharge Diagnosis", "Disposition",
    "Progress Note", "Nursing Note", "Admission Note",
    "Laboratory Results", "Labs", "Imaging", "Procedure Note",
    "Physiotherapy Note", "Allied Health Note",
]
_HEADER_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_into_sections(note_text: str):
    """Split one visit's raw note text into (section_label, section_text)
    pairs using the controlled-vocabulary header detector above. Falls back
    to a single ("Progress Note", full_text) chunk if no headers are found.
    """
    matches = list(_HEADER_PATTERN.finditer(note_text))
    if not matches:
        return [("Progress Note", note_text.strip())]

    sections = []
    for i, m in enumerate(matches):
        label = m.group(1).strip().title()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note_text)
        body = note_text[start:end].strip()
        if body:
            sections.append((label, body))
    # Anything before the first detected header is still real clinical text
    # (often the note's opening line) -- keep it under a generic label.
    if matches[0].start() > 0:
        preamble = note_text[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, ("Progress Note", preamble))
    return sections or [("Progress Note", note_text.strip())]


def build_structural_chunks(person_id: int, visits: list):
    """Convert a patient's chronological visit list into structural chunks
    with injected temporal metadata. Returns a list of chunk dicts ready to
    be tokenized + embedded."""
    chunks = []
    for visit_idx, visit in enumerate(visits):
        raw_text = (visit.get("text") or "").strip()
        visit_dt = visit.get("visit_datetime", "")
        date_str = visit_dt.split("T")[0] if visit_dt else None
        if not raw_text:
            continue
        for section_i, (section_label, section_text) in enumerate(split_into_sections(raw_text)):
            if not section_text:
                continue
            metadata_prefix = f"[Date: {date_str} | Section: {section_label}] "
            chunk_id = f"{person_id}_v{visit_idx}_s{section_i}"
            chunks.append({
                "chunk_id": chunk_id,
                "person_id": person_id,
                "visit_idx": visit_idx,
                "visit_datetime": visit_dt,
                "date": date_str,
                "section": section_label,
                "text": metadata_prefix + section_text,   # metadata-injected, used for BM25 + citations
                "raw_text": section_text,                  # original clinical text, used for MedCPT embedding
            })
    return chunks


@app.function(
    image=ml_image,
    gpu="a10g",
    timeout=3600,
    volumes={"/data": data_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def build_index_for_patients(person_ids: list = None, out_subdir: str = "hybrid_indices"):
    """Build the full episodic index (structural chunks + BM25 tokens +
    MedCPT embeddings) for every requested patient, writing to
    ``/data/<out_subdir>/<person_id>/``.
    """
    import sys
    sys.path.insert(0, "/root")
    import json
    import time

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from common.data_utils import CANONICAL_PATIENT_IDS, load_canonical_cohort
    from common.retrieval_index import simple_tokenize

    person_ids = person_ids or CANONICAL_PATIENT_IDS
    cohort = load_canonical_cohort("/data/patient_sequences.jsonl", person_ids)

    print(f"[{time.strftime('%H:%M:%S')}] Loading {MEDCPT_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MEDCPT_MODEL)
    model = AutoModel.from_pretrained(MEDCPT_MODEL).to("cuda").eval()

    out_root = f"/data/{out_subdir}"
    os.makedirs(out_root, exist_ok=True)

    summary = {}
    for patient in cohort:
        pid = patient["person_id"]
        t0 = time.time()
        chunks = build_structural_chunks(pid, patient.get("visits", []))
        if not chunks:
            print(f"[WARN] patient {pid}: no chunks produced, skipping.")
            continue

        # --- BM25 tokenization (tokenize the metadata-prefixed text so the
        #     injected [Date: ... | Section: ...] tokens are searchable) ---
        bm25_tokens = [simple_tokenize(c["text"]) for c in chunks]

        # --- MedCPT embeddings, batched for throughput ---
        embeddings = []
        batch_size = 16
        raw_texts = [c["raw_text"] for c in chunks]
        with torch.no_grad():
            for i in range(0, len(raw_texts), batch_size):
                batch = raw_texts[i:i + batch_size]
                encoded = tokenizer(
                    batch, truncation=True, padding=True, return_tensors="pt", max_length=512
                ).to("cuda")
                out = model(**encoded).last_hidden_state[:, 0, :]  # CLS-pooled, per MedCPT convention
                embeddings.append(out.float().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)

        pdir = os.path.join(out_root, str(pid))
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, "chunks.jsonl"), "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        with open(os.path.join(pdir, "bm25_tokenized.json"), "w") as f:
            json.dump(bm25_tokens, f)
        np.save(os.path.join(pdir, "medcpt_embeddings.npy"), embeddings)
        with open(os.path.join(pdir, "index_meta.json"), "w") as f:
            json.dump({
                "person_id": pid,
                "num_chunks": len(chunks),
                "num_visits": len(patient.get("visits", [])),
                "medcpt_model": MEDCPT_MODEL,
                "embedding_dim": int(embeddings.shape[1]),
                "bm25_library": "custom BM25Okapi (see common/retrieval_index.py)",
                "temporal_metadata_format": "[Date: YYYY-MM-DD | Section: <clinical_header>] prefix injected into chunk text",
            }, f, indent=2)

        elapsed = time.time() - t0
        summary[str(pid)] = {"n_chunks": len(chunks), "elapsed_sec": round(elapsed, 1)}
        print(f"[OK] patient {pid}: {len(chunks)} chunks indexed in {elapsed:.1f}s")

    with open(os.path.join(out_root, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    data_volume.commit()
    print(f"[DONE] Indexed {len(summary)} patients -> {out_root}")
    return summary


@app.local_entrypoint()
def main(debug: bool = False):
    from common.data_utils import CANONICAL_PATIENT_IDS
    pids = CANONICAL_PATIENT_IDS[:2] if debug else None
    result = build_index_for_patients.remote(person_ids=pids)
    print("DONE:", result)
