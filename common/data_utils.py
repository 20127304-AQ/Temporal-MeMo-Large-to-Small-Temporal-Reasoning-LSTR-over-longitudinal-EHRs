"""
common/data_utils.py
=====================
Shared data-loading and chunking utilities used by ALL baselines (D, E, F, G, H).

Every experiment in the Temporal-MeMo project operates over the exact same
20-patient pilot cohort and the exact same 289-query clinical benchmark, so
that results are directly comparable across architectures. This module
centralizes that logic instead of re-implementing it (with subtle drift) in
every baseline folder, which is exactly what happened during the original
research process and is corrected here for the public repository.

Expected input data layout (see the top-level README "Data" section):

    data/
    ├── patient_sequences.jsonl   # one JSON object per patient:
    │                             #   {"person_id": int, "visits": [
    │                             #       {"visit_datetime": "YYYY-MM-DDTHH:MM:SS", "text": "..."},
    │                             #       ...
    │                             #   ]}
    └── benchmark_queries.xlsx    # one row per benchmark question:
                                  #   person_id, question, answer, reasoning_type,
                                  #   difficulty, is_temporal

Nothing in this file makes any network calls; it is pure local I/O + string/
dict manipulation so it can be unit tested without a GPU or API key.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical 20-patient pilot cohort.
#
# These are the exact person_ids used across every baseline (D through H) in
# this project, sampled once with a fixed random seed (42) from all patients
# in the raw corpus with between 10 and 1,000 visits (enough longitudinal
# depth to exercise temporal reasoning, but not so many that a single
# patient's record is unmanageable). Hard-coding them (rather than
# re-deriving via RNG in every script) guarantees perfect alignment across
# every baseline's results -- this is the single most important
# reproducibility invariant in the whole project.
# ---------------------------------------------------------------------------
CANONICAL_PATIENT_IDS: List[int] = [
    8855233, 8858035, 8860166, 8860822, 8911710,
    8922374, 8945136, 8955898, 8979075, 9070451,
    9077267, 9088069, 9102132, 9126037, 9354504,
    9989372, 11235719, 11617093, 12015052, 12355020,
]

# Overlapping chunk convention used throughout the project: 30 consecutive
# visits per chunk, sliding the window forward by 25 visits each time (i.e.
# a 5-visit overlap between consecutive chunks). The overlap is intentional:
# it lets downstream consumers (LLM extraction, graph construction) merge or
# de-duplicate information that spans a chunk boundary instead of losing it.
DEFAULT_CHUNK_SIZE = 30
DEFAULT_STRIDE = 25

# Visit-count bounds used when the cohort was originally sampled (kept here
# purely for documentation / re-sampling from scratch if you use your own data).
MIN_VISITS = 10
MAX_VISITS = 1000

RANDOM_SEED = 42


def load_patient_sequences(path: str) -> Dict[int, dict]:
    """Load the raw longitudinal patient corpus into a dict keyed by person_id.

    Args:
        path: path to a JSONL file, one patient record per line, each with at
            least ``{"person_id": int, "visits": [...]}``.

    Returns:
        Dict mapping ``person_id -> patient record``.
    """
    id_to_patient: Dict[int, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            id_to_patient[rec["person_id"]] = rec
    return id_to_patient


def load_canonical_cohort(
    patient_sequences_path: str,
    patient_ids: Optional[List[int]] = None,
) -> List[dict]:
    """Load the canonical (or a caller-supplied) patient cohort and sanity-check
    that every patient satisfies the [MIN_VISITS, MAX_VISITS] visit-count
    criterion used for the original sampling.

    Args:
        patient_sequences_path: path to patient_sequences.jsonl.
        patient_ids: optional override list of person_ids; defaults to the
            20 canonical pilot patients.

    Returns:
        List of patient records (dicts) in the same order as ``patient_ids``.
    """
    patient_ids = patient_ids or CANONICAL_PATIENT_IDS
    id_to_patient = load_patient_sequences(patient_sequences_path)

    missing = [pid for pid in patient_ids if pid not in id_to_patient]
    if missing:
        raise RuntimeError(f"Requested patient IDs not found in corpus: {missing}")

    out_of_range = {}
    for pid in patient_ids:
        n_visits = len(id_to_patient[pid].get("visits", []))
        if not (MIN_VISITS <= n_visits <= MAX_VISITS):
            out_of_range[pid] = n_visits
    if out_of_range:
        print(f"[WARN] Patients outside the [{MIN_VISITS},{MAX_VISITS}] visit-count "
              f"range: {out_of_range} (proceeding anyway; this is informational).")

    return [id_to_patient[pid] for pid in patient_ids]


def build_overlapping_chunks(
    visits: List[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> List[dict]:
    """Split a single patient's chronologically-ordered visit list into
    overlapping chunks of ``chunk_size`` visits, advancing by ``stride``
    visits each step (so consecutive chunks overlap by
    ``chunk_size - stride`` visits).

    This exact (30, 25) chunking scheme is used to:
      * generate multi-granularity synthesis prompts for the 32B "Generator"
        model (Baselines D-prep / G / H graph extraction),
      * build the structural/episodic RAG index chunk boundaries feeding
        into Baselines E, F, G, H (via ``common.retrieval_index``).

    Args:
        visits: chronologically-sorted list of visit dicts, each with at
            least ``visit_datetime`` and ``text``.
        chunk_size: number of visits per chunk.
        stride: step size between chunk starts (< chunk_size implies overlap).

    Returns:
        List of chunk dicts: ``{chunk_index, start, end, n_visits,
        date_start, date_end, visits}``.
    """
    n = len(visits)
    chunks: List[dict] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_visits = visits[start:end]
        chunks.append({
            "chunk_index": idx,
            "start": start,
            "end": end,
            "n_visits": len(chunk_visits),
            "date_start": chunk_visits[0].get("visit_datetime") if chunk_visits else None,
            "date_end": chunk_visits[-1].get("visit_datetime") if chunk_visits else None,
            "visits": chunk_visits,
        })
        idx += 1
        if end == n:
            break
        start += stride
    return chunks


def load_benchmark_queries(spreadsheet_path: str, person_id: Optional[int] = None) -> List[dict]:
    """Load the 289-query clinical benchmark (or a single patient's subset).

    Args:
        spreadsheet_path: path to the benchmark_queries.xlsx spreadsheet with
            columns ``person_id, question, answer, reasoning_type,
            difficulty, is_temporal``.
        person_id: if given, filter to a single patient's queries.

    Returns:
        List of row dicts.
    """
    import pandas as pd  # local import: keeps this module importable without pandas installed

    df = pd.read_excel(spreadsheet_path)
    if person_id is not None:
        df = df[df["person_id"] == person_id]
    return df.to_dict("records")


def format_full_patient_history(patient_record: dict, max_chars: Optional[int] = None) -> str:
    """Concatenate a patient's ENTIRE visit history into one plain-text blob,
    each visit prefixed with its timestamp. This is the "naive" long-context
    representation fed directly into small models in Baseline D -- no
    chunking, no retrieval, no summarization.

    Args:
        patient_record: a single patient dict from ``load_canonical_cohort``.
        max_chars: if given, hard-truncate the resulting text (this is what
            actually happens on-device when a model's context window is
            smaller than the full history -- the very failure mode Baseline D
            is designed to characterize, so truncation is applied rather than
            silently working around it).

    Returns:
        A single string with one "[visit_datetime] text" line per visit.
    """
    lines = []
    for v in patient_record.get("visits", []):
        lines.append(f"[{v.get('visit_datetime', 'unknown_date')}] {(v.get('text') or '').strip()}")
    full_text = "\n\n".join(lines)
    if max_chars is not None and len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n...[TRUNCATED: context window exceeded]"
    return full_text


if __name__ == "__main__":
    # Lightweight smoke test -- requires only the standard library, run with:
    #   python -m common.data_utils /path/to/patient_sequences.jsonl
    import sys

    if len(sys.argv) > 1:
        cohort = load_canonical_cohort(sys.argv[1])
        print(f"Loaded {len(cohort)} canonical patients.")
        for p in cohort[:3]:
            chunks = build_overlapping_chunks(p["visits"])
            print(f"  person_id={p['person_id']}: {len(p['visits'])} visits -> {len(chunks)} chunks")
    else:
        print("Usage: python -m common.data_utils /path/to/patient_sequences.jsonl")
