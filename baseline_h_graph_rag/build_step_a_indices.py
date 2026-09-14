"""
Baseline H — Phase 1, Step A: UUID-Pointer Episodic Grounding Index
======================================================================

Repackages the episodic index built for Baselines E/F/G
(``baseline_ef_rag/build_episodic_index.py`` output, ``shared/hybrid_indices/{pid}/``)
into the format Baseline H's temporal graph needs: every chunk is assigned a
**stable, deterministic UUID** (``uuid5``), and chunks are re-sorted into
strict chronological order with an explicit ``sequence_rank``.

**Why UUIDs matter architecturally.** Baseline H's temporal knowledge graph
(built in ``build_temporal_graph.py``) stores ONLY UUID pointers into this
grounding index — never any clinical text directly. This is a deliberate
design choice to prevent "parametric hallucination": every claim the
Graph-RAG system makes can be traced back to a specific, retrievable chunk
of real source text, and the graph itself carries zero risk of encoding
fabricated facts (it only encodes *structure*: which event happened when,
relative to which other events).

Output (``shared/workflow_H_indices/patient_{person_id}/``):
  - chunks.jsonl       : {uuid, chunk_id, person_id, visit_idx, sequence_rank,
                          visit_datetime, date, section, text, raw_text}
  - bm25_tokenized.json, medcpt_embeddings.npy : re-ordered to match
  - uuid_index.json    : {uuid -> chunk_idx}, for O(1) graph-node resolution
  - index_meta.json    : per-patient metadata + provenance
  - manifest.json / global_uuid_map.json : project-wide summaries (written once, in main())
"""
import json
import os
import uuid

import numpy as np

SRC_DIR = "shared/hybrid_indices"
OUT_DIR = "shared/workflow_H_indices"
COHORT_PATH = "shared/pilot_artifacts/cohort_20_timelines.json"

# Deterministic UUID namespace -- re-running this script is idempotent.
WORKFLOW_H_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "temporal-memo:baseline-h:phase1:chunk-uuid")


def chunk_uuid_for(person_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(WORKFLOW_H_NAMESPACE, f"{person_id}:{chunk_id}"))


def process_patient(person_id: str, manifest_entry: dict, global_uuid_map: dict) -> int:
    src, dst = os.path.join(SRC_DIR, person_id), os.path.join(OUT_DIR, f"patient_{person_id}")
    os.makedirs(dst, exist_ok=True)

    for name in ("chunks.jsonl", "bm25_tokenized.json", "medcpt_embeddings.npy", "index_meta.json"):
        if not os.path.exists(os.path.join(src, name)):
            raise FileNotFoundError(f"Missing expected source artifact: {os.path.join(src, name)}. "
                                     f"Run baseline_ef_rag/build_episodic_index.py first.")

    with open(os.path.join(src, "chunks.jsonl")) as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(src, "bm25_tokenized.json")) as f:
        bm25_tokens = json.load(f)
    embeddings = np.load(os.path.join(src, "medcpt_embeddings.npy"))
    with open(os.path.join(src, "index_meta.json")) as f:
        src_meta = json.load(f)

    n = len(chunks)
    assert len(bm25_tokens) == n and embeddings.shape[0] == n, "index component length mismatch"

    # Chronological ordering (stable sort keeps original section order within a visit).
    order = sorted(range(n), key=lambda i: (chunks[i]["visit_idx"], chunks[i].get("date", "") or ""))

    out_chunks = []
    for rank, i in enumerate(order):
        c = chunks[i]
        cu = chunk_uuid_for(person_id, c["chunk_id"])
        rec = {
            "uuid": cu, "chunk_id": c["chunk_id"], "person_id": c["person_id"],
            "visit_idx": c["visit_idx"], "sequence_rank": rank,
            "visit_datetime": c["visit_datetime"], "date": c.get("date"), "section": c.get("section"),
            "text": c["text"], "raw_text": c.get("raw_text"),
        }
        out_chunks.append(rec)
        global_uuid_map[cu] = {"person_id": person_id, "chunk_id": c["chunk_id"]}

    with open(os.path.join(dst, "chunks.jsonl"), "w") as f:
        for rec in out_chunks:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(dst, "bm25_tokenized.json"), "w") as f:
        json.dump([bm25_tokens[i] for i in order], f)
    np.save(os.path.join(dst, "medcpt_embeddings.npy"), embeddings[order])
    with open(os.path.join(dst, "uuid_index.json"), "w") as f:
        json.dump({rec["uuid"]: idx for idx, rec in enumerate(out_chunks)}, f)
    with open(os.path.join(dst, "index_meta.json"), "w") as f:
        json.dump({
            "person_id": int(person_id), "num_chunks": n,
            "num_visits": src_meta.get("num_visits"),
            "medcpt_model": src_meta.get("medcpt_model", "ncbi/MedCPT-Article-Encoder"),
            "embedding_dim": int(embeddings.shape[1]),
            "uuid_namespace": str(WORKFLOW_H_NAMESPACE),
            "uuid_scheme": "uuid5(namespace, f'{person_id}:{chunk_id}') -- deterministic/reproducible",
            "temporal_metadata_format": "[Date: YYYY-MM-DD | Section: <clinical_header>] injected into chunk text",
            "source_provenance": f"{SRC_DIR}/{person_id}/ (structural chunking + MedCPT embeddings, "
                                  f"see baseline_ef_rag/build_episodic_index.py); re-sorted chronologically "
                                  f"and re-keyed with UUID pointers for Baseline H.",
        }, f, indent=2)

    manifest_entry.update({"num_chunks": n, "num_visits": src_meta.get("num_visits")})
    return n


def main():
    with open(COHORT_PATH) as f:
        patient_ids = [str(p["person_id"]) for p in json.load(f)]

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"patients": {}, "total_chunks": 0}
    global_uuid_map = {}

    total = 0
    for pid in patient_ids:
        manifest["patients"][pid] = {}
        n = process_patient(pid, manifest["patients"][pid], global_uuid_map)
        total += n
        print(f"[OK] patient {pid}: {n} chunks repackaged with UUIDs -> {OUT_DIR}/patient_{pid}/")

    manifest.update({"total_chunks": total, "num_patients": len(patient_ids),
                      "uuid_namespace": str(WORKFLOW_H_NAMESPACE)})
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT_DIR, "global_uuid_map.json"), "w") as f:
        json.dump(global_uuid_map, f)

    print(f"[DONE] total_chunks={total}, uuid_map_size={len(global_uuid_map)}")
    assert total == len(global_uuid_map), "UUID collisions detected!"


if __name__ == "__main__":
    main()
