"""
Baseline H — Tri-Modal Search Space
======================================

Extends ``common.retrieval_index`` with a THIRD retrieval modality on top of
lexical (BM25) and semantic (MedCPT): a **topological/temporal** search over
the Allen's-interval event graph built by ``build_temporal_graph.py``.

  1. **Lexical**   — BM25Okapi over per-patient ``bm25_tokenized.json`` /
                      ``chunks.jsonl`` (reuses ``common.retrieval_index.BM25Okapi``).
  2. **Semantic**   — cosine similarity over per-patient
                      ``medcpt_embeddings.npy``; query embedded at call time
                      with ``ncbi/MedCPT-Query-Encoder`` (done by the caller /
                      Modal GPU process — this module only ranks given a
                      vector).
  3. **Temporal / Topological** — filter the patient's event-graph nodes by
                      controlled-vocabulary ``event_type`` and/or an
                      approximate ``date_hint`` prefix match on
                      ``time_start``, then resolve each matching node's
                      ``chunk_uuid`` back to its chunk via ``uuid_index.json``.

Pure Python + NumPy, no GPU/API dependency — can be unit tested locally, and
is uploaded into the Modal container image alongside the GPU-hosted 3B SLM
ReAct loop (``slm_react_engine.py``).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.retrieval_index import BM25Okapi, simple_tokenize  # noqa: E402

WORKFLOW_H_INDICES_DIR = "shared/workflow_H_indices"

EVENT_TYPE_VOCAB = [
    "diagnosis", "disposition", "encounter", "imaging", "lab_result",
    "medication", "other", "procedure", "triage", "vital_sign",
]


class PatientTriModalIndex:
    """Loads and holds all three modalities (lexical / semantic /
    temporal-graph) for a single patient, exposing a unified
    ``search(strategy, query, date_hint, top_k)`` API used by the 3B SLM's
    ReAct action step."""

    def __init__(self, person_id: int, base_dir: str = WORKFLOW_H_INDICES_DIR, graph: Optional[dict] = None):
        self.person_id = person_id
        pdir = os.path.join(base_dir, f"patient_{person_id}")

        with open(os.path.join(pdir, "index_meta.json")) as f:
            self.meta = json.load(f)
        self.chunks: List[dict] = []
        with open(os.path.join(pdir, "chunks.jsonl")) as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
        with open(os.path.join(pdir, "bm25_tokenized.json")) as f:
            self.bm25_tokenized = json.load(f)
        assert len(self.bm25_tokenized) == len(self.chunks)

        self.embeddings = np.load(os.path.join(pdir, "medcpt_embeddings.npy")).astype(np.float32)
        assert self.embeddings.shape[0] == len(self.chunks)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb_normed = self.embeddings / norms

        self.bm25 = BM25Okapi(self.bm25_tokenized)

        with open(os.path.join(pdir, "uuid_index.json")) as f:
            self.uuid_index: Dict[str, int] = json.load(f)  # chunk_uuid -> chunk_idx

        self.dates = [c.get("date", "") or "" for c in self.chunks]

        self.graph_nodes: List[dict] = []
        if graph is not None:
            self.graph_nodes = [n for n in graph.get("nodes", []) if n.get("person_id") == person_id]
            self.graph_nodes.sort(key=lambda n: n.get("time_start") or "")

    def _date_filtered_indices(self, date_hint):
        if not date_hint or str(date_hint).strip().lower() in ("none", "unknown", "n/a", ""):
            return None
        hint = str(date_hint).strip()
        idxs = [i for i, d in enumerate(self.dates) if d.startswith(hint)]
        return idxs if idxs else None

    def search_lexical(self, query_text: str, date_hint=None, top_k: int = 5) -> List[dict]:
        q_tokens = simple_tokenize(query_text)
        subset = self._date_filtered_indices(date_hint)
        if subset is not None:
            scores = self.bm25.get_scores(q_tokens, subset_idx=np.array(subset))
            order = np.argsort(-scores)[:top_k]
            top_idx, top_scores = [subset[i] for i in order], [float(scores[i]) for i in order]
        else:
            scores = self.bm25.get_scores(q_tokens)
            order = np.argsort(-scores)[:top_k]
            top_idx, top_scores = list(order), [float(scores[i]) for i in order]
        return [{"chunk_idx": int(i), "score": s, "modality": "lexical", **self.chunks[i]}
                for i, s in zip(top_idx, top_scores)]

    def search_semantic(self, query_embedding, date_hint=None, top_k: int = 5) -> List[dict]:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        subset = self._date_filtered_indices(date_hint)
        if subset is not None:
            sims = self._emb_normed[subset] @ qn
            order = np.argsort(-sims)[:top_k]
            top_idx, top_scores = [subset[i] for i in order], [float(sims[i]) for i in order]
        else:
            sims = self._emb_normed @ qn
            order = np.argsort(-sims)[:top_k]
            top_idx, top_scores = list(order), [float(sims[i]) for i in order]
        return [{"chunk_idx": int(i), "score": s, "modality": "semantic", **self.chunks[i]}
                for i, s in zip(top_idx, top_scores)]

    def search_temporal(self, event_type=None, date_hint=None, top_k: int = 5) -> List[dict]:
        """Topological/graph search: filter the patient's event-graph nodes
        by controlled-vocabulary event_type and/or an approximate date_hint
        prefix on time_start, then resolve each matching node's chunk_uuid
        back to its chunk via uuid_index. This is the "temporal" modality
        that lexical/semantic search cannot replicate: it answers "what
        happened *around this time*" from graph structure, not text match.
        """
        et = (event_type or "").strip().lower()
        candidates = self.graph_nodes
        if et and et in EVENT_TYPE_VOCAB:
            candidates = [n for n in candidates if n.get("event_type") == et]
        hint = None if (not date_hint or str(date_hint).strip().lower() in ("none", "unknown", "n/a", "")) else str(date_hint).strip()
        if hint:
            filtered = [n for n in candidates if (n.get("time_start") or "").startswith(hint)]
            if filtered:
                candidates = filtered

        hits, seen_idx = [], set()
        for n in candidates[:top_k]:
            cidx = self.uuid_index.get(n["chunk_uuid"])
            if cidx is None or cidx in seen_idx:
                continue
            seen_idx.add(cidx)
            hits.append({"chunk_idx": int(cidx), "score": 1.0, "modality": "temporal",
                         "event_type": n.get("event_type"), "time_start": n.get("time_start"),
                         **self.chunks[cidx]})
        return hits

    def search(self, strategy: str, query=None, query_embedding=None, date_hint=None, top_k: int = 5) -> List[dict]:
        strategy = (strategy or "").strip().lower()
        if strategy == "lexical":
            return self.search_lexical(query or "", date_hint=date_hint, top_k=top_k)
        elif strategy == "semantic":
            if query_embedding is None:
                return self.search_lexical(query or "", date_hint=date_hint, top_k=top_k)
            return self.search_semantic(query_embedding, date_hint=date_hint, top_k=top_k)
        elif strategy in ("temporal", "topological", "graph"):
            return self.search_temporal(event_type=query, date_hint=date_hint, top_k=top_k)
        # Unknown strategy string from the SLM -- fall back to lexical so the
        # loop still makes forward progress instead of erroring out.
        return self.search_lexical(query or "", date_hint=date_hint, top_k=top_k)


def load_temporal_graph(base_dir: str = WORKFLOW_H_INDICES_DIR) -> dict:
    with open(os.path.join(base_dir, "temporal_graph.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    graph = load_temporal_graph()
    idx = PatientTriModalIndex(8855233, graph=graph)
    print(f"Loaded patient 8855233: {len(idx.chunks)} chunks, {len(idx.graph_nodes)} graph nodes")
    for h in idx.search_lexical("lower back pain paracetamol ibuprofen", top_k=5):
        print(f"  [lexical {h['score']:.3f}] {h['date']} | {h['section']}: {h['raw_text'][:90]}")
    for h in idx.search_temporal(event_type="medication", top_k=5):
        print(f"  [temporal] {h['date']} | {h['section']} (event_type={h['event_type']}): {h['raw_text'][:90]}")
