"""
common/retrieval_index.py
==========================
Shared hybrid (lexical + semantic) retrieval index used by Baselines E, F, G
and (via ``baseline_h_graph_rag.tri_modal_retrieval``, which extends this
same design with a third temporal-graph modality) Baseline H.

Adapted, consolidated, and heavily commented from the original per-baseline
copies of this module (which were near-identical across
``baseline_g_adaptive_retrieval`` and the pre-Workflow-H hybrid pipelines),
so that every RAG-based baseline in this repository shares ONE scoring
implementation and cannot silently drift apart.

Index-on-disk format (see ``baseline_ef_rag/build_episodic_index.py`` for the
builder), one directory per patient:

    <index_dir>/<person_id>/
    ├── chunks.jsonl            # one structural chunk per line:
    │                           #   {chunk_id, person_id, visit_idx,
    │                           #    visit_datetime, date, section,
    │                           #    text, raw_text}
    │                           #   NOTE: `text` already has temporal
    │                           #   metadata injected as a prefix, e.g.
    │                           #   "[Date: 2019-06-13 | Section: Progress Note] ..."
    ├── bm25_tokenized.json     # list[list[str]], index-aligned with chunks.jsonl
    ├── medcpt_embeddings.npy   # float32 (n_chunks, 768) MedCPT-Article-Encoder
    │                           #   embeddings, index-aligned with chunks.jsonl
    └── index_meta.json         # {person_id, num_chunks, medcpt_model, ...}

This module has NO GPU / API dependency for the lexical (BM25) half. The
semantic (MedCPT) half only needs a pre-computed 768-dim query embedding
vector to rank against; producing that vector requires
``ncbi/MedCPT-Query-Encoder`` (a small, fast, non-GPU-mandatory
transformer), which each baseline's orchestration script calls at query
time (kept out of this module so it stays dependency-light and unit
testable with a random dummy vector).
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/\.]*")


def simple_tokenize(text: str) -> List[str]:
    """Lightweight lowercase alnum tokenizer, used consistently both when the
    corpus was originally tokenized (see ``build_episodic_index.py``) and
    when queries are tokenized here, so lexical scores are self-consistent.
    """
    return [t.lower() for t in _WORD_RE.findall(text)]


class BM25Okapi:
    """Pure NumPy re-implementation of ``rank_bm25.BM25Okapi`` (identical
    defaults: k1=1.5, b=0.75, epsilon=0.25), so the retrieval index has zero
    hard dependency on the ``rank_bm25`` package while remaining a drop-in
    equivalent scorer for a corpus tokenized with that library's conventions.
    """

    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(tokenized_corpus)
        self.doc_len = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.corpus_size else 0.0

        df: Counter = Counter()
        self.doc_freqs: List[Counter] = []
        for doc in tokenized_corpus:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs.keys():
                df[term] += 1

        # Standard Robertson-Sparck-Jones IDF with the BM25Okapi negative-IDF
        # floor (epsilon * average IDF) applied to terms that would otherwise
        # get a negative weight (very common terms).
        self.idf: Dict[str, float] = {}
        neg_idf_terms = []
        avg_idf = 0.0
        for term, freq in df.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term] = idf
            avg_idf += idf
            if idf < 0:
                neg_idf_terms.append(term)
        avg_idf = avg_idf / len(df) if df else 0.0
        eps = self.epsilon * avg_idf
        for term in neg_idf_terms:
            self.idf[term] = eps

    def get_scores(self, query_tokens: List[str], subset_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Score every document (or a subset, if ``subset_idx`` is given --
        used by Baseline H's date-hinted temporal search) against a
        tokenized query."""
        idxs = subset_idx if subset_idx is not None else np.arange(self.corpus_size)
        scores = np.zeros(len(idxs))
        for q in query_tokens:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            tf = np.array([self.doc_freqs[i].get(q, 0) for i in idxs], dtype=np.float64)
            dl = self.doc_len[idxs]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            scores += idf * (tf * (self.k1 + 1) / np.where(denom == 0, 1, denom))
        return scores


class PatientHybridIndex:
    """Loads and holds the BM25 + MedCPT index for a single patient, and
    exposes top-k search over each modality plus a merged/deduped hybrid
    search. This is the retrieval backbone shared by Baselines E, F, and G.
    """

    def __init__(self, person_id: int, base_dir: str):
        self.person_id = person_id
        pdir = os.path.join(base_dir, str(person_id))

        with open(os.path.join(pdir, "index_meta.json")) as f:
            self.meta = json.load(f)

        self.chunks: List[dict] = []
        with open(os.path.join(pdir, "chunks.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        with open(os.path.join(pdir, "bm25_tokenized.json")) as f:
            self.bm25_tokenized = json.load(f)
        assert len(self.bm25_tokenized) == len(self.chunks), (
            f"bm25_tokenized ({len(self.bm25_tokenized)}) / chunks ({len(self.chunks)}) length mismatch"
        )

        self.embeddings = np.load(os.path.join(pdir, "medcpt_embeddings.npy")).astype(np.float32)
        assert self.embeddings.shape[0] == len(self.chunks), (
            f"embeddings ({self.embeddings.shape[0]}) / chunks ({len(self.chunks)}) length mismatch"
        )
        # Pre-normalize embeddings once so cosine similarity reduces to a
        # single dot product at query time.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb_normed = self.embeddings / norms

        self.bm25 = BM25Okapi(self.bm25_tokenized)

    def search_bm25(self, query_text: str, top_k: int = 5) -> List[dict]:
        q_tokens = simple_tokenize(query_text)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {"chunk_idx": int(i), "score": float(scores[i]), "modality": "bm25", **self.chunks[i]}
            for i in top_idx
        ]

    def search_medcpt(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        sims = self._emb_normed @ qn
        top_idx = np.argsort(-sims)[:top_k]
        return [
            {"chunk_idx": int(i), "score": float(sims[i]), "modality": "medcpt", **self.chunks[i]}
            for i in top_idx
        ]

    def hybrid_search(self, query_text: str, query_embedding: Optional[np.ndarray], top_k_each: int = 5) -> List[dict]:
        """Top-K BM25 + Top-K MedCPT, deduplicated by chunk_idx. Chunks
        retrieved by BOTH modalities are ranked first (a strong evidence
        signal), then remaining chunks by their raw modality score.
        """
        bm25_hits = self.search_bm25(query_text, top_k=top_k_each)
        medcpt_hits = self.search_medcpt(query_embedding, top_k=top_k_each) if query_embedding is not None else []

        by_idx: Dict[int, dict] = {}
        for h in bm25_hits:
            by_idx[h["chunk_idx"]] = dict(h, modalities=["bm25"])
        for h in medcpt_hits:
            if h["chunk_idx"] in by_idx:
                by_idx[h["chunk_idx"]]["modalities"].append("medcpt")
            else:
                by_idx[h["chunk_idx"]] = dict(h, modalities=["medcpt"])
        merged = list(by_idx.values())
        merged.sort(key=lambda h: (-len(h["modalities"]), -h["score"]))
        return merged


if __name__ == "__main__":
    # Local smoke test (no GPU/API needed): the BM25 half is fully real; the
    # MedCPT half is exercised with a random unit vector purely to validate
    # shapes/plumbing (real query embeddings require ncbi/MedCPT-Query-Encoder
    # at inference time, see e.g. baseline_g_adaptive_retrieval/orchestrator.py).
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m common.retrieval_index <index_dir> <person_id> [query text]")
        sys.exit(0)

    index_dir, person_id = sys.argv[1], int(sys.argv[2])
    query_text = " ".join(sys.argv[3:]) or "insulin pen malfunction"

    idx = PatientHybridIndex(person_id, base_dir=index_dir)
    print(f"Loaded patient {person_id}: {len(idx.chunks)} chunks, embeddings shape {idx.embeddings.shape}")

    for h in idx.search_bm25(query_text, top_k=5):
        print(f"  [BM25 {h['score']:.3f}] {h.get('date')} | {h.get('section')}: {h.get('raw_text', '')[:90]}")

    rng = np.random.default_rng(0)
    dummy_vec = rng.normal(size=idx.embeddings.shape[1])
    hybrid_hits = idx.hybrid_search(query_text, dummy_vec, top_k_each=5)
    print(f"\nHybrid merged hits (dummy semantic vector, plumbing check only): {len(hybrid_hits)} unique chunks")
