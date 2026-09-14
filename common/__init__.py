"""
Temporal-MeMo :: common

Shared, dependency-light utilities used by every baseline in this repository:

- ``data_utils``: canonical patient cohort loading, deterministic overlapping
  chunking of longitudinal visit sequences, and benchmark-query loading.
- ``retrieval_index``: a pure-Python/NumPy hybrid lexical (BM25) + semantic
  (MedCPT cosine-similarity) retrieval index that every RAG-based baseline
  (E, F, G, H) builds on top of.

Keeping this logic in one place guarantees that every baseline in the repo
uses byte-identical chunking, cohort selection, and retrieval scoring code,
which is what makes their evaluation numbers directly comparable.
"""
from . import data_utils  # noqa: F401
from . import retrieval_index  # noqa: F401
