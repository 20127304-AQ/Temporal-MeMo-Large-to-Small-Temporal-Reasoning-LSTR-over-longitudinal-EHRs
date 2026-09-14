"""Evaluation utilities shared by every baseline (D, E, F, G, H).

- ``text_metrics``: pure-Python Exact Match / Precision / Recall / Token-F1 / ROUGE-L.
- ``bertscore_eval``: Modal GPU job computing BERTScore-F1.
- ``llm_judge``: blinded LLM judge (Claude or Mistral) for clinical correctness
  scoring + structured clinical-event extraction.
- ``temporal_metrics``: Temporal MAE (days) and Kendall's Tau, computed from
  the LLM judge's extracted events.
"""
