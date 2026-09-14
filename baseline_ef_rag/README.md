# Baselines E & F — Pure Episodic RAG and Standard Hybrid RAG

## Shared preparation

Both baselines search the same **Episodic Index** ("Stream A"), built once by
`build_episodic_index.py`:

1. **Structural chunking** — each visit note is split by detected clinical
   section headers (Chief Complaint, HPI, Assessment, Plan, Medications, …).
2. **Temporal metadata injection** — every chunk's text is prefixed with
   `[Date: YYYY-MM-DD | Section: <header>]`, making dates/sections directly
   lexically searchable and citable.
3. **BM25 tokenization** of the metadata-prefixed text.
4. **MedCPT embedding** (`ncbi/MedCPT-Article-Encoder`, 768-dim) of the raw
   clinical text for semantic search.

```bash
modal run baseline_ef_rag/build_episodic_index.py
```

Output: `shared/hybrid_indices/{person_id}/{chunks.jsonl, bm25_tokenized.json,
medcpt_embeddings.npy, index_meta.json}` — this is the exact schema consumed
by `common.retrieval_index.PatientHybridIndex`, and also reused as an input
to Baseline G and (via UUID-repackaging) Baseline H.

## Baseline E — Pure Episodic RAG

```
Question
   │
   ▼
Claude: decompose -> search query
   │
   ▼
Hybrid BM25+MedCPT search  ──►  Sufficiency gate (Claude)
   │                                    │
   │            INSUFFICIENT (reformulate, cap 15 chunks)
   │◄───────────────────────────────────┘
   ▼  SUFFICIENT
Claude: synthesize final answer (episodic evidence ONLY, cited)
```

No parametric memory of any kind — everything the model knows about this
patient comes from what it retrieves. This isolates the pure "retrieval
value-add" and is the control against which F and G are compared.

```bash
python baseline_ef_rag/run_baseline_e_pure_episodic.py --patient_ids 8855233,8858035
```

## Baseline F — Standard Hybrid RAG (Dual-Memory)

Adds a lightweight **Semantic Memory** stream ("Stream B"): a per-patient
LoRA adapter (`r=16`, on `meta-llama/Llama-3.1-8B-Instruct`) trained on
trend-only reflections (deliberately *excluding* exact dates/values — those
stay the RAG index's job).

```
                 ┌─── Stream A: Hybrid BM25+MedCPT (facts/dates) ───┐
Question ───────►│                                                  ├──► Claude Executive
                 └─── Stream B: Llama-3.1-8B + LoRA (trends) ───────┘     (epistemic hierarchy
                                                                            synthesis)
```

**Epistemic hierarchy** used at synthesis time:
- Facts, dates, medications, labs → trust Stream A (RAG), cited.
- Trajectory / "improving vs worsening" / relative ordering → Stream B may
  inform this, explicitly flagged as such.
- On conflict → Stream A wins.

```bash
# 1. Generate trend-only reflections (frozen Qwen2.5-32B)
modal run baseline_ef_rag/train_trend_adapter.py --step reflect

# 2. Train per-patient LoRA adapters (Llama-3.1-8B, r=16)
modal run baseline_ef_rag/train_trend_adapter.py --step train

# 3. Run Phase-3 dual-memory synthesis
python baseline_ef_rag/run_baseline_f_hybrid_rag.py --patient_ids 8855233,8858035
```

Both baselines write results to `phase3_results_{person_id}.jsonl` in the
schema consumed by `evaluation/` (see the top-level README for the full
Phase-4 evaluation pipeline: text-overlap metrics, BERTScore, blinded LLM
judge, Temporal MAE, and Kendall's Tau).
