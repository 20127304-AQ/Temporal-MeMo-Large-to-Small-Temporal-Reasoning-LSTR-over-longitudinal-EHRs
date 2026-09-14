# Baseline G — Adaptive Evidence Retrieval + Retrieval-Policy Distillation

## Architecture

```
Phase 1: Retrieval-Policy Distillation (prep)
  30-visit chunks ──► Qwen2.5-32B-Instruct ──► JSON retrieval policies
                       (frozen "retrieval engineer")   (Sub_query, Keywords,
                                                         Synonyms, Sections,
                                                         Date Ranges)

Phase 2: Policy Injection (prep)
  (chunk, policy) SFT pairs ──► LoRA (r=16) on Llama-3.2-1B-Instruct
                                 -> one adapter per patient

Phase 3: Adaptive Evidence Retrieval + Executive Synthesis (inference)

  Question ──► Claude: decompose into evidence requirements
                        │
             ┌──────────▼──────────────────────────────┐
             │  loop up to MAX_HOPS times:              │
             │   1. SLM (1B+adapter) → retrieval cue    │
             │   2. Hybrid BM25+MedCPT search            │
             │   3. Claude: sufficiency check             │
             └──────────┬──────────────────────────────┘
                        ▼ SUFFICIENT
              Claude: synthesize final answer (cited)
```

## Why "Retrieval-Policy Distillation" instead of knowledge distillation

Earlier baselines (D) tried to make a small model *memorize clinical facts*
directly — this is brittle and hallucination-prone. Baseline G instead
teaches the small model to *memorize how to search this specific patient's
chart* (which keywords/sections/date-ranges are informative for which kinds
of events). Facts themselves always come from real retrieved text, never
from the small model's parameters — eliminating a large class of
hallucinations while still personalizing retrieval per patient.

## Usage

```bash
# Phase 1a: canonical cohort + chunks
modal run baseline_g_adaptive_retrieval/build_cohort_and_chunks.py

# Phase 1b: retrieval-policy distillation (Qwen2.5-32B)
modal run baseline_g_adaptive_retrieval/generate_retrieval_policies.py

# Phase 2a: build SFT pairs
modal run baseline_g_adaptive_retrieval/build_sft_data.py

# Phase 2b: train per-patient LoRA adapters (Llama-3.2-1B, r=16)
modal run baseline_g_adaptive_retrieval/train_adapters.py

# Phase 3: adaptive retrieval + Claude executive synthesis
python baseline_g_adaptive_retrieval/orchestrator.py \
  --patient_ids 8855233,8858035 --max_hops 3
```

`--max_hops` supports both experimental configurations used in this
project: the original safety cap of **3** hops, and an extended "bounded
sufficiency" variant of **15** hops. See the top-level README for the
(counter-intuitive) empirical finding: raising the cap from 3→15 barely
moved Kendall's Tau (0.205 vs 0.324 in one measured run), indicating
event-ordering failures are not primarily a "not enough evidence gathered"
problem.

## Requirements

- `ANTHROPIC_API_KEY` in your project `.env` (Executive model calls).
- A deployed Modal service exposing the SLM router
  (`SLMGuidedRetrieval.generate_cue(person_id, requirement)`), built from the
  Phase-2 adapters in `train_adapters.py` — see that file's docstring for the
  adapter loading pattern to wrap in a `modal.Cls`.
- Episodic index built via `baseline_ef_rag/build_episodic_index.py`.

Output schema (`phase3_results_{person_id}.jsonl`) matches every other
baseline in this repo, so `evaluation/` scripts apply unchanged.
