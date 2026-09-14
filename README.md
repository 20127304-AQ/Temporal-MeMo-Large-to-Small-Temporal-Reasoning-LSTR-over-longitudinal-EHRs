# Temporal-MeMo: Baselines D–H for Temporal Clinical Question Answering over Longitudinal EHRs

This repository consolidates the experimental pipeline code for **five
architectural baselines** developed during the Temporal-MeMo research
project, which studies how well different LLM architectures — from naive
long-context prompting to Graph-RAG with an autonomous small model — answer
**temporal clinical questions** over longitudinal Electronic Health Record
(EHR) timelines (e.g. "When was the patient started on insulin, and how did
their glycemic control trend afterward?").

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Installation](#installation)
- [API Keys](#api-keys)
- [Quickstart per Baseline](#quickstart-per-baseline)
- [Evaluation](#evaluation)
- [Historical Results (Reference)](#historical-results-reference)
- [Design Principles](#design-principles)
- [License](#license)

## Architecture Overview

| Baseline | Name | Memory Mechanism | Executive / Reasoning Model |
|---|---|---|---|
| **D** | Small Model Long-Context | None (full raw history in prompt) | `Llama-3.2-1B` / `Llama-3.1-8B-Instruct` |
| **E** | Pure Episodic RAG | Retrieval only (BM25 + MedCPT) | Claude (Sonnet 4.5) |
| **F** | Standard Hybrid RAG | Retrieval + lightweight semantic-trend LoRA (r=16, 8B) | Claude (Sonnet 4.5), epistemic-hierarchy synthesis |
| **G** | Adaptive Evidence Retrieval + Retrieval-Policy Distillation | Retrieval, guided by a policy-distilled LoRA (r=16, 1B) | Claude (Sonnet 4.5), multi-hop adaptive loop |
| **H** | Graph-RAG + Adaptive SLM | UUID-pointer temporal knowledge graph (Allen's interval relations) + tri-modal search | `Llama-3.2-3B-Instruct` (single model, no external Executive) |

```
D: Full history ──────────────────────────────────► Small Model ──► Answer

E: Question ──► Claude decompose ──► BM25+MedCPT ──► sufficiency loop ──► Claude synthesize (episodic only)

F: Question ──┬─► BM25+MedCPT (Stream A: facts)  ──┐
              └─► Llama-8B+LoRA (Stream B: trend) ──┴─► Claude synthesize (epistemic hierarchy)

G: Question ──► Claude decompose ──► [Llama-1B+policy-LoRA generates search cue ──► BM25+MedCPT
                                       ──► Claude sufficiency check]×N hops ──► Claude synthesize

H: Question ──► Llama-3B ReAct loop: {choose lexical|semantic|temporal search ──► tri-modal
                retrieval ──► sufficiency check}×≤5 hops ──► Llama-3B grounded synthesis
```

Every baseline is compared on the **same** 20-patient pilot cohort and the
**same** 289-question clinical benchmark, so results are directly
comparable across architectures.

## Repository Structure

```
temporal_memo_github_repo/
├── README.md                          <- you are here
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── common/                            # shared utilities used by every baseline
│   ├── data_utils.py                  #   cohort loading, overlapping chunking, query loading
│   └── retrieval_index.py             #   pure NumPy BM25 + hybrid BM25/MedCPT search
│
├── baseline_d_long_context/           # Baseline D
│   └── run_baseline_d.py              #   vLLM long-context inference (1B & 8B)
│
├── baseline_ef_rag/                   # Baselines E & F
│   ├── build_episodic_index.py        #   structural chunking + temporal metadata + BM25/MedCPT
│   ├── run_baseline_e_pure_episodic.py#   Pure Episodic RAG (Baseline E)
│   ├── train_trend_adapter.py         #   Stream B: trend-reflection LoRA (Baseline F prep)
│   └── run_baseline_f_hybrid_rag.py   #   Standard Hybrid RAG (Baseline F)
│
├── baseline_g_adaptive_retrieval/     # Baseline G
│   ├── build_cohort_and_chunks.py     #   Phase 1a: canonical cohort + chunks
│   ├── generate_retrieval_policies.py #   Phase 1b: Qwen2.5-32B retrieval-policy distillation
│   ├── build_sft_data.py              #   Phase 2a: (chunk, policy) SFT pairs
│   ├── train_adapters.py              #   Phase 2b: per-patient LoRA (r=16, Llama-3.2-1B)
│   └── orchestrator.py                #   Phase 3: adaptive retrieval + Claude synthesis
│
├── baseline_h_graph_rag/              # Baseline H
│   ├── build_step_a_indices.py        #   Phase 1, Step A: UUID-pointer grounding index
│   ├── build_windows.py               #   Phase 1, Step B prep: 30-visit windows
│   ├── extract_temporal_events.py     #   Phase 1, Step B: Qwen2.5-32B event extraction
│   ├── build_temporal_graph.py        #   Phase 1, Step B post: deterministic Allen's-interval graph
│   ├── tri_modal_retrieval.py         #   lexical + semantic + temporal-graph search
│   ├── slm_react_engine.py            #   Phases 2&3: Llama-3.2-3B ReAct loop (Modal service)
│   └── orchestrator.py                #   Phases 2&3: local driver
│
└── evaluation/                        # shared by every baseline
    ├── text_metrics.py                #   EM / Precision / Recall / Token-F1 / ROUGE-L
    ├── bertscore_eval.py              #   BERTScore-F1 (Modal GPU)
    ├── llm_judge.py                   #   blinded LLM judge + structured event extraction
    └── temporal_metrics.py            #   Temporal MAE (days) + Kendall's Tau
```

Each subdirectory has its own `README.md` with architecture diagrams and
usage details specific to that baseline.

## Data

This repository expects two input files (not included — bring your own EHR
corpus, or point at a de-identified public dataset such as Synthea):

```
data/
├── patient_sequences.jsonl   # one JSON object per patient:
│                             #   {"person_id": int, "visits": [
│                             #       {"visit_datetime": "YYYY-MM-DDTHH:MM:SS", "text": "..."},
│                             #       ...
│                             #   ]}
└── spreadsheet.xlsx          # benchmark questions, one row per query:
                              #   person_id, question, answer, reasoning_type,
                              #   difficulty, is_temporal
```

The 20-patient pilot cohort used throughout this project's development
(`common/data_utils.CANONICAL_PATIENT_IDS`) was sampled once with a fixed
seed (42) from patients with 10–1,000 visits, then held fixed across every
baseline for direct comparability. Re-sample your own cohort from
`common/data_utils.load_canonical_cohort()` if using different data.

## Installation

```bash
git clone <this-repo>
cd temporal_memo_github_repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Heavy GPU workloads (vLLM serving, LoRA training, embedding) run on Modal:
pip install modal
modal setup
```

Most scripts in this repo are Modal (`modal run script.py`) scripts for GPU
workloads, plus a few plain Python CLI scripts for local orchestration
(the Claude/Mistral-calling "Executive" loops, which are network- not
compute-bound).

## API Keys

Create a `.env` file in the repo root (never commit it — see `.gitignore`):

```bash
ANTHROPIC_API_KEY=sk-ant-...     # required for Baselines E, F, G (Claude Executive)
MISTRAL_API_KEY=...              # optional: fallback LLM-judge provider (evaluation/llm_judge.py)
HF_TOKEN=...                     # for gated HuggingFace models, if applicable
```

Scripts load `.env` explicitly via `python-dotenv` with `override=True` —
they never rely on ambient shell/platform environment variables for API
credentials. Modal scripts additionally require these to be registered as
Modal secrets (`modal.Secret.from_name(...)`); see each script's header for
which secret name it expects (`anthropic-secret`, `mistral-secret`,
`huggingface-secret`).

## Quickstart per Baseline

```bash
# --- Baseline D: Small Model Long-Context ---
modal run baseline_d_long_context/run_baseline_d.py --model 1b
modal run baseline_d_long_context/run_baseline_d.py --model 8b

# --- Baselines E & F: build shared episodic index first ---
modal run baseline_ef_rag/build_episodic_index.py
python baseline_ef_rag/run_baseline_e_pure_episodic.py            # Baseline E
modal run baseline_ef_rag/train_trend_adapter.py --step reflect   # Baseline F prep
modal run baseline_ef_rag/train_trend_adapter.py --step train     # Baseline F prep
python baseline_ef_rag/run_baseline_f_hybrid_rag.py                # Baseline F

# --- Baseline G: Adaptive Evidence Retrieval ---
modal run baseline_g_adaptive_retrieval/build_cohort_and_chunks.py
modal run baseline_g_adaptive_retrieval/generate_retrieval_policies.py
modal run baseline_g_adaptive_retrieval/build_sft_data.py
modal run baseline_g_adaptive_retrieval/train_adapters.py
python baseline_g_adaptive_retrieval/orchestrator.py --max_hops 3

# --- Baseline H: Graph-RAG + Adaptive 3B SLM ---
python baseline_h_graph_rag/build_step_a_indices.py
python baseline_h_graph_rag/build_windows.py
modal run baseline_h_graph_rag/extract_temporal_events.py
python baseline_h_graph_rag/build_temporal_graph.py
modal deploy baseline_h_graph_rag/slm_react_engine.py
python baseline_h_graph_rag/orchestrator.py
```

See each baseline's `README.md` for full architecture diagrams and flags.

## Evaluation

Every baseline writes `phase3_results_{person_id}.jsonl` in the same
schema, so the evaluation pipeline in `evaluation/` applies unchanged to
all of them:

```bash
modal run evaluation/bertscore_eval.py --results-glob "path/to/phase3_results_*.jsonl"
modal run evaluation/llm_judge.py --results-glob "path/to/phase3_results_*.jsonl"
python evaluation/temporal_metrics.py --judge_results evaluation_results/llm_judge_raw.jsonl
```

This produces the 10 core metrics used throughout the project: Exact Match,
Precision, Recall, Token-F1, ROUGE-L, BERTScore-F1, Average Judge Score
(0–3), LLM-Judge Accuracy (%), **Temporal MAE (days)**, and **Kendall's
Tau**. See `evaluation/README.md` for the full metric table.

## Historical Results (Reference)

The following table reproduces the final cross-baseline metrics recorded at
the conclusion of this project's internal research process (20-patient
pilot cohort, 289 queries; consolidated in the project's research log under
"Temporal-MeMo Consolidated Final Project Report"). These are **historical
numbers from prior runs**, provided as a reference point for what this
architecture family is capable of — not a guarantee of what re-running this
repository's code will reproduce on your own data/judge model/API
versions.

| Metric | MeMo 1B (PLoRA) | MeMo 8B (PLoRA) | Direct 1B (D) | Direct 8B (D) | Direct Claude 4.5 | Claude 4.5 + RAG | Hybrid 1B+RAG | Hybrid 8B+RAG (F) | Baseline G (15-hop) |
|---|---|---|---|---|---|---|---|---|---|
| LLM-Judge Accuracy (%) | 11.9 | 13.2 | 21.2 | 58.5 | 70.1 | 59.2 | 60.9 | 59.2 | 49.8 |
| **Temporal MAE (days) ↓** | 88.9 | 25.0 | 42.5 | 4.9 | 5.70 | 18.14 | 18.47 | 18.1 | 5.81 |
| **Kendall's Tau ↑** | 0.980 | 0.850 | 0.755 | 0.909 | 0.805 | 0.782 | 0.811 | 0.794 | 0.246 |

Two additional real, fully-executed evaluation runs from this project
(each with 289/289 queries judged, 0 pipeline errors) are worth calling out
specifically:

- **Baseline G, Executive-model isolation.** Swapping Baseline G's
  Executive from Qwen2.5-32B-Instruct to `claude-sonnet-4-5-20250929` (same
  retrieval architecture, same benchmark, same judge) *improved* LLM-Judge
  Accuracy and Temporal MAE substantially, but *slightly worsened* Kendall's
  Tau (0.205 vs. 0.324) — evidence that Baseline G's event-ordering
  weakness is not primarily an Executive-model-quality problem.
- **Baseline H (Workflow H), full 289-query run.** The Llama-3.2-3B ReAct
  agent completed all 289 queries with 0 pipeline errors. The LLM-judge
  pass had to fall back from Claude/GPT-4o (both accounts had genuinely
  exhausted API credit, verified via real API test calls) to
  `ministral-8b-latest`, which still achieved 99.0% valid-JSON judge output
  across the full run.

### The MAE / Tau trade-off (key qualitative finding)

Across every architecture tested, **absolute-date accuracy (Temporal MAE)
and relative-ordering accuracy (Kendall's Tau) do not move together.**
Small parametric models (MeMo 1B) can achieve near-perfect event ordering
(Tau ≈ 0.98) while being wildly wrong about absolute dates (MAE ≈ 89 days);
RAG-heavy systems pin down dates far more precisely (MAE < 20 days) but
often struggle more with sequencing (Tau as low as ≈ 0.2–0.8). Any single
scalar "accuracy" metric will hide this trade-off — always report both.

## Design Principles

These principles, established early in the research process, motivate the
architectural choices visible throughout this codebase:

1. **Scalability** — avoid fine-tuning large models per patient; prefer
   lightweight adapters (LoRA) or retrieval over full fine-tuning.
2. **Modularity** — separate knowledge internalization (Memory) from
   reasoning (Executive) wherever a parametric memory component exists.
3. **Auditability** — grounded synthesis with verifiable citations
   (`[Date: ... | Section: ...]` / `[chunk_id: ...]` tags); target a 0%
   Unsupported-Fact Rate.
4. **Grounded reasoning** — decouple temporal *routing* (e.g. Baseline H's
   graph) from factual *storage* (raw text chunks, referenced by UUID
   pointer, never duplicated into the graph itself).
5. **Compute efficiency** — enforce formal, mathematical termination
   conditions in agentic retrieval loops (hop caps, duplicate-strategy
   detection) to prevent infinite cycles and runaway API/GPU spend.

## License

MIT — see `LICENSE`.
