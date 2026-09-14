# Baseline H — Graph-RAG Prep + Workflow H Evaluation (Llama-3.2-3B)

## Architecture

```
Phase 1 (prep):
  Episodic index (Baselines E/F)            30-visit windows
        │                                          │
        ▼                                          ▼
  Step A: UUID-pointer                    Step B: Qwen2.5-32B extracts
  grounding index                          (chunk_uuid, event_type,
  (build_step_a_indices.py)                time_start, time_end) triples
        │                                          │
        └───────────────► build_temporal_graph.py ◄┘
                           (Allen's 13 relations,
                            computed DETERMINISTICALLY,
                            never by the LLM)
                                   │
                                   ▼
                         temporal_graph.json
                    (UUID pointers + relations ONLY,
                     zero clinical text stored)

Phases 2 & 3 (inference): a SINGLE Llama-3.2-3B-Instruct model does
everything -- no separate Executive model.

  Question ──► ReAct loop (max 5 hops):
                 1. Action: choose lexical / semantic / temporal search
                 2. Tri-modal retrieval (BM25 + MedCPT + graph lookup)
                 3. Sufficiency check
                 [duplicate-strategy signature -> force-halt cycle guard]
              ──► Grounded synthesis, cited [Date: ... | Section: ...]
```

## Why a graph, and why UUID pointers only

Every earlier baseline in this project either (a) stores clinical facts
parametrically (Baseline D, and the PLoRA components of F/G) or (b) relies
on an expensive frontier-model Executive for every reasoning step (E, F, G).
Baseline H's graph is a **third option**: encode *only structure* (which
event happened when, relative to which others) using Allen's interval
algebra, and store nothing but UUID pointers back to real retrievable
source text at every graph node. This makes hallucination structurally
impossible at the graph level — the graph can be topologically wrong, but
it cannot *invent* a clinical fact, because it contains no clinical facts.

## Why a single 3B model instead of a large Executive

This is the most aggressive cost/complexity baseline in the project:
`Llama-3.2-3B-Instruct` performs decomposition, retrieval-strategy choice,
sufficiency judgment, AND synthesis, entirely locally, with a formal
mathematical halting condition (duplicate-strategy detection) replacing the
external Executive's judgment calls used in E/F/G.

## Usage

```bash
# Phase 1, Step A: UUID-pointer grounding index
python baseline_h_graph_rag/build_step_a_indices.py

# Phase 1, Step B (prep): windowed extraction jobs
python baseline_h_graph_rag/build_windows.py

# Phase 1, Step B: Qwen2.5-32B event extraction
modal run baseline_h_graph_rag/extract_temporal_events.py

# Phase 1, Step B (post-process): deterministic Allen's-interval graph
python baseline_h_graph_rag/build_temporal_graph.py

# Deploy the 3B SLM ReAct engine as a Modal service
modal deploy baseline_h_graph_rag/slm_react_engine.py

# Phases 2 & 3: run the adaptive ReAct loop over the benchmark
python baseline_h_graph_rag/orchestrator.py --patient_ids 8855233,8858035
```

Output schema (`phase3_results_{person_id}.jsonl`) matches every other
baseline in this repo, so `evaluation/` scripts apply unchanged — including
a Baseline-H-specific metric, **Retrieval Iterations** (`n_iterations`),
useful for comparing agentic efficiency across E/F/G/H.
