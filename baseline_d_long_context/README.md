# Baseline D — Small Model Long-Context Prompting

## Architecture

The naive control: the *entire* raw patient visit history is concatenated
into a single plain-text block and fed directly into a small instruction
model's context window. No chunking, no retrieval, no memory adapters, no
summarization — whatever the model can natively attend to is all it has.

```
Full raw patient history (all visits, chronological)
        │
        ▼
┌───────────────────────────┐
│  Llama-3.2-1B-Instruct    │   <- native context window only
│  Llama-3.1-8B-Instruct    │
└───────────┬───────────────┘
            ▼
      Final Answer
```

## Why this baseline exists

It establishes the **performance floor**: how well can a small model answer
temporal clinical questions when the model is simply shown everything, with
no architectural help? This is also the baseline most exposed to the
["Lost in the Middle"](https://arxiv.org/abs/2307.03172) phenomenon — facts
buried in the middle of a long context are recalled far less reliably than
facts near the beginning or end, even when the context technically fits
within the model's advertised window.

## Models

| Role | Model |
|---|---|
| Memory Model A | `meta-llama/Llama-3.2-1B-Instruct` |
| Memory Model B | `meta-llama/Llama-3.1-8B-Instruct` |

Both served locally with vLLM (`temperature=0.0`, deterministic decoding) on
a single Modal GPU.

## Usage

```bash
# 1B model
modal run baseline_d_long_context/run_baseline_d.py --model 1b

# 8B model
modal run baseline_d_long_context/run_baseline_d.py --model 8b

# Quick smoke test (5 queries only)
modal run baseline_d_long_context/run_baseline_d.py --model 1b --debug
```

Outputs are written to `baseline_d_results_{1b,8b}.jsonl` in the shared
evaluation schema (`person_id, question, ground_truth_answer, final_answer,
reasoning_type, difficulty, is_temporal`) so they can be scored directly with
the scripts in `evaluation/` (see the top-level README for the full
Phase-4 pipeline).

## Notes on truncation

Some patients in the pilot cohort have visit histories long enough that even
a 128K-token context window is insufficient. Rather than silently working
around this (e.g. by summarizing or chunking, which would turn this into a
different baseline), `run_baseline_d.py` truncates the history to fit and
**explicitly logs which patients were truncated** (`context_was_truncated`
field in the output records) — this failure mode is precisely what the
baseline is meant to characterize, not something to hide.
