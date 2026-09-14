# Evaluation Utilities

Every baseline (D through H) writes its outputs to the same
`phase3_results_{person_id}.jsonl` schema:

```json
{
  "person_id": 8855233,
  "question": "...",
  "ground_truth_answer": "...",
  "final_answer": "...",
  "reasoning_type": "...",
  "difficulty": "...",
  "is_temporal": true
}
```

...which lets this evaluation pipeline apply **identically** to every
baseline for a fair, apples-to-apples comparison.

## Pipeline

```
phase3_results_*.jsonl
        │
        ├──► text_metrics.py        (pure Python, deterministic)
        │      -> Exact Match, Precision, Recall, Token-F1, ROUGE-L
        │
        ├──► bertscore_eval.py      (Modal GPU, roberta-large)
        │      -> BERTScore-F1
        │
        ├──► llm_judge.py           (Modal, blinded Claude/Mistral judge)
        │      -> correctness score (0-3), accuracy %,
        │         + structured (event, date, duration_days) extraction
        │         from BOTH reference and candidate answers
        │
        └──► temporal_metrics.py    (pure Python, local)
               -> event matching (token-F1 >= 0.15)
               -> Temporal MAE (days)
               -> Kendall's Tau (rank correlation)
```

## Usage

```bash
# 1. Deterministic text metrics + BERTScore (Modal GPU)
modal run evaluation/bertscore_eval.py \
  --results-glob "shared/baseline_g_artifacts/phase3_results_*.jsonl" \
  --out-dir evaluation_results

# 2. Blinded LLM judge (Modal)
modal run evaluation/llm_judge.py \
  --results-glob "shared/baseline_g_artifacts/phase3_results_*.jsonl" \
  --out-path evaluation_results/llm_judge_raw.jsonl \
  --judge-provider claude   # or "mistral" as a fallback

# 3. Temporal MAE + Kendall's Tau + final report
python evaluation/temporal_metrics.py \
  --judge_results evaluation_results/llm_judge_raw.jsonl \
  --local_metrics evaluation_results/local_metrics_summary.json \
  --out_path evaluation_results/phase4_final_metrics.json
```

## The 10 core metrics

| # | Metric | Range | Better | Computed by |
|---|---|---|---|---|
| 1 | Exact Match | [0,1] | ↑ | `text_metrics.py` |
| 2 | Precision (token overlap) | [0,1] | ↑ | `text_metrics.py` |
| 3 | Recall (token overlap) | [0,1] | ↑ | `text_metrics.py` |
| 4 | Token-F1 | [0,1] | ↑ | `text_metrics.py` |
| 5 | ROUGE-L | [0,1] | ↑ | `text_metrics.py` |
| 6 | BERTScore-F1 | [0,1] | ↑ | `bertscore_eval.py` |
| 7 | Avg. Judge Score | [0,3] | ↑ | `llm_judge.py` |
| 8 | LLM-Judge Accuracy % | [0,100] | ↑ | `llm_judge.py` |
| 9 | **Temporal MAE (days)** | [0,∞) | ↓ | `temporal_metrics.py` |
| 10 | **Kendall's Tau** | [-1,1] | ↑ | `temporal_metrics.py` |

Metrics 9 and 10 are the project's headline temporal-reasoning metrics and
measure genuinely different failure modes — see the top-level README for
the full discussion of how they diverge across architectures.

## Judge model substitutions (disclosed, not hidden)

The default judge is `claude-sonnet-4-5-20250929`. During this project's
actual Workflow H (Baseline H) evaluation, both the Anthropic and OpenAI
accounts had genuinely exhausted API credit (verified via real API test
calls returning HTTP 400/429), so the judge was switched to
`ministral-8b-latest` (Mistral) for that run only, after validating it on a
smoke test. `llm_judge.py` supports both providers via `--judge_provider`;
always report which judge model produced a given set of numbers.
