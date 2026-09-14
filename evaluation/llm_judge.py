"""
evaluation/llm_judge.py
=========================
Blinded LLM-as-a-judge evaluation, used identically across every baseline in
this repository for clinical correctness scoring AND for extracting the
structured clinical events that ``temporal_metrics.py`` uses to compute
Temporal MAE and Kendall's Tau.

**Blinding.** The judge is shown only the QUESTION, the REFERENCE (ground
truth) answer, and the CANDIDATE (model) answer — never which system or
baseline produced the candidate — so scores cannot be biased by
system-identity priors.

**Two judge outputs per query:**
  1. **Correctness score** (0-3 integer scale) + a boolean "accurate" flag
     (score >= 2), with a one-sentence justification.
  2. **Structured clinical events**, extracted independently and
     *symmetrically* from BOTH the reference and candidate text: for each
     event, ``{"event": str, "date": "YYYY-MM-DD" or null, "duration_days":
     number or null}``. Symmetry matters here — the exact same extraction
     procedure is applied to both texts, so any observed date/ordering
     discrepancy reflects a genuine model error, not an extraction-prompt
     asymmetry.

**Model choice.** Defaults to ``claude-sonnet-4-5-20250929``. If Anthropic
credits are unavailable, this project's actual evaluation runs fell back to
``ministral-8b-latest`` (Mistral) after verifying via real API test calls
that Anthropic/OpenAI accounts had exhausted credit — see the top-level
README's "Judge model substitutions" note. Pass ``--judge_provider mistral``
to use that fallback.

Usage:
    modal run evaluation/llm_judge.py \\
        --results_glob "shared/baseline_g_artifacts/phase3_results_*.jsonl" \\
        --out_path evaluation_results/llm_judge_raw.jsonl
"""
import glob
import json
import os

import modal

app = modal.App("temporal-memo-llm-judge")

judge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("anthropic==0.76.0", "mistralai==1.2.3", "tenacity")
)

SYSTEM_PROMPT = """You are a meticulous, blinded clinical QA evaluator. You will be shown a clinical
question, a REFERENCE answer, and a CANDIDATE answer. You do NOT know and must NOT guess which
system produced the CANDIDATE answer — judge purely on content, comparing it against the
REFERENCE. Respond with ONLY a single valid JSON object, no markdown fences, no commentary.
"""

USER_PROMPT_TEMPLATE = """QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{candidate}

TASKS:
1. Score the CANDIDATE's clinical correctness relative to the REFERENCE on an integer scale:
   0 = incorrect or hallucinated / contradicts reference
   1 = partially correct, missing key facts or containing notable errors
   2 = mostly correct, minor omissions/imprecision but core clinical facts match
   3 = fully correct, equivalent in clinical substance to the reference
   Also give a boolean "candidate_accurate" = true if score >= 2, else false.

2. Independently extract structured clinical EVENTS mentioned in EACH of the two answers
   (REFERENCE and CANDIDATE), applying the IDENTICAL extraction procedure to both texts.
   For each event, extract:
     - "event": short description of the clinical event/fact.
     - "date": absolute date in "YYYY-MM-DD" format if determinable from the text (resolve
       relative dates like "3 days before admission" if the anchor date is known); else null.
       Do NOT invent a date if it cannot be determined.
     - "duration_days": a numeric duration in days if the event describes a duration/interval
       (e.g. "admitted for 4 days" -> 4); otherwise null.
   If an answer has no extractable dated/duration events, return an empty list for it.

Respond with ONLY this JSON schema (no other text):
{{
  "score": <int 0-3>,
  "candidate_accurate": <bool>,
  "reasoning": "<one sentence>",
  "reference_events": [{{"event": "<str>", "date": "<YYYY-MM-DD or null>", "duration_days": <number or null>}}],
  "candidate_events": [{{"event": "<str>", "date": "<YYYY-MM-DD or null>", "duration_days": <number or null>}}]
}}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(text[start:end + 1])


@app.function(
    image=judge_image, timeout=180, max_containers=12,
    secrets=[modal.Secret.from_name("anthropic-secret")],
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=2.0),
)
def judge_one_claude(record: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    question = record.get("question", "")
    reference = record.get("ground_truth_answer", "") or "(no reference answer provided)"
    candidate = record.get("final_answer", "") or "(no candidate answer provided)"
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, reference=reference, candidate=candidate)

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929", max_tokens=1500, temperature=0.0,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_prompt}],
        )
        parsed = _extract_json(resp.content[0].text)
        parsed["_parse_error"] = False
    except Exception as e:
        parsed = {"score": None, "candidate_accurate": None, "reasoning": f"PARSE/API ERROR: {e}",
                  "reference_events": [], "candidate_events": [], "_parse_error": True}

    return {"person_id": record.get("person_id"), "question": question, "ground_truth_answer": reference,
            "final_answer": candidate, "reasoning_type": record.get("reasoning_type"),
            "difficulty": record.get("difficulty"), "judge_result": parsed}


@app.function(
    image=judge_image, timeout=180, max_containers=4,
    secrets=[modal.Secret.from_name("mistral-secret")],
    retries=modal.Retries(max_retries=6, backoff_coefficient=2.0, initial_delay=2.0),
)
def judge_one_mistral(record: dict) -> dict:
    """Fallback judge used when Anthropic/OpenAI credits are exhausted (a
    real situation encountered during this project's Workflow H evaluation
    -- see the top-level README). Model: ``ministral-8b-latest``."""
    from mistralai import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    question = record.get("question", "")
    reference = record.get("ground_truth_answer", "") or "(no reference answer provided)"
    candidate = record.get("final_answer", "") or "(no candidate answer provided)"
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, reference=reference, candidate=candidate)

    try:
        resp = client.chat.complete(
            model="ministral-8b-latest", temperature=0.0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        )
        parsed = _extract_json(resp.choices[0].message.content)
        parsed["_parse_error"] = False
    except Exception as e:
        parsed = {"score": None, "candidate_accurate": None, "reasoning": f"PARSE/API ERROR: {e}",
                  "reference_events": [], "candidate_events": [], "_parse_error": True}

    return {"person_id": record.get("person_id"), "question": question, "ground_truth_answer": reference,
            "final_answer": candidate, "reasoning_type": record.get("reasoning_type"),
            "difficulty": record.get("difficulty"), "judge_result": parsed}


@app.local_entrypoint()
def main(results_glob: str = "shared/baseline_g_artifacts/phase3_results_*.jsonl",
          out_path: str = "evaluation_results/llm_judge_raw.jsonl",
          judge_provider: str = "claude", limit: int = 0):
    records = []
    for fp in sorted(glob.glob(results_glob)):
        with open(fp) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    print(f"[INFO] Loaded {len(records)} records from {results_glob}")
    if limit:
        records = records[:limit]
        print(f"[DEBUG MODE] Limiting to first {limit} records.")

    judge_fn = judge_one_claude if judge_provider == "claude" else judge_one_mistral

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_done = n_errors = 0
    with open(out_path, "w") as f:
        for result in judge_fn.map(records, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                n_errors += 1
                print(f"[ERROR] {result}")
                continue
            f.write(json.dumps(result) + "\n")
            n_done += 1
            if n_done % 20 == 0:
                print(f"[PROGRESS] {n_done}/{len(records)} judged...")
    print(f"[DONE] Judged {n_done}/{len(records)} queries, {n_errors} hard errors -> {out_path}")
