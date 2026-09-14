"""
evaluation/text_metrics.py
============================
Pure-Python, dependency-light, fully deterministic text-overlap metrics used
identically across EVERY baseline in this repository for apples-to-apples
comparison. No external ML packages needed here (SQuAD-style normalization +
bag-of-words token overlap + ROUGE-L via word-level LCS).

These are intentionally simple/classic metrics — they are complemented by
``bertscore_eval.py`` (semantic similarity) and ``llm_judge.py`` (clinical
correctness + structured event extraction) for a fuller picture; no single
metric here is claimed to fully capture clinical answer quality on its own.
"""
import re
import string
from collections import Counter
from typing import Tuple


def normalize_text(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation, drop English
    articles, collapse whitespace. Applied identically to predictions and
    ground truth so neither side gets an unfair advantage from surface form."""
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def get_tokens(s: str):
    if not s:
        return []
    return normalize_text(s).split()


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 iff the normalized strings are identical. In practice this is
    almost always 0 for free-text clinical answers (see the top-level README
    "Why Exact Match is near-zero everywhere" note) — included for
    completeness / consistency with prior QA-evaluation conventions."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def precision_recall_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Bag-of-words token overlap precision/recall/F1 (SQuAD-style token-F1)."""
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _lcs_length(a, b) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F-measure (beta=1) using word-level LCS, standard ROUGE-L
    recall/precision/F convention."""
    pred_tokens, gt_tokens = get_tokens(prediction), get_tokens(ground_truth)
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    lcs = _lcs_length(pred_tokens, gt_tokens)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(pred_tokens), lcs / len(gt_tokens)
    return (2 * prec * rec) / (rec + prec) if (rec + prec) > 0 else 0.0


def compute_all_text_metrics(prediction: str, ground_truth: str) -> dict:
    p, r, f1 = precision_recall_f1(prediction, ground_truth)
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "precision": p,
        "recall": r,
        "token_f1": f1,
        "rouge_l": rouge_l(prediction, ground_truth),
    }


if __name__ == "__main__":
    pred = "The patient received paracetamol and ibuprofen with reasonable effect."
    gt = ("Yes. She was given paracetamol and ibuprofen with reasonable effect, "
          "could mobilise with minimal discomfort.")
    print(compute_all_text_metrics(pred, gt))
