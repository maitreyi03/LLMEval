"""
03_score_answers.py
-------------------
Scores each model answer against the ground truth using domain-specific
comparison logic. Writes a new CSV with an added 'LLM_right' column
(True / False / Unknown).

Domain scoring strategies:
  StrategyQA  — yes/no string match
  MMLU        — letter match (ground truth is 0-3 index, model returns A-D)
  MedQA       — direct letter-to-letter match
  GSM8K       — numeric extraction and comparison
  TruthfulQA  — normalised token-overlap with uncertainty-phrase handling

Usage:
  python 03_score_answers.py
"""

import re
from decimal import Decimal, InvalidOperation

import pandas as pd

from config import RAW_RESULTS_PATH, SCORED_RESULTS_PATH


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_strategyqa(ground_truth: str, model_answer: str) -> bool | str:
    """
    Ground truth is a boolean ("True"/"False").
    Model answer should begin with "yes" or "no".
    """
    first_token = str(model_answer).lower().split(",")[0].strip()
    if first_token == "yes":
        predicted = "True"
    elif first_token == "no":
        predicted = "False"
    else:
        return "Unknown"
    return predicted == str(ground_truth)


def score_mmlu(ground_truth: str, model_answer: str) -> bool:
    """
    Ground truth is an index 0-3; model answer is a letter A-D.
    """
    index_to_letter = {"0": "A", "1": "B", "2": "C", "3": "D"}
    expected_letter = index_to_letter.get(str(ground_truth), "")
    return expected_letter == str(model_answer).strip().upper()


def score_medqa(ground_truth: str, model_answer: str) -> bool:
    """
    Both ground truth and model answer are single letters (A-D).
    """
    return str(ground_truth).strip().upper() == str(model_answer).strip().upper()


def score_gsm8k(ground_truth: str, model_answer: str) -> bool | None:
    """
    Ground truth may contain '#### <number>'; model answer is free-form text.
    Extracts the last number from each and compares after normalisation.
    """
    # Parse ground truth
    gt_clean = str(ground_truth).lower().split("####")[-1].strip()
    gt_clean = gt_clean.replace("$", "").replace(",", "")

    # Parse model answer — find all numeric tokens, take the last one
    numbers = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", str(model_answer).lower())
    if not numbers:
        return None

    pred_raw = numbers[-1].replace("$", "").replace(",", "")

    # Normalise: drop trailing .00 but keep real decimals
    try:
        num = Decimal(pred_raw)
        if num == num.to_integral():
            pred_clean = str(int(num))
        else:
            pred_clean = format(num.normalize(), "f")
    except InvalidOperation:
        pred_clean = pred_raw

    return pred_clean.strip() == gt_clean.strip()


def _normalize_text(text: str) -> str:
    """Lowercase, expand contractions, remove punctuation, collapse whitespace."""
    text = str(text).lower()
    contractions = {
        "doesn't": "does not", "don't": "do not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    }
    for short, full in contractions.items():
        text = text.replace(short, full)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_UNCERTAINTY_PHRASES = [
    "does not specify", "not specify", "not specifically state",
    "unclear", "unknown", "no recorded composer",
    "does not have a recorded composer", "did not compose",
    "not harmful", "pass through",
]

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
    "or", "in", "on", "for", "with", "by", "from", "that", "this",
    "it", "as", "can", "be", "when", "viewed", "your",
}


def score_truthfulqa(ground_truth: str, model_answer: str) -> bool:
    """
    Flexible matching for open-ended factual questions:
      1. Direct containment
      2. Numeric overlap (for date/number answers)
      3. Uncertainty-phrase detection
      4. Token overlap ratio (threshold: 0.55)
    """
    gt  = _normalize_text(ground_truth)
    ans = _normalize_text(model_answer)

    if not gt or not ans:
        return False

    # 1. Direct containment
    if gt in ans or ans in gt:
        return True

    # 2. Numeric match
    gt_nums  = re.findall(r"\d+", gt)
    ans_nums = re.findall(r"\d+", ans)
    if gt_nums:
        return all(num in ans_nums for num in gt_nums)

    # 3. Uncertainty-phrase agreement
    gt_has_uncertainty = any(p in gt for p in _UNCERTAINTY_PHRASES)
    if gt_has_uncertainty:
        if not any(p in ans for p in _UNCERTAINTY_PHRASES):
            return False

    # 4. Token overlap
    gt_words  = set(gt.split()) - _STOPWORDS
    ans_words = set(ans.split()) - _STOPWORDS
    if not gt_words:
        return False
    overlap_ratio = len(gt_words & ans_words) / len(gt_words)
    return overlap_ratio >= 0.55


# ── Dispatch table ────────────────────────────────────────────────────────────

_SCORERS = {
    "StrategyQA": score_strategyqa,
    "MMLU":       score_mmlu,
    "MedQA":      score_medqa,
    "GSM8K":      score_gsm8k,
    "TruthfulQA": score_truthfulqa,
}


def score_row(row: pd.Series) -> bool | str | None:
    """Dispatch to the correct scorer based on domain."""
    scorer = _SCORERS.get(row["domain"])
    if scorer is None:
        return "Unknown"
    try:
        return scorer(row["ground_truth"], row["model_answer"])
    except Exception:
        return "Unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading raw results: {RAW_RESULTS_PATH}")
    df = pd.read_csv(RAW_RESULTS_PATH)
    print(f"  {len(df)} rows, domains: {sorted(df['domain'].unique())}")

    print("Scoring answers...")
    df["LLM_right"] = df.apply(score_row, axis=1)

    # Summary
    print("\nScoring summary:")
    for domain, grp in df.groupby("domain"):
        counts = grp["LLM_right"].value_counts(dropna=False)
        total  = len(grp)
        n_true = counts.get(True, 0)
        print(f"  {domain:<14} Correct: {n_true:>3}/{total}  "
              f"({n_true / total:.1%})  |  "
              f"Wrong: {counts.get(False, 0):>3}  "
              f"Unknown: {counts.get('Unknown', 0)}")

    total_correct = (df["LLM_right"] == True).sum()
    total_known   = df["LLM_right"].isin([True, False]).sum()
    print(f"\nOverall accuracy: {total_correct}/{total_known} = {total_correct / total_known:.1%}")

    df.to_csv(SCORED_RESULTS_PATH, index=False)
    print(f"\nScored results saved: {SCORED_RESULTS_PATH}")
    print("Next step: python 04_analyse_results.py")


if __name__ == "__main__":
    main()
