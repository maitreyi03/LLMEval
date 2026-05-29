"""
01_extract_datasets.py
----------------------
Downloads the first 100 questions from each of the five benchmark datasets
and saves them as CSVs in the configured DATA_DIR.

Datasets:
  - GSM8K       (math reasoning)
  - MMLU        (academic, 57 subjects)
  - StrategyQA  (commonsense reasoning)
  - MedQA USMLE (medical, 4-option MCQ)
  - TruthfulQA  (factual / misconception)

Usage:
  python 01_extract_datasets.py
"""

import json
import pandas as pd
import requests
from datasets import load_dataset

from config import DATA_DIR, DATASET_PATHS


# ── 1. GSM8K ──────────────────────────────────────────────────────────────────

def extract_gsm8k(n: int = 100) -> pd.DataFrame:
    """Download first n rows from the GSM8K train split via GitHub raw URL."""
    print("Extracting GSM8K...")
    url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/train.jsonl"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    rows = []
    for i, line in enumerate(response.text.splitlines()[:n]):
        item = json.loads(line)
        rows.append({
            "id":           i + 1,
            "question":     item["question"],
            "answer":       item["answer"],
            "final_answer": item["answer"].split("####")[-1].strip(),
        })

    df = pd.DataFrame(rows)
    path = DATASET_PATHS["GSM8K"]
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ── 2. MMLU ───────────────────────────────────────────────────────────────────

def extract_mmlu(n: int = 100) -> pd.DataFrame:
    """Load first n rows from the MMLU test split (all subjects)."""
    print("Extracting MMLU...")
    dataset = load_dataset("cais/mmlu", "all")
    rows = []
    for i, item in enumerate(dataset["test"].select(range(n))):
        choices = item["choices"]
        answer_index = item["answer"]
        rows.append({
            "id":            i + 1,
            "question":      item["question"],
            "choice_A":      choices[0],
            "choice_B":      choices[1],
            "choice_C":      choices[2],
            "choice_D":      choices[3],
            "answer_index":  answer_index,
            "answer_letter": ["A", "B", "C", "D"][answer_index],
            "answer_text":   choices[answer_index],
        })

    df = pd.DataFrame(rows)
    path = DATASET_PATHS["MMLU"]
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ── 3. StrategyQA ─────────────────────────────────────────────────────────────

def extract_strategyqa(n: int = 100) -> pd.DataFrame:
    """Download first n rows from the StrategyQA training split."""
    print("Extracting StrategyQA...")
    url = (
        "https://huggingface.co/datasets/voidful/StrategyQA/"
        "resolve/main/strategyqa_train.json"
    )
    data = requests.get(url, timeout=30).json()
    rows = []
    for i, item in enumerate(data[:n]):
        answer_bool = item["answer"]
        rows.append({
            "id":           i + 1,
            "question":     item["question"],
            "answer_bool":  answer_bool,
            "answer_text":  "yes" if answer_bool else "no",
            "decomposition": " | ".join(item.get("decomposition", [])),
        })

    df = pd.DataFrame(rows)
    path = DATASET_PATHS["StrategyQA"]
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ── 4. MedQA USMLE ────────────────────────────────────────────────────────────

def extract_medqa(n: int = 100) -> pd.DataFrame:
    """Load first n rows from the MedQA USMLE 4-option training split."""
    print("Extracting MedQA (USMLE)...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    rows = []
    for i, item in enumerate(dataset["train"].select(range(n))):
        options = item["options"]
        rows.append({
            "id":            i + 1,
            "question":      item["question"],
            "choice_A":      options.get("A", ""),
            "choice_B":      options.get("B", ""),
            "choice_C":      options.get("C", ""),
            "choice_D":      options.get("D", ""),
            "answer_letter": item["answer_idx"],
            "answer_text":   item["answer"],
            "meta_info":     item.get("meta_info", ""),
        })

    df = pd.DataFrame(rows)
    path = DATASET_PATHS["MedQA"]
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ── 5. TruthfulQA ─────────────────────────────────────────────────────────────

def extract_truthfulqa(n: int = 100) -> pd.DataFrame:
    """Load first n rows from the TruthfulQA training split."""
    print("Extracting TruthfulQA...")
    dataset = load_dataset("domenicrosati/TruthfulQA")
    rows = []
    for i, item in enumerate(dataset["train"].select(range(n))):
        rows.append({
            "id":                i + 1,
            "type":              item["Type"],
            "category":          item["Category"],
            "question":          item["Question"],
            "best_answer":       item["Best Answer"],
            "correct_answers":   item["Correct Answers"],
            "incorrect_answers": item["Incorrect Answers"],
            "source":            item["Source"],
        })

    df = pd.DataFrame(rows)
    path = DATASET_PATHS["TruthfulQA"]
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Saving all datasets to: {DATA_DIR}\n")

    extract_gsm8k()
    extract_mmlu()
    extract_strategyqa()
    extract_medqa()
    extract_truthfulqa()

    print("\nAll datasets extracted successfully.")
    print("Next step: python 02_run_inference.py")


if __name__ == "__main__":
    main()
