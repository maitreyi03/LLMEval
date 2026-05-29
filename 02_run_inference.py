"""
02_run_inference.py
-------------------
Queries the Groq API (LLaMA 3.3-70B) on all five benchmark datasets using
a verbalized confidence prompting strategy. The model is asked to self-report
an integer confidence score (0-100) alongside every answer.

Features:
  - Auto-detects question / answer / option columns in each CSV
  - Uses MCQ prompts for MMLU and MedQA; open-ended prompts for the rest
  - Saves a checkpoint after each domain — safe to interrupt and resume
  - Final results written to RESULTS_DIR/verbalized_confidence_raw_results.csv

Usage:
  python 02_run_inference.py                    # run all domains
  python 02_run_inference.py --domain MMLU      # run one domain only
  python 02_run_inference.py --domain ALL       # explicit all (default)

Requirements:
  pip install groq pandas tqdm
"""

import argparse
import os
import re
import time

import pandas as pd
from groq import Groq
from tqdm import tqdm

from config import (
    GROQ_API_KEY,
    MODEL,
    DATASET_PATHS,
    CHECKPOINT_PATH,
    RAW_RESULTS_PATH,
    SLEEP_BETWEEN_CALLS,
    MAX_RETRIES,
    TEMPERATURE,
    MAX_TOKENS,
    MCQ_DOMAINS,
)


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_OPEN = (
    "You are a knowledgeable assistant. "
    "Always respond in exactly this format:\n\n"
    "Answer: <your answer>\n"
    "Confidence: <number>%\n\n"
    "Confidence must be an integer 0-100 reflecting how likely you are "
    "to be correct. Do not add any text after the confidence line."
)

SYSTEM_PROMPT_MCQ = (
    "You are a knowledgeable assistant. "
    "You will be given a multiple choice question with options A, B, C, and D. "
    "Always respond in exactly this format:\n\n"
    "Answer: <A, B, C, or D>\n"
    "Confidence: <number>%\n\n"
    "Only output the letter of your chosen answer, nothing else. "
    "Confidence must be an integer 0-100 reflecting how likely you are "
    "to be correct. Do not add any text after the confidence line."
)


# ── Column-detection helpers ──────────────────────────────────────────────────

QUESTION_KEYWORDS = ["question", "Question", "prompt", "input", "problem", "query"]
ANSWER_KEYWORDS   = [
    "answer", "Answer", "label", "solution", "ground_truth",
    "best_answer", "correct_answer", "target", "output",
]
OPTION_KEYWORDS = {
    "A": ["option_a", "option a", "opa", "choice_a", "a)", "optiona", "choices"],
    "B": ["option_b", "option b", "opb", "choice_b", "b)", "optionb"],
    "C": ["option_c", "option c", "opc", "choice_c", "c)", "optionc"],
    "D": ["option_d", "option d", "opd", "choice_d", "d)", "optiond"],
}


def find_column(df_cols: list, keywords: list) -> str | None:
    """Return the first column name that contains any keyword (case-insensitive)."""
    for kw in keywords:
        for col in df_cols:
            if kw.lower() in col.lower():
                return col
    return None


def find_option_columns(df_cols: list) -> dict:
    """Return a dict mapping letter → column name for any A/B/C/D columns found."""
    found = {}
    for letter, kws in OPTION_KEYWORDS.items():
        col = find_column(df_cols, kws)
        if col:
            found[letter] = col
    return found


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt_open(question: str) -> str:
    return (
        "Answer the following question and then state your confidence "
        "in your answer as a percentage from 0-100%.\n\n"
        f"Question: {question}"
    )


def build_prompt_mcq(question: str, options: dict) -> str:
    opts_text = "\n".join([f"  {letter}. {text}" for letter, text in options.items()])
    return (
        "Answer the following multiple choice question by selecting A, B, C, or D. "
        "Then state your confidence as a percentage from 0-100%.\n\n"
        f"Question: {question}\n\nOptions:\n{opts_text}"
    )


# ── Response parsers ──────────────────────────────────────────────────────────

def extract_confidence(text: str) -> float | None:
    """Parse the integer confidence score from the model's raw response."""
    match = re.search(r"[Cc]onfidence[:\s]+([0-9]{1,3})\s*%", text)
    if match:
        return min(max(float(match.group(1)), 0.0), 100.0)
    match = re.search(r"([0-9]{1,3})\s*%\s*$", text.strip())
    if match:
        return min(max(float(match.group(1)), 0.0), 100.0)
    return None


def extract_answer(text: str) -> str:
    """Extract the answer portion by stripping the confidence line."""
    lines = [
        line for line in text.strip().split("\n")
        if not re.search(r"[Cc]onfidence[:\s]+[0-9]", line)
    ]
    return " ".join(lines).replace("Answer:", "").strip()


# ── Groq query ────────────────────────────────────────────────────────────────

def query_groq(
    client: Groq,
    question: str,
    domain: str,
    row: dict | None = None,
) -> dict:
    """
    Call the Groq API with the appropriate prompt format for the domain.
    Returns a dict with: raw_response, model_answer, confidence_pct, prompt_type.
    """
    is_mcq = domain in MCQ_DOMAINS

    if is_mcq and row is not None:
        options = {
            letter: str(row[f"opt_{letter}"])
            for letter in ["A", "B", "C", "D"]
            if f"opt_{letter}" in row and pd.notna(row[f"opt_{letter}"])
        }
        prompt        = build_prompt_mcq(question, options)
        system_prompt = SYSTEM_PROMPT_MCQ
    else:
        prompt        = build_prompt_open(question)
        system_prompt = SYSTEM_PROMPT_OPEN

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = response.choices[0].message.content
            return {
                "raw_response":   raw,
                "model_answer":   extract_answer(raw),
                "confidence_pct": extract_confidence(raw),
                "prompt_type":    "mcq" if is_mcq else "open",
            }
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"  Rate limit — waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            elif "401" in err or "auth" in err.lower():
                print("  Auth error — check GROQ_API_KEY in config.py")
                break
            else:
                print(f"  Attempt {attempt + 1} failed: {err[:100]}")
                time.sleep(5 * (attempt + 1))

    return {"raw_response": None, "model_answer": None,
            "confidence_pct": None, "prompt_type": None}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_datasets() -> tuple[dict, dict]:
    """
    Load all CSVs, auto-detect columns, rename to standard names.
    Returns (datasets, option_cols) where:
      datasets    = {domain_name: DataFrame}
      option_cols = {domain_name: {"A": col, "B": col, ...}}
    """
    datasets    = {}
    option_cols = {}

    for name, path in DATASET_PATHS.items():
        if not os.path.exists(path):
            print(f"  Missing: {name} — expected at '{path}'. Run 01_extract_datasets.py first.")
            continue

        df    = pd.read_csv(path)
        q_col = find_column(df.columns.tolist(), QUESTION_KEYWORDS)
        a_col = find_column(df.columns.tolist(), ANSWER_KEYWORDS)
        opts  = find_option_columns(df.columns.tolist())

        if q_col is None:
            print(f"  Cannot detect question column for {name} — skipping.")
            continue

        df = df.rename(columns={q_col: "question"})
        df["ground_truth"] = df[a_col] if (a_col and a_col != q_col) else ""
        df["domain"]       = name

        for letter, col in opts.items():
            df = df.rename(columns={col: f"opt_{letter}"})

        keep = (["question", "ground_truth", "domain"]
                + [f"opt_{l}" for l in opts.keys()])
        datasets[name]    = df[keep].dropna(subset=["question"]).head(100)
        option_cols[name] = {l: f"opt_{l}" for l in opts.keys()}
        print(f"  Loaded {name}: {len(datasets[name])} rows")

    return datasets, option_cols


# ── Main ──────────────────────────────────────────────────────────────────────

def run_inference(domains_to_run: list) -> pd.DataFrame:
    """Run inference for the specified domains and return the full results DataFrame."""
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        raise ValueError(
            "Set your Groq API key in config.py or via the GROQ_API_KEY environment variable."
        )

    client = Groq(api_key=GROQ_API_KEY)

    print("Loading datasets...")
    datasets, _ = load_datasets()

    if not datasets:
        raise RuntimeError("No datasets loaded. Run 01_extract_datasets.py first.")

    # Load checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint_df     = pd.read_csv(CHECKPOINT_PATH)
        completed_domains = set(checkpoint_df["domain"].unique())
        all_results       = checkpoint_df.to_dict("records")
        print(f"Resuming from checkpoint — completed: {sorted(completed_domains)}")
    else:
        all_results       = []
        completed_domains = set()

    # Run selected domains
    for domain_name in domains_to_run:
        if domain_name not in datasets:
            print(f"  Skipping '{domain_name}' — not in loaded datasets.")
            continue
        if domain_name in completed_domains:
            print(f"  Skipping '{domain_name}' — already in checkpoint.")
            continue

        df = datasets[domain_name]
        print(f"\nRunning {domain_name} ({len(df)} questions)...")
        domain_results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=domain_name):
            result = query_groq(
                client=client,
                question=str(row["question"]),
                domain=domain_name,
                row=row.to_dict(),
            )
            domain_results.append({
                "domain":         domain_name,
                "prompt_type":    result["prompt_type"],
                "question":       row["question"],
                "ground_truth":   row["ground_truth"],
                "model_answer":   result["model_answer"],
                "confidence_pct": result["confidence_pct"],
                "raw_response":   result["raw_response"],
            })
            time.sleep(SLEEP_BETWEEN_CALLS)

        all_results.extend(domain_results)

        # Checkpoint after each domain
        pd.DataFrame(all_results).to_csv(CHECKPOINT_PATH, index=False)
        parsed = sum(1 for r in domain_results if r["confidence_pct"] is not None)
        print(f"  {domain_name} done — {parsed}/{len(domain_results)} confidence scores parsed.")
        print(f"  Checkpoint saved: {CHECKPOINT_PATH}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RAW_RESULTS_PATH, index=False)
    print(f"\nFinal results saved: {RAW_RESULTS_PATH}  ({len(results_df)} rows)")
    print("Next step: python 03_score_answers.py")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Groq inference for LLM calibration study.")
    parser.add_argument(
        "--domain",
        default="ALL",
        help="Domain to run: ALL | TruthfulQA | StrategyQA | MMLU | MedQA | GSM8K",
    )
    args = parser.parse_args()

    all_domains = list(DATASET_PATHS.keys())
    if args.domain.upper() == "ALL":
        domains_to_run = all_domains
    elif args.domain in all_domains:
        domains_to_run = [args.domain]
    else:
        print(f"Unknown domain '{args.domain}'. Options: ALL, {', '.join(all_domains)}")
        return

    print(f"Domains to run: {domains_to_run}")
    print(f"Estimated time: ~{len(domains_to_run) * 4} minutes\n")
    run_inference(domains_to_run)


if __name__ == "__main__":
    main()
