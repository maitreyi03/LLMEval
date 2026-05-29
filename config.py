"""
config.py
---------
Central configuration for the LLM Confidence Calibration project.
Edit GROQ_API_KEY and DATA_DIR before running any other script.
"""

import os

# ── API ────────────────────────────────────────────────────────────────────────
# Paste your Groq API key here, or set the GROQ_API_KEY environment variable.
# Get a free key at https://console.groq.com (no credit card required).
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

MODEL: str = "llama-3.3-70b-versatile"

# ── Paths ──────────────────────────────────────────────────────────────────────
# Root directory that will hold raw CSVs and result files.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

DATA_DIR: str = os.path.join(BASE_DIR, "data")       # raw dataset CSVs go here
RESULTS_DIR: str = os.path.join(BASE_DIR, "results") # inference output CSVs
PLOTS_DIR: str = os.path.join(RESULTS_DIR, "plots")  # saved figures

# ── Dataset file paths ─────────────────────────────────────────────────────────
DATASET_PATHS: dict = {
    "TruthfulQA": os.path.join(DATA_DIR, "truthfulqa_first_100.csv"),
    "StrategyQA":  os.path.join(DATA_DIR, "strategyqa_first_100.csv"),
    "MMLU":        os.path.join(DATA_DIR, "mmlu_first_100.csv"),
    "MedQA":       os.path.join(DATA_DIR, "medqa_usmle_first_100.csv"),
    "GSM8K":       os.path.join(DATA_DIR, "gsm8k_first_100.csv"),
}

# ── Result file paths ──────────────────────────────────────────────────────────
CHECKPOINT_PATH: str = os.path.join(RESULTS_DIR, "checkpoint_results.csv")
RAW_RESULTS_PATH: str = os.path.join(RESULTS_DIR, "verbalized_confidence_raw_results.csv")
SCORED_RESULTS_PATH: str = os.path.join(RESULTS_DIR, "verbalized_confidence_scored.csv")
CONFIDENCE_SUMMARY_PATH: str = os.path.join(RESULTS_DIR, "confidence_summary_by_domain.csv")

# ── Inference settings ─────────────────────────────────────────────────────────
SLEEP_BETWEEN_CALLS: float = 2.0   # seconds — keeps within Groq free-tier (30 req/min)
MAX_RETRIES: int = 2
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 300

# ── Analysis settings ──────────────────────────────────────────────────────────
HIGH_CONF_THRESHOLD: float = 0.80  # answers at or above this are "high confidence"
N_CALIBRATION_BINS: int = 10

# Datasets that use multiple-choice format (A/B/C/D)
MCQ_DOMAINS: set = {"MMLU", "MedQA"}

# ── Ensure output directories exist ───────────────────────────────────────────
for _dir in (DATA_DIR, RESULTS_DIR, PLOTS_DIR):
    os.makedirs(_dir, exist_ok=True)
