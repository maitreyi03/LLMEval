# LLM Confidence Calibration

Empirical study of verbalized confidence miscalibration in a large language model (LLaMA 3.3-70B-Versatile) across five benchmark datasets.

---

## Project Overview

This project investigates whether a model's self-reported confidence scores correlate with its actual correctness. The model is asked to answer questions and simultaneously report an integer confidence score from 0–100%. We find that the model expresses high confidence (≥80%) on 96–97% of its incorrect answers, making verbalized confidence an unreliable trust signal.

**Datasets:** GSM8K · MMLU · StrategyQA · MedQA (USMLE) · TruthfulQA  
**Model:** `llama-3.3-70b-versatile` via [Groq API](https://console.groq.com) (free tier)  
**Method:** Verbalized confidence — single-pass self-report

---

## Project Structure

```
llm_calibration/
│
├── config.py                  # All paths, API key, and tunable settings
├── requirements.txt           # Python dependencies
│
├── 01_extract_datasets.py     # Download and save the 5 benchmark CSVs
├── 02_run_inference.py        # Query Groq API, save raw results with checkpointing
├── 03_score_answers.py        # Score model answers against ground truth
├── 04_analyse_results.py      # Calibration analysis and all plots
│
├── data/                      # Raw dataset CSVs (created by step 1)
│   ├── gsm8k_first_100.csv
│   ├── mmlu_first_100.csv
│   ├── strategyqa_first_100.csv
│   ├── medqa_usmle_first_100.csv
│   └── truthfulqa_first_100.csv
│
└── results/                   # All output files (created by steps 2–4)
    ├── checkpoint_results.csv
    ├── verbalized_confidence_raw_results.csv
    ├── verbalized_confidence_scored.csv
    ├── confidence_summary_by_domain.csv
    └── plots/
        ├── domain_accuracy.png
        ├── overconfidence.png
        ├── reliability_diagrams.png
        ├── ece_by_domain.png
        ├── conf_bins_accuracy.png
        └── heatmap_domain_conf.png
```

---

## Setup

### 1. Clone or download the project

```bash
git clone "https://github.com/maitreyi03/LLMEval"
cd llm_calibration
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card required)
3. Click **API Keys** → **Create API Key**
4. Copy the key (it starts with `gsk_`)

### 5. Add your API key

**Option A — Edit config.py** (simplest):
```python
GROQ_API_KEY = "gsk_your_key_here"
```

**Option B — Environment variable** (recommended for shared code):
```bash
export GROQ_API_KEY="gsk_your_key_here"    # macOS / Linux
set GROQ_API_KEY=gsk_your_key_here         # Windows
```

---

## Running the Pipeline

Run the four scripts in order. Each step depends on the output of the previous one.

### Step 1 — Extract datasets

Downloads the first 100 questions from each of the five benchmarks and saves them as CSVs in `data/`.

```bash
python 01_extract_datasets.py
```

Expected output:
```
Extracting GSM8K...       Saved 100 rows → data/gsm8k_first_100.csv
Extracting MMLU...        Saved 100 rows → data/mmlu_first_100.csv
Extracting StrategyQA...  Saved 100 rows → data/strategyqa_first_100.csv
Extracting MedQA (USMLE)... Saved 100 rows → data/medqa_usmle_first_100.csv
Extracting TruthfulQA...  Saved 100 rows → data/truthfulqa_first_100.csv
```

---

### Step 2 — Run inference

Queries the Groq API for all 500 questions. Results are checkpointed after each domain, so you can safely interrupt and resume.

**Run all domains (~20 minutes total):**
```bash
python 02_run_inference.py
```

**Run a single domain only:**
```bash
python 02_run_inference.py --domain MMLU
python 02_run_inference.py --domain TruthfulQA
python 02_run_inference.py --domain StrategyQA
python 02_run_inference.py --domain MedQA
python 02_run_inference.py --domain GSM8K
```

> **Note:** The Groq free tier allows 30 requests/minute and 14,400/day. Each domain takes roughly 4 minutes at the default 2-second delay between calls. Running all five domains takes about 20 minutes.

> **Resuming after interruption:** Re-running `02_run_inference.py` will skip any domains already saved in `results/checkpoint_results.csv` and pick up where it left off.

---

### Step 3 — Score answers

Applies domain-specific correctness logic to the raw model outputs and adds an `LLM_right` column (True / False / Unknown).

```bash
python 03_score_answers.py
```

Expected output:
```
Scoring summary:
  GSM8K          Correct:  96/100  (96.0%)  |  Wrong:   4  Unknown: 0
  MedQA          Correct:  89/100  (89.0%)  |  Wrong:  11  Unknown: 0
  StrategyQA     Correct:  76/100  (76.0%)  |  Wrong:  24  Unknown: 0
  TruthfulQA     Correct:  73/100  (73.0%)  |  Wrong:  27  Unknown: 0
  MMLU           Correct:  60/100  (60.0%)  |  Wrong:  40  Unknown: 0

Overall accuracy: 394/500 = 78.8%
```

---

### Step 4 — Analyse results

Runs the full calibration analysis: domain accuracy tables, overconfidence breakdown, reliability diagrams, ECE scores, confidence bin charts, and a domain × bin heatmap. All figures are saved to `results/plots/`.

```bash
python 04_analyse_results.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/*.csv` | Raw dataset CSVs (100 questions each) |
| `results/checkpoint_results.csv` | Inference results saved after each domain |
| `results/verbalized_confidence_raw_results.csv` | Final merged raw inference output |
| `results/verbalized_confidence_scored.csv` | Scored results with `LLM_right` column |
| `results/confidence_summary_by_domain.csv` | Per-domain accuracy and calibration metrics |
| `results/plots/domain_accuracy.png` | Accuracy bar chart + confidence scatter plot |
| `results/plots/overconfidence.png` | Confidence distribution (correct vs wrong) + overconfidence rate |
| `results/plots/reliability_diagrams.png` | Calibration curves per domain |
| `results/plots/ece_by_domain.png` | Expected Calibration Error bar chart |
| `results/plots/conf_bins_accuracy.png` | Accuracy per confidence bin (overall) |
| `results/plots/heatmap_domain_conf.png` | Accuracy heatmap: domain × confidence bin |

---

## Configuration

All settings are in `config.py`. Common things to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_API_KEY` | `"YOUR_GROQ_API_KEY_HERE"` | Your Groq API key |
| `MODEL` | `"llama-3.3-70b-versatile"` | Groq model to use |
| `SLEEP_BETWEEN_CALLS` | `2.0` | Seconds between API calls (free tier: 30 req/min) |
| `HIGH_CONF_THRESHOLD` | `0.80` | Threshold for "high confidence" in overconfidence analysis |
| `N_CALIBRATION_BINS` | `10` | Number of bins for ECE computation |

---

## Key Results

| Domain | Accuracy | Cal. Gap | ECE | Note |
|--------|----------|----------|-----|------|
| GSM8K | 96% | 0.04 | 0.000 | Well-calibrated — high confidence is warranted |
| MedQA | 89% | 0.02 | 0.043 | Minor overconfidence, acceptable |
| StrategyQA | 76% | 0.18 | 0.115 | Consistent overconfidence on reasoning tasks |
| TruthfulQA | 73% | 0.21 | 0.124 | Absorbed misinformation expressed confidently |
| MMLU | 60% | 0.39 | 0.000* | Largest calibration gap; ECE is an artifact |

*MMLU ECE of 0.000 is a measurement artifact — all scores fall in one bin (100%), not evidence of good calibration.

**Headline finding:** 96–97% of all incorrect answers are assigned confidence scores ≥ 80%. The confidence distributions for correct and incorrect responses are nearly indistinguishable, meaning verbalized confidence carries minimal diagnostic signal.

---

## Requirements

- Python 3.10+
- Internet access (for Groq API and HuggingFace dataset downloads)
- Free Groq API key
