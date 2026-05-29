"""
04_analyse_results.py
---------------------
Full calibration analysis of the scored inference results. Produces:
  - Domain-level accuracy and calibration gap table
  - Overconfidence analysis (% of wrong answers at high confidence)
  - Reliability diagrams (calibration curves) per domain
  - Expected Calibration Error (ECE) per domain
  - Confidence bin vs accuracy chart
  - Domain × confidence-bin accuracy heatmap
  - Final summary CSV

All figures are saved to PLOTS_DIR.

Usage:
  python 04_analyse_results.py
"""

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from config import (
    SCORED_RESULTS_PATH,
    CONFIDENCE_SUMMARY_PATH,
    PLOTS_DIR,
    HIGH_CONF_THRESHOLD,
    N_CALIBRATION_BINS,
)

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11
sns.set_theme(style="whitegrid", palette="muted")


# ── Data loading & cleaning ───────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load scored CSV, normalise LLM_right, and add a fractional confidence column."""
    df = pd.read_csv(path)

    def _norm(val):
        s = str(val).strip().upper()
        if s == "TRUE":  return True
        if s == "FALSE": return False
        return None

    df["LLM_right_norm"] = df["LLM_right"].apply(_norm)
    df = df.dropna(subset=["domain"]).reset_index(drop=True)
    df["confidence"] = df["confidence_pct"] / 100.0
    return df


# ── Domain-level stats ────────────────────────────────────────────────────────

def compute_domain_stats(df_known: pd.DataFrame) -> pd.DataFrame:
    """Compute per-domain accuracy, mean confidence, and calibration gap."""
    stats = (
        df_known
        .groupby("domain")
        .agg(
            total     =("correct", "count"),
            correct   =("correct", "sum"),
            accuracy  =("correct", "mean"),
            mean_conf =("confidence", "mean"),
            std_conf  =("confidence", "std"),
        )
        .reset_index()
    )
    stats["wrong"]           = stats["total"] - stats["correct"]
    stats["calibration_gap"] = stats["mean_conf"] - stats["accuracy"]
    return stats.sort_values("accuracy", ascending=False)


def print_domain_stats(stats: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  DOMAIN-LEVEL ACCURACY & CALIBRATION")
    print("=" * 65)
    print(
        stats[["domain", "total", "correct", "wrong", "accuracy", "mean_conf", "calibration_gap"]]
        .to_string(index=False, float_format="{:.3f}".format)
    )


# ── Plot 1: Domain accuracy bar + scatter ─────────────────────────────────────

def plot_domain_accuracy(stats: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = [
        "#2ecc71" if a >= 0.7 else "#e67e22" if a >= 0.5 else "#e74c3c"
        for a in stats["accuracy"]
    ]
    axes[0].barh(stats["domain"], stats["accuracy"], color=colors, edgecolor="white")
    axes[0].axvline(0.5, color="gray", linestyle="--", lw=1.2, label="50% baseline")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Domain Accuracy")
    axes[0].set_xlim(0, 1.05)
    for bar, val in zip(axes[0].patches, stats["accuracy"]):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=9)
    axes[0].legend()

    scatter = axes[1].scatter(
        stats["mean_conf"], stats["accuracy"],
        s=stats["total"] * 4,
        c=stats["calibration_gap"],
        cmap="RdYlGn_r", vmin=-0.3, vmax=0.3,
        edgecolors="k", lw=0.5,
    )
    for _, row in stats.iterrows():
        axes[1].annotate(row["domain"], (row["mean_conf"], row["accuracy"]),
                         textcoords="offset points", xytext=(6, 3), fontsize=8)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[1].set_xlabel("Mean Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Mean Confidence\n(bubble size = # questions)")
    plt.colorbar(scatter, ax=axes[1], label="Calibration gap (conf − acc)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 2: Overconfidence analysis ──────────────────────────────────────────

def compute_overconfidence(df_known: pd.DataFrame) -> pd.DataFrame:
    """Return per-domain counts of high-confidence wrong answers."""
    df_wrong = df_known[~df_known["correct"]].copy()
    df_hcw   = df_wrong[df_wrong["confidence"] >= HIGH_CONF_THRESHOLD]

    hcw = df_hcw.groupby("domain").agg(
        highconf_wrong=("correct", "count"),
        mean_conf_hcw =("confidence", "mean"),
    ).reset_index()

    total_wrong = df_wrong.groupby("domain").size().reset_index(name="total_wrong")
    hcw = hcw.merge(total_wrong, on="domain")
    hcw["pct_of_wrong"] = hcw["highconf_wrong"] / hcw["total_wrong"]
    return hcw.sort_values("pct_of_wrong", ascending=False)


def plot_overconfidence(df_known: pd.DataFrame, hcw: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, grp, color in [
        ("Correct", df_known[df_known["correct"]],  "#2ecc71"),
        ("Wrong",   df_known[~df_known["correct"]], "#e74c3c"),
    ]:
        axes[0].hist(grp["confidence_pct"], bins=20, alpha=0.6,
                     label=label, color=color, edgecolor="white")
    axes[0].axvline(HIGH_CONF_THRESHOLD * 100, color="gray", linestyle="--",
                    lw=1.2, label=f"{HIGH_CONF_THRESHOLD:.0%} threshold")
    axes[0].set_xlabel("Confidence (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution: Correct vs Wrong")
    axes[0].legend()

    axes[1].barh(hcw["domain"], hcw["pct_of_wrong"], color="#e74c3c", edgecolor="white")
    axes[1].set_xlabel(f"% of wrong answers with conf ≥ {HIGH_CONF_THRESHOLD:.0%}")
    axes[1].set_title("Overconfidence Rate per Domain")
    axes[1].set_xlim(0, 1)
    for bar, val in zip(axes[1].patches, hcw["pct_of_wrong"]):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 3: Reliability diagrams ──────────────────────────────────────────────

def plot_reliability_diagrams(df_known: pd.DataFrame, save_path: str):
    domains = sorted(df_known["domain"].unique())
    ncols   = 3
    nrows   = int(np.ceil(len(domains) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, domain in enumerate(domains):
        sub    = df_known[df_known["domain"] == domain]
        y_true = sub["correct"].astype(int).values
        y_prob = sub["confidence"].values
        ax     = axes[i]
        n_bins = min(10, max(3, len(sub) // 10))

        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="quantile"
            )
            ax.plot(prob_pred, prob_true, "o-", color="#3498db", lw=2, ms=5, label="Model")
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{domain}  (n={len(sub)})")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction Correct")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Reliability Diagrams by Domain", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── ECE ───────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_CALIBRATION_BINS,
) -> float:
    """Weighted mean |confidence - accuracy| across equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(conf - acc)
    return ece


def compute_ece_table(df_known: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in sorted(df_known["domain"].unique()):
        sub = df_known[df_known["domain"] == domain]
        ece = expected_calibration_error(
            sub["correct"].astype(int).values,
            sub["confidence"].values,
        )
        rows.append({"domain": domain, "ECE": ece, "n": len(sub)})
    return pd.DataFrame(rows).sort_values("ECE", ascending=False)


def plot_ece(ece_df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [
        "#e74c3c" if e > 0.15 else "#e67e22" if e > 0.08 else "#2ecc71"
        for e in ece_df["ECE"]
    ]
    bars = ax.bar(ece_df["domain"], ece_df["ECE"], color=colors, edgecolor="white")
    ax.axhline(0.1, color="gray", linestyle="--", lw=1.2, label="ECE = 0.10 guideline")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("ECE by Domain (lower is better)")
    ax.legend()
    for bar, val in zip(bars, ece_df["ECE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 5: Confidence bins vs accuracy ──────────────────────────────────────

def plot_conf_bins_accuracy(df_known: pd.DataFrame, save_path: str):
    df_known = df_known.copy()
    bin_edges  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = ["0-10", "11-20", "21-30", "31-40", "41-50",
                  "51-60", "61-70", "71-80", "81-90", "91-100"]
    df_known["conf_bin"] = pd.cut(
        df_known["confidence_pct"], bins=bin_edges, right=True, labels=bin_labels
    )

    bin_stats = (
        df_known
        .groupby("conf_bin", observed=True)
        .agg(count=("correct", "count"), accuracy=("correct", "mean"))
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(bin_stats["conf_bin"], bin_stats["count"],
            color="#95a5a6", alpha=0.5, label="# questions")
    ax2.plot(bin_stats["conf_bin"], bin_stats["accuracy"],
             "o-", color="#2980b9", lw=2, ms=7, label="Accuracy")

    ax1.set_xlabel("Confidence Bin (%)")
    ax1.set_ylabel("Number of Questions", color="#95a5a6")
    ax2.set_ylabel("Accuracy", color="#2980b9")
    ax2.set_ylim(0, 1.1)
    plt.title("Accuracy per Confidence Bin (Overall)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 6: Heatmap ───────────────────────────────────────────────────────────

def plot_heatmap(df_known: pd.DataFrame, save_path: str):
    df_known = df_known.copy()
    bin_edges  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = ["0-10", "11-20", "21-30", "31-40", "41-50",
                  "51-60", "61-70", "71-80", "81-90", "91-100"]
    df_known["conf_bin"] = pd.cut(
        df_known["confidence_pct"], bins=bin_edges, right=True, labels=bin_labels
    )
    pivot = (
        df_known
        .groupby(["domain", "conf_bin"], observed=True)["correct"]
        .mean()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Accuracy by Domain × Confidence Bin")
    ax.set_xlabel("Confidence Bin (%)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(
    domain_stats: pd.DataFrame,
    ece_df: pd.DataFrame,
    hcw: pd.DataFrame,
) -> pd.DataFrame:
    summary = domain_stats.merge(ece_df[["domain", "ECE"]], on="domain")
    summary = summary.merge(
        hcw[["domain", "highconf_wrong", "pct_of_wrong"]], on="domain", how="left"
    )
    display_cols = [
        "domain", "total", "correct", "wrong",
        "accuracy", "mean_conf", "calibration_gap", "ECE", "pct_of_wrong",
    ]
    summary.columns = [
        "Domain", "Total", "Correct", "Accuracy", "Mean Conf", "Std Conf",
        "Wrong", "Cal. Gap", "ECE", "High-Conf Wrong", "% Wrong HC",
    ]
    return summary.sort_values("Accuracy", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading scored results: {SCORED_RESULTS_PATH}")
    df = load_and_clean(SCORED_RESULTS_PATH)

    df_known = df[df["LLM_right_norm"].notna()].copy()
    df_known["correct"] = df_known["LLM_right_norm"].astype(bool)
    print(f"  {len(df_known)} scoreable rows (excluding unknowns)")

    # Domain stats
    domain_stats = compute_domain_stats(df_known)
    print_domain_stats(domain_stats)

    # Overconfidence
    hcw = compute_overconfidence(df_known)
    print(f"\nHigh-confidence (≥{HIGH_CONF_THRESHOLD:.0%}) wrong answers by domain:")
    print(hcw.to_string(index=False, float_format="{:.3f}".format))

    # ECE
    ece_df = compute_ece_table(df_known)
    print("\nExpected Calibration Error by domain:")
    print(ece_df.to_string(index=False, float_format="{:.4f}".format))

    # Plots
    print("\nGenerating plots...")
    plot_domain_accuracy(domain_stats,  f"{PLOTS_DIR}/domain_accuracy.png")
    plot_overconfidence(df_known, hcw,  f"{PLOTS_DIR}/overconfidence.png")
    plot_reliability_diagrams(df_known, f"{PLOTS_DIR}/reliability_diagrams.png")
    plot_ece(ece_df,                    f"{PLOTS_DIR}/ece_by_domain.png")
    plot_conf_bins_accuracy(df_known,   f"{PLOTS_DIR}/conf_bins_accuracy.png")
    plot_heatmap(df_known,              f"{PLOTS_DIR}/heatmap_domain_conf.png")

    # Summary CSV
    summary = build_summary_table(domain_stats, ece_df, hcw)
    summary.to_csv(CONFIDENCE_SUMMARY_PATH, index=False)
    print(f"\nSummary table saved: {CONFIDENCE_SUMMARY_PATH}")

    # Final print
    total_correct = df_known["correct"].sum()
    total_known   = len(df_known)
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"  Model          : {df['domain'].count()} total questions")
    print(f"  Overall accuracy: {total_correct}/{total_known} = {total_correct / total_known:.1%}")
    print(f"  Confidence range: {df['confidence_pct'].min():.0f}–{df['confidence_pct'].max():.0f}%")
    print("=" * 62)
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
