# make_jamia_figures_clean.py
# Generates publication-ready JAMIA figures (Main Figures 1–4) + optional confusion matrices (Supplement)
#
# Requirements:
#   pip install pandas numpy matplotlib scikit-learn openpyxl
#
# Expected folder structure (from the pipeline):
#   sdoh_metrics_outputs/
#     Employment_status__metrics.json
#     Housing_issues__metrics.json
#     ...
#     Employment_status__eval_majority_gold.csv
#     Housing_issues__eval_majority_gold.csv
#     ...
#     confusion_matrices/
#        Employment_status__llm_maj_vote_cm.csv   (optional)
#
# Output:
#   figures_jamia/
#     Fig1_MacroF1_Heatmap.png/pdf
#     Fig2_UnanimousAgreement_CoveragePrecision.png/pdf
#     Fig3_ModelAgreement_Strata.png/pdf
#     Fig4_IAA_KappaAlpha.png/pdf
#     SUPP_ConfusionMatrix_*.png/pdf (optional)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional (only needed if recomputing CMs from eval CSVs)
try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("sdoh_metrics_outputs")
FIG_DIR = Path("figures_jamia")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Domain order (edit to match the manuscript)
DOMAIN_ORDER = [
    "Employment_status",
    "Housing_issues",
    "Parental_status",
    "Social_support",
    "Transportation_issues",
    "Relationship_status",  # if/when available
]

MODEL_ORDER = ["m1", "m2", "m3", "llm_maj_vote"]  # keys in metrics.json

# Confusion matrices you might want (recommended as supplement)
SUPP_CM_DOMAINS = ["Employment_status", "Social_support"]
SUPP_CM_MODEL = "llm_maj_vote"  # or "m1"/"m2"/"m3"


# -----------------------------
# Publication labels (Title Case, no underscores)
# -----------------------------
DOMAIN_LABELS = {
    "Employment_status": "Employment Status",
    "Housing_issues": "Housing Issues",
    "Parental_status": "Parental Status",
    "Social_support": "Social Support",
    "Transportation_issues": "Transportation Issues",
    "Relationship_status": "Relationship Status",
}

MODEL_LABELS = {
    "m1": "DeepSeek",
    "m2": "GPT-oss",
    "m3": "Mistral",
    "llm_maj_vote": "LLM Majority Vote",
}

STRATA_LABELS = {
    "all3_same": "All Three Agree",
    "maj2_same": "Two Agree (Majority)",
    "all3_diff": "All Three Disagree",
}

# -----------------------------
# Global style (clean margins, legible typography)
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

SAVE_DPI = 300
PAD_INCHES = 0.02  # tight, journal-friendly margins


# -----------------------------
# Helpers
# -----------------------------
def load_metrics_jsons(input_dir: Path) -> dict:
    metrics_files = sorted(input_dir.glob("*__metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No *__metrics.json found in {input_dir.resolve()}")

    out = {}
    for fp in metrics_files:
        with open(fp, "r") as f:
            j = json.load(f)
        sheet = j.get("sheet") or fp.name.split("__metrics.json")[0]
        out[sheet] = j
    return out

def safe_get(d, path, default=np.nan):
    """path is list of keys; returns default if missing."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def display_domain(domain_key: str) -> str:
    return DOMAIN_LABELS.get(domain_key, domain_key.replace("_", " ").title())

def display_model(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key.replace("_", " ").title())

def clean_axes(ax):
    # Clean, minimal spines (journal-friendly)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)

def save_fig(fig, outpath_base: Path):
    # tight layout + tight file bounding box for crisp margins
    fig.tight_layout(pad=0.6)
    fig.savefig(str(outpath_base.with_suffix(".png")), dpi=SAVE_DPI, bbox_inches="tight", pad_inches=PAD_INCHES)
    fig.savefig(str(outpath_base.with_suffix(".pdf")), bbox_inches="tight", pad_inches=PAD_INCHES)
    plt.close(fig)

def _text_color_for_cell(im, value, nan_color="black"):
    """Choose white/black text for readability against a heatmap cell background."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return nan_color
    # Convert to RGBA and compute relative luminance
    norm = im.norm
    cmap = im.cmap
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.55 else "white"

def annotate_heatmap(ax, im, data, fmt="{:.2f}", fontsize=9):
    """Annotate a heatmap with dynamic (contrast-aware) text colors."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = "NA" if np.isnan(v) else fmt.format(v)
            color = _text_color_for_cell(im, v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize, color=color)


# -----------------------------
# Build summary dataframe from metrics.json
# -----------------------------
def build_summary_df(metrics_by_sheet: dict) -> pd.DataFrame:
    rows = []
    for sheet, j in metrics_by_sheet.items():
        row = {
            "sheet": sheet,
            "n_common_notes": safe_get(j, ["n_common_notes"], np.nan),
            "n_majority_gold": safe_get(j, ["n_majority_gold"], np.nan),
            "n_iaa_items": safe_get(j, ["n_iaa_items"], np.nan),
            "pct_all3_agree": safe_get(j, ["pct_all3_agree"], np.nan),
            "fleiss_kappa": safe_get(j, ["fleiss_kappa"], np.nan),
            "krippendorff_alpha": safe_get(j, ["krippendorff_alpha_nominal"], np.nan),
            "unanimity_coverage": safe_get(j, ["unanimity_coverage_on_eval"], np.nan),
            "unanimity_precision": safe_get(j, ["unanimity_precision_on_eval"], np.nan),
        }
        for m in MODEL_ORDER:
            row[f"{m}_acc"] = safe_get(j, ["metrics", m, "accuracy"], np.nan)
            row[f"{m}_macro_f1"] = safe_get(j, ["metrics", m, "macro_f1"], np.nan)
        row["llm_stratum_counts_eval"] = safe_get(j, ["llm_stratum_counts_eval"], {})
        rows.append(row)

    df = pd.DataFrame(rows)

    df["sheet"] = df["sheet"].astype(str)
    df["sheet_order"] = df["sheet"].apply(lambda s: DOMAIN_ORDER.index(s) if s in DOMAIN_ORDER else 999)
    df = df.sort_values("sheet_order").drop(columns=["sheet_order"]).reset_index(drop=True)
    return df


# -----------------------------
# FIG 1: Heatmap macro-F1 by domain x model
# -----------------------------
def fig1_heatmap_macro_f1(df_summary: pd.DataFrame):
    domains = df_summary["sheet"].tolist()
    domains_disp = [display_domain(d) for d in domains]

    data = np.array([[r.get(f"{m}_macro_f1", np.nan) for m in MODEL_ORDER]
                     for _, r in df_summary.iterrows()], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.55 * len(domains))))
    im = ax.imshow(data, aspect="auto")  # default colormap

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([display_model(m) for m in MODEL_ORDER], rotation=0)
    ax.set_yticks(range(len(domains_disp)))
    ax.set_yticklabels(domains_disp)

    ax.set_xlabel("Model")
    # ax.set_title("Macro-F1 by SDOH Domain and Model (Majority-Vote Gold Standard)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Macro-F1")

    annotate_heatmap(ax, im, data, fmt="{:.3f}", fontsize=8.5)
    clean_axes(ax)
    save_fig(fig, FIG_DIR / "Fig1_MacroF1_Heatmap")


# -----------------------------
# FIG 2: Unanimity precision vs coverage by domain
# -----------------------------
def fig2_unanimity_precision_coverage(df_summary: pd.DataFrame):
    domains = df_summary["sheet"].tolist()
    domains_disp = [display_domain(d) for d in domains]

    cov = df_summary["unanimity_coverage"].astype(float).values
    prec = df_summary["unanimity_precision"].astype(float).values

    n = len(domains_disp)
    x = np.arange(n)
    width = 0.38

    # Figure size: fixed width, slightly adaptive height
    height = max(3.6, 0.55 * n) if n > 6 else 3.6
    fig, ax = plt.subplots(figsize=(8.6, height))

    ax.bar(x - width / 2, cov, width, label="Coverage (Fraction of Evaluation Notes)")
    ax.bar(x + width / 2, prec, width, label="Precision (Within Unanimous Subset)")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    # ax.set_title("Unanimous Model Agreement: Coverage and Precision")

    # Value labels above bars
    for i, (c, p) in enumerate(zip(cov, prec)):
        if np.isnan(c):
            c_text, c_y = "NA", 0.03
        else:
            c_text, c_y = f"{c:.2f}", min(1.02, c + 0.03)

        if np.isnan(p):
            p_text, p_y = "NA", 0.03
        else:
            p_text, p_y = f"{p:.2f}", min(1.02, p + 0.03)

        ax.text(i - width / 2, c_y, c_text, ha="center", va="bottom", fontsize=8.5)
        ax.text(i + width / 2, p_y, p_text, ha="center", va="bottom", fontsize=8.5)

    clean_axes(ax)

    # X labels: horizontal, 2 lines (split on space)
    wrapped_domains = [d.replace(" ", "\n") for d in domains_disp]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_domains, rotation=0, ha="center", va="top")

    # Legend: below x-axis, outside, one row
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.6,
    )

    # Extra bottom margin for tick labels + legend
    fig.subplots_adjust(bottom=0.34)

    save_fig(fig, FIG_DIR / "Fig2_UnanimousAgreement_CoveragePrecision")


# -----------------------------
# FIG 3: Model agreement strata proportions (stacked)
# -----------------------------
def fig3_llm_agreement_strata(df_summary: pd.DataFrame):
    domains = df_summary["sheet"].tolist()
    domains_disp = [display_domain(d) for d in domains]

    strata = ["all3_same", "maj2_same", "all3_diff"]
    props = {s: [] for s in strata}

    for _, r in df_summary.iterrows():
        counts = r.get("llm_stratum_counts_eval", {}) or {}
        total = float(sum(counts.values())) if isinstance(counts, dict) else 0.0
        for s in strata:
            v = float(counts.get(s, 0)) / total if total > 0 else np.nan
            props[s].append(v)

    x = np.arange(len(domains_disp))
    fig, ax = plt.subplots(figsize=(8.6, max(3.2, 0.55 * len(domains_disp))))

    bottom = np.zeros(len(domains_disp), dtype=float)
    for s in strata:
        vals = np.array(props[s], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=STRATA_LABELS.get(s, s))
        bottom = np.where(np.isnan(vals), bottom, bottom + np.nan_to_num(vals))

    ax.set_xticks(x)
    ax.set_xticklabels(domains_disp, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_title("Distribution of Model-Agreement Strata (Evaluation Set)")
    ax.legend(loc="best", frameon=False)

    clean_axes(ax)
    save_fig(fig, FIG_DIR / "Fig3_ModelAgreement_Strata")


# -----------------------------
# FIG 4: IAA kappa + alpha by domain
# -----------------------------
def fig4_iaa(df_summary: pd.DataFrame):
    domains = df_summary["sheet"].tolist()
    domains_disp = [display_domain(d) for d in domains]

    kappa = df_summary["fleiss_kappa"].astype(float).values
    alpha = df_summary["krippendorff_alpha"].astype(float).values

    x = np.arange(len(domains_disp))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.6, max(3.2, 0.55 * len(domains_disp))))
    ax.bar(x - width/2, kappa, width, label="Fleiss’ κ")
    ax.bar(x + width/2, alpha, width, label="Krippendorff’s α (Nominal)")

    ax.set_xticks(x)
    ax.set_xticklabels(domains_disp, rotation=30, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Agreement")
    ax.set_title("Inter-Annotator Agreement by SDOH Domain")
    ax.legend(loc="best", frameon=False)

    for i, (k, a) in enumerate(zip(kappa, alpha)):
        ax.text(i - width/2, min(1.02, (k if not np.isnan(k) else 0) + 0.03), "NA" if np.isnan(k) else f"{k:.2f}",
                ha="center", va="bottom", fontsize=8.5)
        ax.text(i + width/2, min(1.02, (a if not np.isnan(a) else 0) + 0.03), "NA" if np.isnan(a) else f"{a:.2f}",
                ha="center", va="bottom", fontsize=8.5)

    clean_axes(ax)
    save_fig(fig, FIG_DIR / "Fig4_IAA_KappaAlpha")


# -----------------------------
# OPTIONAL: Confusion matrices (supplement)
# -----------------------------
def plot_confusion_matrix_from_csv(cm_csv: Path, title: str, outbase: Path):
    cm = pd.read_csv(cm_csv, index_col=0)
    labels = cm.index.tolist()
    data = cm.values.astype(float)

    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(l).title() for l in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(l).title() for l in labels])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Count")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = _text_color_for_cell(im, v)
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=8, color=color)

    clean_axes(ax)
    save_fig(fig, outbase)

def compute_cm_from_eval_csv(eval_csv: Path, gold_col="gold_label", pred_col="llm_maj_vote"):
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required to compute confusion matrices from eval CSVs.")
    df = pd.read_csv(eval_csv, dtype=str)
    if gold_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Missing columns in {eval_csv.name}: need {gold_col} and {pred_col}")
    y_true = df[gold_col].fillna("Unknown").tolist()
    y_pred = df[pred_col].fillna("Unknown").tolist()
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)

def make_supp_confusion_matrices():
    cm_dir = INPUT_DIR / "confusion_matrices"
    for domain in SUPP_CM_DOMAINS:
        cm_csv = cm_dir / f"{domain}__{SUPP_CM_MODEL}_cm.csv"
        domain_disp = display_domain(domain)

        if cm_csv.exists():
            plot_confusion_matrix_from_csv(
                cm_csv,
                title=f"{domain_disp}: Confusion Matrix ({display_model(SUPP_CM_MODEL)} vs Gold Standard)",
                outbase=FIG_DIR / f"SUPP_ConfusionMatrix_{domain}_{SUPP_CM_MODEL}"
            )
            continue

        eval_csv = INPUT_DIR / f"{domain}__eval_majority_gold.csv"
        if eval_csv.exists():
            cm = compute_cm_from_eval_csv(eval_csv, gold_col="gold_label", pred_col=SUPP_CM_MODEL)
            tmp_csv = FIG_DIR / f"SUPP_ConfusionMatrix_{domain}_{SUPP_CM_MODEL}.csv"
            cm.to_csv(tmp_csv)

            plot_confusion_matrix_from_csv(
                tmp_csv,
                title=f"{domain_disp}: Confusion Matrix ({display_model(SUPP_CM_MODEL)} vs Gold Standard)",
                outbase=FIG_DIR / f"SUPP_ConfusionMatrix_{domain}_{SUPP_CM_MODEL}"
            )
        else:
            print(f"[WARN] Missing both CM csv and eval csv for {domain}. Skipping confusion matrix.")


# -----------------------------
# MAIN
# -----------------------------
def main():
    metrics_by_sheet = load_metrics_jsons(INPUT_DIR)
    df_summary = build_summary_df(metrics_by_sheet)

    df_summary.to_csv(FIG_DIR / "Summary_For_Figures.csv", index=False)

    fig1_heatmap_macro_f1(df_summary)
    fig2_unanimity_precision_coverage(df_summary)
    fig3_llm_agreement_strata(df_summary)
    fig4_iaa(df_summary)

    make_supp_confusion_matrices()

    print(f"✅ Done. Figures saved in: {FIG_DIR.resolve()}")

if __name__ == "__main__":
    main()
