#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# =========================
# SETTINGS
# =========================
INPUT_FILENAME = "sdoh_all_notes_with_llm_strata.csv"
OUTPUT_ROOT_FOLDER = "jamia_descriptives_outputs"

KEYS = ["subject_id", "hadm_id", "note_id"]

DOMAINS = [
    "relationship_status",
    "employment_status",
    "housing_issues",
    "parental_status",
    "social_support",
    "transportation_issues",
]

MODEL_SPECS = [
    ("m1", "{domain}__m1", "DeepSeek"),
    ("m2", "{domain}__m2", "GPT-OSS"),
    ("m3", "{domain}__m3", "Mistral"),
    ("majvote", "{domain}", "Majority vote (≥2)"),
]

VALID_LABELS = {
    "employment_status": {"employed", "underemployed", "unemployed", "disability", "retired", "student"},
    "housing_issues": {"financial_status", "undomiciled", "other"},
    "transportation_issues": {"distance", "resources", "other"},
    "social_support": {"plus", "minus"},
    "relationship_status": {"widowed", "married", "single", "partnered", "divorced"},
    "parental_status": {"yes", "no"},
}

CATS = ["Predicted", "Unsure", "Not mentioned"]
CAT_XTICK = ["Pred", "Unsure", "NM"]

CAT_COLORS = {
    "Predicted": "#4C78A8",      # azul
    "Unsure": "#F58518",         # naranja
    "Not mentioned": "#54A24B",  # verde
}


def ensure_cols(df, cols, name="INPUT"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")


def normalize_key_columns(df, keys):
    out = df.copy()
    for k in keys:
        out[k] = out[k].astype("string").str.strip()
    return out


def build_outdir(script_dir: Path) -> Path:
    root = script_dir / OUTPUT_ROOT_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = root / f"run_{run_tag}_grid_barplots"
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "tables").mkdir(exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    return outdir


def categorize_labels(domain: str, s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s_norm = s.str.strip().str.lower()

    is_empty = s.isna() | s_norm.eq("").fillna(False)
    is_not_mentioned = s_norm.eq("not_mentioned").fillna(False)
    is_pred = s_norm.isin(VALID_LABELS[domain]).fillna(False)

    out = np.select(
        [
            (is_empty | is_not_mentioned).to_numpy(dtype=bool),
            is_pred.to_numpy(dtype=bool),
        ],
        [
            "Not mentioned",
            "Predicted",
        ],
        default="Unsure",
    )
    return pd.Series(out, index=s.index)


def compute_mix_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    for domain in DOMAINS:
        for internal_name, template, display_name in MODEL_SPECS:
            col = template.format(domain=domain)
            ensure_cols(df, [col], name="INPUT")
            cats = categorize_labels(domain, df[col])
            counts = cats.value_counts(dropna=False).to_dict()
            for cat in CATS:
                n = int(counts.get(cat, 0))
                rows.append({
                    "domain": domain,
                    "model": display_name,
                    "model_internal": internal_name,
                    "category": cat,
                    "n": n,
                    "total": n_total,
                    "prop": n / n_total if n_total else np.nan
                })
    return pd.DataFrame(rows)


def plot_grid_bars(mix_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    nrows = len(DOMAINS)
    ncols = len(MODEL_SPECS)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10), sharey=True)

    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    legend_handles = None

    for i, domain in enumerate(DOMAINS):
        for j, (_, _, display_name) in enumerate(MODEL_SPECS):
            ax = axes[i, j]
            sub = mix_df[(mix_df["domain"] == domain) & (mix_df["model"] == display_name)]
            vals = [float(sub.loc[sub["category"] == cat, "prop"].values[0]) for cat in CATS]

            bar_colors = [CAT_COLORS[cat] for cat in CATS]
            bars = ax.bar(range(len(CATS)), vals, color=bar_colors)

            if legend_handles is None:
                legend_handles = bars

            ax.set_ylim(0, 1.0)

            # X axis only on bottom row
            ax.set_xticks(range(len(CATS)))
            if i == nrows - 1:
                ax.set_xticklabels(CAT_XTICK)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            # Column titles (model names)
            if i == 0:
                ax.set_title(display_name)

            # Row labels
            if j == 0:
                ax.set_ylabel(domain.replace("_", " ").title())

            ax.grid(axis="y", linestyle=":", linewidth=0.6)

            # Add % labels above bars
            labels = [f"{v*100:.1f}%" for v in vals]
            try:
                ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
            except Exception:
                for b, lab in zip(bars, labels):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.02,
                        lab,
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

    # No overall title
    fig.legend(
        legend_handles, CATS,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / INPUT_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    outdir = build_outdir(script_dir)
    table_dir = outdir / "tables"
    fig_dir = outdir / "figures"

    df = pd.read_csv(input_path)
    ensure_cols(df, KEYS, "INPUT")
    df = normalize_key_columns(df, KEYS)

    cohort_summary = {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique(dropna=True)),
        "n_hadm": int(df["hadm_id"].nunique(dropna=True)),
        "n_notes": int(df["note_id"].nunique(dropna=True)),
    }
    pd.DataFrame([cohort_summary]).to_csv(table_dir / "table_1a_cohort_counts.csv", index=False)

    mix_df = compute_mix_table(df)
    mix_df.to_csv(table_dir / "supplementary_table_s1_label_mix_domain_model.csv", index=False)
    mix_df.groupby(["domain", "model"])["prop"].sum().reset_index().to_csv(
        table_dir / "sanity_check_prop_sums.csv", index=False
    )

    plot_grid_bars(
        mix_df,
        fig_dir / "figure_1_grid_label_mix_domain_x_model.png",
        fig_dir / "figure_1_grid_label_mix_domain_x_model.pdf",
    )

    print("Saved to:", outdir)


if __name__ == "__main__":
    main()