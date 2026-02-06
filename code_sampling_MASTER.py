import os
import re
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Tus archivos y prefijos EXACTOS
CATEGORIES = {
    "relationship_status": {
        "path": "consenso_relationship_status.csv",
        "prefix": "Relationship status",
        "valid": ["married", "partnered", "divorced", "widowed", "single"],
    },
    "employment_status": {
        "path": "consenso_employment_status.csv",
        "prefix": "Employment status",
        "valid": ["employed", "underemployed", "unemployed", "disability", "retired", "student"],
    },
    "housing_issues": {
        "path": "consenso_housing_issues.csv",
        "prefix": "Housing issues",
        "valid": ["financial_status", "undomiciled", "other"],
    },
    "parental_status": {
        "path": "consenso_parental_status.csv",
        "prefix": "Parental status",
        "valid": ["yes", "no"],
    },
    "social_support": {
        "path": "consenso_social_support.csv",
        "prefix": "Social support",
        "valid": ["plus", "minus"],
    },
    "transportation_issues": {
        "path": "consenso_transportation_issues.csv",
        "prefix": "Transportation issues",
        "valid": ["distance", "resources", "other"],
    },
}

ID_COLS = ["subject_id", "hadm_id", "note_id"]
MAX_PER_COMBO = 3

OUTDIR = "combo_sampling_outputs"
os.makedirs(OUTDIR, exist_ok=True)


# =========================
# Helpers
# =========================
def normalize_token(x) -> str:
    """Normalize raw model output to a clean lowercase token (or empty string)."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return ""
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_+\-]", "", s)
    return s

def to_valid_or_unknown(x: str, valid_set: set) -> str:
    """Map any non-valid token to 'unknown'."""
    tok = normalize_token(x)
    if tok in valid_set:
        return tok
    return "unknown"

def combo_type(m1, m2, m3):
    """Quick tag for analysis."""
    if m1 == m2 == m3:
        return "all3_same"
    if (m1 == m2) or (m1 == m3) or (m2 == m3):
        return "maj2_same"
    return "all3_diff"


# =========================
# Main per-category sampling
# =========================
all_samples = []

for cat, cfg in CATEGORIES.items():
    path = cfg["path"]
    prefix = cfg["prefix"]
    valid_set = set(cfg["valid"])

    df = pd.read_csv(path)
    # normalize IDs
    for c in ID_COLS:
        if c not in df.columns:
            raise ValueError(f"[{cat}] Missing required ID column: {c}")
    df["note_id"] = df["note_id"].astype(str).str.strip()
    df["hadm_id"] = df["hadm_id"].astype(str).str.strip()
    df["subject_id"] = df["subject_id"].astype(str).str.strip()

    col_deepseek = f"{prefix}_deepseek"
    col_gptoss   = f"{prefix}_gptoss"
    col_mistral  = f"{prefix}_mistral"

    missing_cols = [c for c in [col_deepseek, col_gptoss, col_mistral] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{cat}] Missing model columns: {missing_cols}. Found: {df.columns.tolist()}")

    # map to valid or unknown
    df[f"{cat}__m1"] = df[col_deepseek].apply(lambda x: to_valid_or_unknown(x, valid_set))
    df[f"{cat}__m2"] = df[col_gptoss].apply(lambda x: to_valid_or_unknown(x, valid_set))
    df[f"{cat}__m3"] = df[col_mistral].apply(lambda x: to_valid_or_unknown(x, valid_set))

    # build combo
    df[f"{cat}__combo"] = df[f"{cat}__m1"] + "|" + df[f"{cat}__m2"] + "|" + df[f"{cat}__m3"]
    df[f"{cat}__combo_type"] = df.apply(lambda r: combo_type(r[f"{cat}__m1"], r[f"{cat}__m2"], r[f"{cat}__m3"]), axis=1)

    # sample up to MAX_PER_COMBO per combo
    def sample_group(g):
        k = min(MAX_PER_COMBO, len(g))
        return g.sample(n=k, replace=False, random_state=RANDOM_STATE)

    df_s = df.groupby(f"{cat}__combo", group_keys=False).apply(sample_group).reset_index(drop=True)

    # minimal output columns for annotation (you can add note_link later by merge)
    out_cols = ID_COLS + [
        f"{cat}__m1", f"{cat}__m2", f"{cat}__m3",
        f"{cat}__combo", f"{cat}__combo_type",
        col_deepseek, col_gptoss, col_mistral
    ]
    df_s_out = df_s[out_cols].copy()
    df_s_out["category"] = cat

    # stats: how many combos exist, how many sampled, and original frequencies
    stats = (
        df.groupby(f"{cat}__combo")
          .size()
          .reset_index(name="n_in_full_data")
          .merge(
              df_s.groupby(f"{cat}__combo").size().reset_index(name="n_sampled"),
              on=f"{cat}__combo",
              how="left"
          )
    )
    stats["n_sampled"] = stats["n_sampled"].fillna(0).astype(int)
    stats = stats.sort_values(["n_in_full_data", f"{cat}__combo"], ascending=[False, True])

    # save per-category
    sample_path = os.path.join(OUTDIR, f"sample_{cat}_max{MAX_PER_COMBO}_per_combo.csv")
    stats_path  = os.path.join(OUTDIR, f"stats_{cat}_combo_freqs.csv")
    df_s_out.to_csv(sample_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(f"[{cat}] combos in full data: {stats.shape[0]} | sampled notes: {df_s_out.shape[0]}")
    print(f"  saved: {sample_path}")
    print(f"  saved: {stats_path}")

    all_samples.append(df_s_out)

# =========================
# Optional: master union across categories
# =========================
df_master = pd.concat(all_samples, ignore_index=True)
master_path = os.path.join(OUTDIR, f"MASTER_all_categories_max{MAX_PER_COMBO}_per_combo.csv")
df_master.to_csv(master_path, index=False)
print(f"\nSaved master: {master_path}")

# Optional: summary of master counts
summary = (df_master.groupby(["category"])
           .agg(n_sampled_notes=("note_id", "nunique"),
                n_rows=("note_id", "size"))
           .reset_index())
summary_path = os.path.join(OUTDIR, "MASTER_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")
