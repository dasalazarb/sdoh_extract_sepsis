# compute_sdoh_metrics.py
import os
import re
import json
import numpy as np
import pandas as pd

from collections import Counter, defaultdict

# Optional sklearn (highly recommended). If not available, the script will still run
# but will skip some detailed reports.
try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        f1_score,
        precision_recall_fscore_support
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ============================================================
# CONFIG
# ============================================================

EXCEL_FILES = {
    "Diego":  "form_Diego_sdoh.xlsx",
    "Daniel": "form_Daniel_sdoh.xlsx",
    "Pankaj": "form_Pankaj_sdoh.xlsx",
}

OUTDIR = "sdoh_metrics_outputs"
os.makedirs(OUTDIR, exist_ok=True)

SHEETS = {
    "Employment_status": {
        "cat_key": "employment_status",
        "labels": ["employed", "underemployed", "unemployed", "disability", "retired", "student"],
    },
    "Relationship_status": {
        "cat_key": "relationship_status",
        "labels": ["married", "partnered", "divorced", "widowed", "single"],
    },
    "Parental_status": {
        "cat_key": "parental_status",
        "labels": ["yes", "no"],
    },
    "Social_support": {
        "cat_key": "social_support",
        "labels": ["plus", "minus"],
    },
    "Housing_issues": {
        "cat_key": "housing_issues",
        "labels": ["financial_status", "undomiciled", "other"],
    },
    "Transportation_issues": {
        "cat_key": "transportation_issues",
        "labels": ["distance", "resources", "other"],
    },
}

ID_COLS = ["subject_id", "hadm_id", "note_id"]

# columns expected for model predictions (you created these)
MODEL_COLS = ["m1_pred", "m2_pred", "m3_pred", "combo"]

# how to interpret marked cells in annotation label columns
MARK_RE = re.compile(r"^\s*x\s*$", flags=re.IGNORECASE)


# ============================================================
# Helpers: normalization & extraction
# ============================================================

def norm_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s

def norm_label(x):
    """Lowercase, normalize underscores; empty -> unknown."""
    s = norm_str(x).lower()
    if s in {"", "nan", "none", "null"}:
        return "unknown"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_+\-]", "", s)
    if s == "":
        return "unknown"
    return s

def pick_human_label(row, label_cols):
    """
    Determine which label the annotator chose (marked with x).
    Returns:
      - label (str)
      - multi_flag (bool)
      - chosen_list (list[str])
    """
    chosen = []
    for col in label_cols:
        v = row.get(col, "")
        if isinstance(v, str) and MARK_RE.match(v):
            chosen.append(col)
        elif not isinstance(v, str) and v == 1:
            # just in case someone used 1 instead of 'x'
            chosen.append(col)

    if len(chosen) == 0:
        return None, False, []
    if len(chosen) == 1:
        return chosen[0], False, chosen
    # multiple labels marked -> problematic
    return "__MULTI__", True, chosen

def majority_vote(labels):
    """
    Majority vote among 3 annotators.
    labels: list of 3 labels (strings or None)
    Returns gold_label, status
    """
    clean = [l for l in labels if l not in [None, "__MULTI__"]]
    if len(clean) < 2:
        return None, "too_few_raters"

    c = Counter(clean)
    top_label, top_count = c.most_common(1)[0]
    if top_count >= 2:
        return top_label, "majority"
    return "__NO_MAJORITY__", "no_majority"

def model_stratum(m1, m2, m3):
    if m1 == m2 == m3:
        return "all3_same"
    if (m1 == m2) or (m1 == m3) or (m2 == m3):
        return "maj2_same"
    return "all3_diff"

def llm_majority_vote(m1, m2, m3):
    """Return label if at least 2 agree, else '__TIE__'."""
    if m1 == m2 or m1 == m3:
        return m1
    if m2 == m3:
        return m2
    return "__TIE__"

def safe_value_counts(series):
    vc = series.value_counts(dropna=False)
    return vc.to_dict()


# ============================================================
# IAA: Fleiss' kappa
# ============================================================

def fleiss_kappa(ratings, categories):
    """
    ratings: list of list[str] size N x n_raters (ideally 3)
             Missing allowed as None, but items with <2 ratings are dropped.
    categories: list of category names (strings)

    Returns kappa or None if cannot compute.
    """
    cats = list(categories)
    cat_to_idx = {c: i for i, c in enumerate(cats)}

    # Build matrix N x k counts per item
    M = []
    for row in ratings:
        row_clean = [r for r in row if r not in [None, "__MULTI__"]]
        if len(row_clean) < 2:
            continue
        counts = [0] * len(cats)
        for r in row_clean:
            if r in cat_to_idx:
                counts[cat_to_idx[r]] += 1
            else:
                # unseen category -> ignore
                pass
        # Need at least 2 ratings counted
        if sum(counts) < 2:
            continue
        M.append(counts)

    if len(M) == 0:
        return None

    M = np.array(M, dtype=float)
    n = M.sum(axis=1)  # number of ratings per item (should be 3)
    k = M.shape[1]

    # P_i for each item
    P_i = []
    for i in range(M.shape[0]):
        ni = n[i]
        if ni <= 1:
            continue
        Pi = (np.sum(M[i] * (M[i] - 1.0))) / (ni * (ni - 1.0))
        P_i.append(Pi)

    if len(P_i) == 0:
        return None

    Pbar = float(np.mean(P_i))

    # p_j overall
    p_j = np.sum(M, axis=0) / np.sum(n)
    Pbar_e = float(np.sum(p_j ** 2))

    if abs(1.0 - Pbar_e) < 1e-12:
        return None

    kappa = (Pbar - Pbar_e) / (1.0 - Pbar_e)
    return float(kappa)


# ============================================================
# IAA: Krippendorff's alpha (nominal)
# ============================================================

def krippendorff_alpha_nominal(ratings):
    """
    ratings: list of list[str] size N x n_raters
             Missing allowed as None, '__MULTI__' treated as missing.
    Nominal distance: delta(a,b)=0 if same else 1.
    """
    # Collect category set
    vals = []
    for row in ratings:
        vals.extend([r for r in row if r not in [None, "__MULTI__"]])
    vals = [v for v in vals if v is not None]
    if len(vals) == 0:
        return None

    # Observed disagreement Do
    Do_num = 0.0
    Do_den = 0.0

    for row in ratings:
        row_clean = [r for r in row if r not in [None, "__MULTI__"]]
        m = len(row_clean)
        if m < 2:
            continue
        # all unordered pairs
        for i in range(m):
            for j in range(i + 1, m):
                Do_num += 0.0 if row_clean[i] == row_clean[j] else 1.0
                Do_den += 1.0

    if Do_den == 0:
        return None
    Do = Do_num / Do_den

    # Expected disagreement De
    counts = Counter(vals)
    total = sum(counts.values())
    if total <= 1:
        return None

    # For nominal: De = 1 - sum(p_c^2)
    p2 = sum((c / total) ** 2 for c in counts.values())
    De = 1.0 - p2
    if De <= 1e-12:
        return None

    alpha = 1.0 - (Do / De)
    return float(alpha)


# ============================================================
# Metrics computation
# ============================================================

def compute_model_metrics(y_true, y_pred, labels_order):
    """
    Returns dict with overall + per-class.
    """
    out = {}
    # filter out missing
    mask = [(t is not None) and (p is not None) for t, p in zip(y_true, y_pred)]
    yt = [y_true[i] for i, ok in enumerate(mask) if ok]
    yp = [y_pred[i] for i, ok in enumerate(mask) if ok]

    out["n_eval"] = len(yt)
    if len(yt) == 0:
        return out

    if SKLEARN_OK:
        out["accuracy"] = float(accuracy_score(yt, yp))
        out["macro_f1"] = float(f1_score(yt, yp, average="macro", labels=labels_order, zero_division=0))
        out["micro_f1"] = float(f1_score(yt, yp, average="micro", labels=labels_order, zero_division=0))
        pr, rc, f1, sup = precision_recall_fscore_support(
            yt, yp, labels=labels_order, zero_division=0
        )
        per_class = {}
        for i, lab in enumerate(labels_order):
            per_class[lab] = {
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
        out["per_class"] = per_class
    else:
        # lightweight fallback: only accuracy
        out["accuracy"] = float(np.mean([1 if a == b else 0 for a, b in zip(yt, yp)]))

    return out

def compute_confusion(yt, yp, labels_order):
    if not SKLEARN_OK:
        return None
    cm = confusion_matrix(yt, yp, labels=labels_order)
    return pd.DataFrame(cm, index=labels_order, columns=labels_order)

def compute_stratum_summary(df_eval, gold_col, pred_cols):
    """
    pred_cols: dict with keys m1/m2/m3/maj_vote/unanimous_pred etc.
    Returns summary DataFrame by stratum.
    """
    rows = []
    for stratum, g in df_eval.groupby("llm_stratum"):
        row = {"llm_stratum": stratum, "n": len(g)}
        yt = g[gold_col].tolist()

        for name, col in pred_cols.items():
            yp = g[col].tolist()
            met = compute_model_metrics(yt, yp, labels_order=sorted(set(yt + yp)))
            row[f"{name}_acc"] = met.get("accuracy", np.nan)
            row[f"{name}_macro_f1"] = met.get("macro_f1", np.nan) if SKLEARN_OK else np.nan
            row[f"{name}_n_eval"] = met.get("n_eval", 0)

        # unanimous rule: precision + coverage on accepted subset
        if "unanimous_pred" in pred_cols:
            accepted = g[g["unanimous_pred"] != "__REJECT__"].copy()
            row["unanimous_coverage"] = len(accepted) / len(g) if len(g) else 0.0
            if len(accepted):
                row["unanimous_precision"] = float(np.mean(accepted["unanimous_pred"] == accepted[gold_col]))
            else:
                row["unanimous_precision"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("llm_stratum")


# ============================================================
# Read one Excel sheet
# ============================================================

def read_annotator_sheet(excel_path, sheet_name, valid_labels):
    """
    Reads a sheet and returns:
      df with ID cols + m1_pred/m2_pred/m3_pred/combo (if present) + human_label + flags
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    # normalize IDs
    for c in ID_COLS:
        if c not in df.columns:
            raise ValueError(f"[{excel_path}::{sheet_name}] Missing required column: {c}")
        df[c] = df[c].astype(str).str.strip()

    # ensure note_id is string
    df["note_id"] = df["note_id"].astype(str).str.strip()

    # normalize model preds if exist
    for c in MODEL_COLS:
        if c in df.columns:
            df[c] = df[c].apply(norm_label)

    # Annotation label columns: valid_labels + unknown
    label_cols = list(valid_labels) + ["unknown"]
    # some spreadsheets may not include 'unknown' column name exactly (rare)
    # if missing, try fallback by searching a column that lowercases to 'unknown'
    cols_lower = {c.lower(): c for c in df.columns}
    if "unknown" not in df.columns and "unknown" in cols_lower:
        df.rename(columns={cols_lower["unknown"]: "unknown"}, inplace=True)

    missing_label_cols = [c for c in label_cols if c not in df.columns]
    if missing_label_cols:
        raise ValueError(
            f"[{excel_path}::{sheet_name}] Missing label columns: {missing_label_cols}. "
            f"Available columns include: {df.columns.tolist()}"
        )

    human_labels = []
    multi_flags = []
    multi_lists = []

    for _, row in df.iterrows():
        lab, multi, chosen = pick_human_label(row, label_cols)
        human_labels.append(lab)
        multi_flags.append(multi)
        multi_lists.append(chosen)

    out = df[ID_COLS].copy()
    # carry model columns if available
    for c in MODEL_COLS:
        if c in df.columns:
            out[c] = df[c].values
        else:
            out[c] = None

    out["human_label"] = human_labels
    out["human_multi"] = multi_flags
    out["human_multi_list"] = [json.dumps(x) for x in multi_lists]

    return out


# ============================================================
# Main
# ============================================================

def main():
    # validate files exist
    for a, p in EXCEL_FILES.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file for {a}: {p}")

    summary_rows = []

    for sheet_name, cfg in SHEETS.items():
        cat_key = cfg["cat_key"]
        valid_labels = [norm_label(x) for x in cfg["labels"]]  # normalize
        label_space = valid_labels + ["unknown"]

        print(f"\n=== Processing sheet: {sheet_name} ({cat_key}) ===")

        # read all annotators
        dfs = {}
        for annot, path in EXCEL_FILES.items():
            df_a = read_annotator_sheet(path, sheet_name, valid_labels=label_space[:-1])  # pass valid (without unknown), function adds unknown
            df_a.rename(columns={"human_label": f"human_{annot}"}, inplace=True)
            df_a.rename(columns={"human_multi": f"multi_{annot}"}, inplace=True)
            df_a.rename(columns={"human_multi_list": f"multi_list_{annot}"}, inplace=True)
            dfs[annot] = df_a

        # intersection of notes across all 3 annotators
        keys_sets = []
        for annot, d in dfs.items():
            keys_sets.append(set(zip(d["subject_id"], d["hadm_id"], d["note_id"])))
        common_keys = set.intersection(*keys_sets)

        print(f"Common notes across 3 annotators: {len(common_keys)}")

        # merge on keys (inner merge on common only)
        def filter_common(d):
            mask = list(zip(d["subject_id"], d["hadm_id"], d["note_id"]))
            return d[[k in common_keys for k in mask]].copy()

        dD = filter_common(dfs["Diego"])
        dN = filter_common(dfs["Daniel"])
        dP = filter_common(dfs["Pankaj"])

        # merge Diego with Daniel
        merged = pd.merge(
            dD,
            dN,
            on=ID_COLS,
            how="inner",
            suffixes=("", "_danieltmp"),
        )
        # add Pankaj
        merged = pd.merge(
            merged,
            dP,
            on=ID_COLS,
            how="inner",
            suffixes=("", "_pankajtmp"),
        )

        # keep one copy of model cols (prefer Diego file, but should match)
        # columns present: m1_pred, m2_pred, m3_pred, combo from Diego sheet.
        # if missing there, attempt from Daniel/Pankaj.
        for c in MODEL_COLS:
            if merged[c].isna().all():
                # try take from Daniel temp columns if exist
                alt = f"{c}_danieltmp"
                if alt in merged.columns:
                    merged[c] = merged[alt]
                alt2 = f"{c}_pankajtmp"
                if merged[c].isna().all() and alt2 in merged.columns:
                    merged[c] = merged[alt2]

        # create LLM stratum based on model preds
        merged["llm_stratum"] = merged.apply(
            lambda r: model_stratum(r["m1_pred"], r["m2_pred"], r["m3_pred"]), axis=1
        )

        # build gold (majority vote across humans)
        gold = []
        gold_status = []
        bad_multi = 0
        missing_any = 0
        no_majority = 0

        for _, r in merged.iterrows():
            labs = [r["human_Diego"], r["human_Daniel"], r["human_Pankaj"]]
            # count multi/missing for tracking
            if "__MULTI__" in labs:
                bad_multi += 1
            if any(l is None for l in labs):
                missing_any += 1

            g, status = majority_vote(labs)
            gold.append(g)
            gold_status.append(status)
            if status == "no_majority":
                no_majority += 1

        merged["gold_label"] = gold
        merged["gold_status"] = gold_status

        # IAA computations (only where all 3 have single labels)
        iaa_mask = (
            merged["human_Diego"].notna() &
            merged["human_Daniel"].notna() &
            merged["human_Pankaj"].notna() &
            (merged["human_Diego"] != "__MULTI__") &
            (merged["human_Daniel"] != "__MULTI__") &
            (merged["human_Pankaj"] != "__MULTI__")
        )
        ratings = merged.loc[iaa_mask, ["human_Diego", "human_Daniel", "human_Pankaj"]].values.tolist()

        pct_all3_agree = np.nan
        if len(ratings) > 0:
            pct_all3_agree = float(np.mean([1.0 if (r[0] == r[1] == r[2]) else 0.0 for r in ratings]))

        fk = fleiss_kappa(ratings, categories=label_space)
        ka = krippendorff_alpha_nominal(ratings)

        # Evaluation set for model metrics: rows with majority gold
        eval_df = merged[merged["gold_status"] == "majority"].copy()

        # compute ensembles
        eval_df["llm_maj_vote"] = eval_df.apply(
            lambda r: llm_majority_vote(r["m1_pred"], r["m2_pred"], r["m3_pred"]), axis=1
        )
        eval_df["unanimous_pred"] = eval_df.apply(
            lambda r: r["m1_pred"] if (r["m1_pred"] == r["m2_pred"] == r["m3_pred"]) else "__REJECT__", axis=1
        )

        # labels order for reporting (fixed)
        labels_order = label_space

        # model metrics
        metrics_out = {}
        for name, col in {
            "m1": "m1_pred",
            "m2": "m2_pred",
            "m3": "m3_pred",
            "llm_maj_vote": "llm_maj_vote",
        }.items():
            metrics_out[name] = compute_model_metrics(
                eval_df["gold_label"].tolist(),
                eval_df[col].tolist(),
                labels_order=labels_order
            )

        # unanimity precision/coverage
        accepted = eval_df[eval_df["unanimous_pred"] != "__REJECT__"].copy()
        unanimity_precision = np.nan
        unanimity_coverage = 0.0
        if len(eval_df):
            unanimity_coverage = len(accepted) / len(eval_df)
            if len(accepted):
                unanimity_precision = float(np.mean(accepted["unanimous_pred"] == accepted["gold_label"]))

        # confusion matrices
        cm_dir = os.path.join(OUTDIR, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)
        if SKLEARN_OK and len(eval_df):
            for name, col in {
                "m1": "m1_pred",
                "m2": "m2_pred",
                "m3": "m3_pred",
                "llm_maj_vote": "llm_maj_vote",
            }.items():
                cm = compute_confusion(eval_df["gold_label"].tolist(), eval_df[col].tolist(), labels_order)
                if cm is not None:
                    cm.to_csv(os.path.join(cm_dir, f"{sheet_name}__{name}_cm.csv"))

        # stratum summary
        stratum_cols = {
            "m1": "m1_pred",
            "m2": "m2_pred",
            "m3": "m3_pred",
            "llm_maj_vote": "llm_maj_vote",
            "unanimous_pred": "unanimous_pred",
        }
        stratum_df = compute_stratum_summary(eval_df, gold_col="gold_label", pred_cols=stratum_cols)
        stratum_df.to_csv(os.path.join(OUTDIR, f"{sheet_name}__stratum_summary.csv"), index=False)

        # save merged + eval datasets
        merged.to_csv(os.path.join(OUTDIR, f"{sheet_name}__merged_all_common.csv"), index=False)
        eval_df.to_csv(os.path.join(OUTDIR, f"{sheet_name}__eval_majority_gold.csv"), index=False)

        # save metrics json per sheet
        with open(os.path.join(OUTDIR, f"{sheet_name}__metrics.json"), "w") as f:
            json.dump({
                "sheet": sheet_name,
                "cat_key": cat_key,
                "n_common_notes": int(len(merged)),
                "n_iaa_items": int(len(ratings)),
                "pct_all3_agree": pct_all3_agree,
                "fleiss_kappa": fk,
                "krippendorff_alpha_nominal": ka,
                "n_majority_gold": int(len(eval_df)),
                "n_no_majority_gold": int(no_majority),
                "n_multi_marked": int(bad_multi),
                "n_missing_any": int(missing_any),
                "unanimity_coverage_on_eval": unanimity_coverage,
                "unanimity_precision_on_eval": unanimity_precision,
                "metrics": metrics_out,
                "gold_label_counts_eval": safe_value_counts(eval_df["gold_label"]),
                "llm_stratum_counts_eval": safe_value_counts(eval_df["llm_stratum"]),
            }, f, indent=2)

        # summary row
        summary_rows.append({
            "sheet": sheet_name,
            "cat_key": cat_key,
            "n_common_notes": int(len(merged)),
            "n_iaa_items": int(len(ratings)),
            "pct_all3_agree": pct_all3_agree,
            "fleiss_kappa": fk,
            "krippendorff_alpha_nominal": ka,
            "n_majority_gold": int(len(eval_df)),
            "n_no_majority_gold": int(no_majority),
            "n_multi_marked": int(bad_multi),
            "n_missing_any": int(missing_any),
            "unanimity_coverage": unanimity_coverage,
            "unanimity_precision": unanimity_precision,
            "m1_acc": metrics_out.get("m1", {}).get("accuracy", np.nan),
            "m2_acc": metrics_out.get("m2", {}).get("accuracy", np.nan),
            "m3_acc": metrics_out.get("m3", {}).get("accuracy", np.nan),
            "majvote_acc": metrics_out.get("llm_maj_vote", {}).get("accuracy", np.nan),
            "m1_macro_f1": metrics_out.get("m1", {}).get("macro_f1", np.nan),
            "m2_macro_f1": metrics_out.get("m2", {}).get("macro_f1", np.nan),
            "m3_macro_f1": metrics_out.get("m3", {}).get("macro_f1", np.nan),
            "majvote_macro_f1": metrics_out.get("llm_maj_vote", {}).get("macro_f1", np.nan),
        })

        print(f"Saved outputs for {sheet_name} in {OUTDIR}/")

    # overall summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTDIR, "SUMMARY_all_sheets.csv"), index=False)
    print("\n✅ Done. Summary saved to:", os.path.join(OUTDIR, "SUMMARY_all_sheets.csv"))
    if not SKLEARN_OK:
        print("⚠ sklearn not available. Install scikit-learn to get full metrics tables.")


if __name__ == "__main__":
    main()
