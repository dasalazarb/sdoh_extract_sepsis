#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
code_cox_survival_model_Ep_v2.py

Cox PH survival models using:
  - Baseline clinical covariates (X)
  - Cleaner SDOH documentation-state indicators E (patient-level; mutually exclusive)
  - Two-component SDOH signal per domain:
      * adversity strength
      * decision confidence
  - Minimal, vital sensitivity analyses

Key decisions:
  1) Use the aggregated domain column (e.g., employment_status) to derive E states.
  2) Do NOT merge SDOH by hadm_id (pre-ICU hadm_id may not match index hadm_id).
     Instead, build patient-level SDOH and merge into stay-level cohort by subject_id.
  3) Note ordering: use the LAST note_id (or last numeric component of note_id if alphanumeric).
  4) Domains included (binary adversity only):
       employment_status, housing_issues, transportation_issues, social_support
     relationship_status and parental_status are excluded from E/p in this script.

E definition (note-level -> patient-level, mutually exclusive):
  note state:
      observed_adverse      = valid aggregated label in adverse set
      observed_no_adverse   = valid non-adverse label OR not_mentioned
      uncertain             = unsure
      no_evidence           = unknown / invalid / empty
  patient state priority across notes:
      observed_adverse > observed_no_adverse > uncertain > no_evidence

p replacement (note-level, from m1/m2/m3):
  adversity_strength_note = (# adverse votes) / 3
  decision_conf_note      = (# decisive votes) / 3
  where decisive = {valid label, not_mentioned}
  and unsure/unknown/invalid contribute 0 to decisiveness.

Aggregation:
  Primary: adversity_strength_max + decision_conf_mean
  Sensitivity 1: adversity_strength_mean + decision_conf_mean
  Sensitivity 2: adversity_strength_max_hiconf + decision_conf_mean
                 (high-confidence note = decision_conf_note >= 2/3)
  Sensitivity 3: vote_max + decision_conf_mean

Robustness:
  - Handles alphanumeric note_id by creating note_order
  - Ridge penalization + automatic dropping of constant/ultra-rare/separation-like columns
  - Robust SE clustered by subject_key (string)

Outputs (prefix OUT_PREFIX):
  - *__sdoh_patient_features.csv
  - *__sanity_*.csv
  - *__summary.csv, *__ph_test.csv, *__fit_meta.json
  - *__dropped_cols.csv (if any)

"""

# ---------- Python 3.10 compatibility patch for lifelines (datetime.UTC is 3.11+) ----------
import datetime
from datetime import timezone
if not hasattr(datetime, "UTC"):
    datetime.UTC = timezone.utc

import json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


# =========================
# CONFIG (EDIT PATHS)
# =========================
SURV_PATH = "model_ready_sepsis_survival_stayid_plusSOFA.csv"
SDOH_PATH = "sdoh_all_notes_with_llm_strata.csv"   # or sdoh_all_notes_with_llm_strata_EDITED.csv

OUT_PREFIX = "cox_sdoh_Ep_v2"

TIME0_COL = "icu_intime"
EVENT_COL = "death_30d"
DOD_COL = "dod_adjusted"
HORIZON_DAYS = 30.0

DOMAINS = ["employment_status", "housing_issues", "transportation_issues", "social_support"]

# Baseline covariates (kept only if present)
CAT_COLS = ["gender", "race", "admission_type", "icu_type"]
NUM_COLS = [
    "age_at_icu_intime",
    "charlson_12m_prior_with_age",
    "sofa_24hours",
    # optional labs
    "lactate", "creatinine", "bun", "bilirubin_total", "platelets", "wbc", "pao2",
    # doc intensity created below
    "docint_log",
]
TOPK_CATS = 12

# Cox stability
PENALIZER_PRIMARY = 1.0
PENALIZER_FALLBACK = 5.0
LOWVAR_EPS = 1e-10
CONDVAR_EPS = 1e-10
BIN_PREV_MIN = 0.001         # drop binary cols with prevalence <0.1% or >99.9%
MIN_NONMISS_STRATUM = 8      # for conditional variance checks

# Label tokens
UNKNOWN_LABELS = {"unknown", "", "nan", "null"}   # treat any other non-valid/non-special token as unknown too
UNSURE_LABEL = "unsure"
NM_LABEL = "not_mentioned"

# Valid labels per domain (your vocabulary)
VALID_LABELS = {
    "employment_status": {"employed", "underemployed", "unemployed", "disability", "retired", "student"},
    "housing_issues": {"financial_status", "undomiciled", "other"},
    "transportation_issues": {"distance", "resources", "other"},
    "social_support": {"plus", "minus"},
}

# Adverse labels per domain (binary adversity)
ADVERSE_LABELS = {
    "employment_status": {"unemployed", "underemployed", "disability"},
    "social_support": {"minus"},
    "housing_issues": VALID_LABELS["housing_issues"],                 # any valid implies issue present
    "transportation_issues": VALID_LABELS["transportation_issues"],   # any valid implies issue present
}


def sdoh_usecols(col: str) -> bool:
    base = {"subject_id", "note_id"}
    if col in base:
        return True
    for d in DOMAINS:
        if col in {d, f"{d}__m1", f"{d}__m2", f"{d}__m3", f"{d}__agree_n"}:
            return True
    return False


# =========================
# Helpers
# =========================
def to_dt(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def cap_categories(s: pd.Series, topk=12, other="OTHER", unknown="UNKNOWN") -> pd.Series:
    s = s.astype("object")
    s = s.where(~s.isna(), unknown)
    vc = s.value_counts(dropna=False)
    keep = set(vc.head(topk).index.tolist())
    return s.where(s.isin(keep), other)



def norm_cat(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = " ".join(s.split())
    return s

def collapse_race(x):
    s = norm_cat(x)
    if s == "" or s in {"unknown", "unable to obtain", "patient declined to answer", "declined", "na", "n a", "other unknown", "unobtainable", "not specified"}:
        return "Unknown"
    if "hispanic" in s or "latino" in s or "latina" in s:
        return "Hispanic-Latino"
    if "black" in s or "african" in s:
        return "Black"
    if "asian" in s:
        return "Asian"
    if "white" in s or "caucasian" in s:
        return "White"
    return "Other"

def collapse_admission_type(x):
    s = norm_cat(x)
    if s == "" or s in {"unknown", "na", "n a"}:
        return "Other"
    if "observation" in s:
        return "Observation"
    if "urgent" in s:
        return "Urgent"
    if "elective" in s or "scheduled" in s:
        return "Elective-Scheduled"
    if "emerg" in s or s == "ew emer." or s.startswith("ew emer"):
        return "Emergency"
    return "Other"

def collapse_icu_type(x):
    s = norm_cat(x)
    if s == "":
        return "Other"
    if "micu sicu" in s or "sicu micu" in s:
        return "MICU-SICU"
    if "cvicu" in s or "csru" in s or "cardiac" in s:
        return "CVICU"
    if "tsicu" in s or "trauma" in s:
        return "TSICU"
    if s == "ccu" or "coronary" in s:
        return "CCU"
    if "neuro" in s or "nicu" in s or "neuroscience" in s:
        return "Neuro"
    if s == "micu" or ("medical" in s and "icu" in s):
        return "MICU"
    if s == "sicu" or ("surgical" in s and "icu" in s):
        return "SICU"
    return "Other"

def apply_clinical_category_collapses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "race" in df.columns:
        df["race"] = df["race"].map(collapse_race)
    if "admission_type" in df.columns:
        df["admission_type"] = df["admission_type"].map(collapse_admission_type)
    if "icu_type" in df.columns:
        df["icu_type"] = df["icu_type"].map(collapse_icu_type)
    if "index_icu_type" in df.columns:
        df["index_icu_type"] = df["index_icu_type"].map(collapse_icu_type)
    return df

def norm_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def note_E_state(domain: str, x) -> str:
    """
    Map aggregated domain label to a mutually exclusive note-level E state:
      - observed_adverse
      - observed_no_adverse
      - uncertain
      - no_evidence
    """
    s = norm_label(x)
    if s in VALID_LABELS[domain]:
        if s in ADVERSE_LABELS[domain]:
            return "observed_adverse"
        return "observed_no_adverse"
    if s == NM_LABEL:
        return "observed_no_adverse"
    if s == UNSURE_LABEL:
        return "uncertain"
    return "no_evidence"

def model_decisive_and_adverse(domain: str, x):
    """
    For a per-model output:
      decisive = valid OR not_mentioned
      adverse  = valid AND in ADVERSE_LABELS
      unsure/unknown/invalid -> not decisive
    """
    s = norm_label(x)
    if s in VALID_LABELS[domain]:
        return 1, (1 if s in ADVERSE_LABELS[domain] else 0), "valid"
    if s == NM_LABEL:
        return 1, 0, "nm"
    if s == UNSURE_LABEL:
        return 0, 0, "unsure"
    return 0, 0, "unknown"

def build_time_event(df: pd.DataFrame) -> pd.DataFrame:
    """30-day horizon survival time from ICU admit."""
    df = df.copy()
    df["event"] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype(int)

    t0 = pd.to_datetime(df[TIME0_COL], errors="coerce")
    dod = pd.to_datetime(df[DOD_COL], errors="coerce")

    delta_days = (dod - t0).dt.total_seconds() / 86400.0
    df["time_days"] = np.where(df["event"] == 1, delta_days, HORIZON_DAYS)

    df.loc[df["time_days"].isna(), "time_days"] = HORIZON_DAYS
    df.loc[df["time_days"] < 0, "time_days"] = 0.0
    df.loc[df["time_days"] > HORIZON_DAYS, "time_days"] = HORIZON_DAYS

    df["event_with_missing_dod"] = ((df["event"] == 1) & dod.isna()).astype(int)
    return df

def add_missing_indicators_and_impute(df: pd.DataFrame, cols):
    """Median-impute numeric covariates; create missing indicator only if 0<missing<1."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        miss_frac = df[c].isna().mean()
        if 0 < miss_frac < 1:
            df[f"{c}__missing"] = df[c].isna().astype(int)
        med = pd.to_numeric(df[c], errors="coerce").median(skipna=True)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)
    return df

def drop_problem_columns(X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Drop columns that commonly break Cox fitting (constant/lowvar/ultra-rare/separation-like)."""
    dropped = []
    keep_always = {"subject_key", "time_days", "event"}
    cols = [c for c in X.columns if c not in keep_always]
    e = X["event"].astype(int)

    for c in cols:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() == 0:
            dropped.append((c, "all_missing"))
            continue
        nunq = s.nunique(dropna=True)
        if nunq <= 1:
            dropped.append((c, "constant"))
            continue
        v = s.var(skipna=True)
        if pd.isna(v) or v < LOWVAR_EPS:
            dropped.append((c, "lowvar"))
            continue

        if nunq == 2:
            p = float(s.mean(skipna=True))
            if (p < BIN_PREV_MIN) or (p > 1 - BIN_PREV_MIN):
                dropped.append((c, "binary_ultra_rare"))
                continue

        s0 = s[e == 0]
        s1 = s[e == 1]
        if s0.notna().sum() >= MIN_NONMISS_STRATUM and s1.notna().sum() >= MIN_NONMISS_STRATUM:
            v0 = s0.var(skipna=True)
            v1 = s1.var(skipna=True)
            if (not pd.isna(v0)) and (not pd.isna(v1)) and (min(v0, v1) < CONDVAR_EPS):
                dropped.append((c, "cond_lowvar_separation_like"))
                continue

    if dropped:
        pd.DataFrame(dropped, columns=["column", "reason"]).to_csv(
            f"{OUT_PREFIX}__{model_name}__dropped_cols.csv", index=False
        )

    drop_cols = [c for c, _ in dropped]
    return X.drop(columns=drop_cols, errors="ignore")

def fit_cox(X: pd.DataFrame, model_name: str):
    """Fit CoxPH with ridge + robust clustered SE. Retries with stronger penalizer if needed."""
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    for c in X.columns:
        if c != "subject_key":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    X = drop_problem_columns(X, model_name=model_name)

    X.to_csv(f"{OUT_PREFIX}__{model_name}__design_snapshot.csv", index=False)

    for pen in [PENALIZER_PRIMARY, PENALIZER_FALLBACK]:
        try:
            cph = CoxPHFitter(penalizer=pen, l1_ratio=0.0)
            cph.fit(
                X,
                duration_col="time_days",
                event_col="event",
                robust=True,
                cluster_col="subject_key",
            )

            summ = cph.summary.copy()
            summ["HR"] = np.exp(summ["coef"])
            summ["HR_lower_95"] = np.exp(summ["coef lower 95%"])
            summ["HR_upper_95"] = np.exp(summ["coef upper 95%"])
            summ = summ.sort_values("p")
            summ.to_csv(f"{OUT_PREFIX}__{model_name}__summary.csv")

            ph = proportional_hazard_test(cph, X, time_transform="rank").summary.sort_values("p")
            ph.to_csv(f"{OUT_PREFIX}__{model_name}__ph_test.csv")

            meta = {
                "penalizer_used": pen,
                "n_rows": int(X.shape[0]),
                "n_cols": int(X.shape[1]),
                "concordance_index_": float(getattr(cph, "concordance_index_", np.nan)),
            }
            with open(f"{OUT_PREFIX}__{model_name}__fit_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            return summ, ph, meta

        except Exception as ex:
            print(f"[WARN] Cox fit failed with penalizer={pen}: {type(ex).__name__}: {ex}")

    raise RuntimeError("Cox model failed even after fallback penalizer.")


# =========================
# SDOH patient-level features
# =========================
def add_note_order(sdoh: pd.DataFrame) -> pd.DataFrame:
    """
    Create note_order to implement "last note_id wins" even when note_id is alphanumeric.
    Strategy:
      1) numeric(note_id)
      2) extract last integer from note_id string
      3) fallback: row order within subject
    """
    sdoh = sdoh.copy()
    sdoh["note_id_raw"] = sdoh["note_id"].astype(str)

    # try direct numeric
    sdoh["note_order"] = pd.to_numeric(sdoh["note_id"], errors="coerce")

    # extract last integer group if needed
    mask = sdoh["note_order"].isna()
    if mask.any():
        extracted = sdoh.loc[mask, "note_id_raw"].str.extract(r"(\\d+)(?!.*\\d)", expand=False)
        sdoh.loc[mask, "note_order"] = pd.to_numeric(extracted, errors="coerce")

    # fallback: stable row order within subject
    mask2 = sdoh["note_order"].isna()
    if mask2.any():
        sdoh = sdoh.sort_values(["subject_key"]).copy()
        sdoh.loc[mask2, "note_order"] = sdoh.groupby("subject_key").cumcount() + 1

    sdoh["note_order"] = sdoh["note_order"].astype(int)

    # sanity output
    out = sdoh[["subject_key", "note_id_raw", "note_order"]].head(2000)
    out.to_csv(f"{OUT_PREFIX}__sanity_note_order_head.csv", index=False)
    return sdoh

def build_sdoh_patient_features(sdoh: pd.DataFrame) -> pd.DataFrame:
    """
    Build patient-level SDOH features from note-level file:
      - E state per domain: mutually exclusive patient category
          {no_evidence, observed_no_adverse, observed_adverse, uncertain}
        with no_evidence as the reference level in modeling
      - Adversity strength per domain (patient-level)
      - Decision confidence per domain (patient-level)
      - vote_max retained as a minimal sensitivity analysis
    """
    sdoh = sdoh.copy()

    # subject_key for robust merging
    sdoh["subject_key"] = sdoh["subject_id"].astype(str)

    # note ordering
    sdoh = add_note_order(sdoh)

    # de-dup by (subject_key, note_order)
    before = len(sdoh)
    sdoh = sdoh.sort_values(["subject_key", "note_order"]).drop_duplicates(["subject_key", "note_order"], keep="last")
    after = len(sdoh)
    print(f"[SDOH] dedup by (subject_key,note_order): {before} -> {after}")

    # notes count
    notes_n = sdoh.groupby("subject_key", as_index=False)["note_order"].count().rename(columns={"note_order": "sdoh_notes_n"})

    # --- E states from aggregated domain columns ---
    # Patient state priority: observed_adverse > observed_no_adverse > uncertain > no_evidence
    E_priority = {"no_evidence": 0, "uncertain": 1, "observed_no_adverse": 2, "observed_adverse": 3}
    E_state_counts = []
    E_parts = [notes_n]

    for d in DOMAINS:
        if d not in sdoh.columns:
            raise ValueError(f"Missing aggregated domain column: {d}")

        sdoh[f"{d}__E_note_state"] = sdoh[d].map(lambda x: note_E_state(d, x))

        # note-level sanity counts
        vc = sdoh[f"{d}__E_note_state"].value_counts(dropna=False).to_dict()
        row = {"domain": d}
        for k in ["no_evidence", "uncertain", "observed_no_adverse", "observed_adverse"]:
            row[f"n_{k}"] = int(vc.get(k, 0))
        E_state_counts.append(row)

        tmp = sdoh[["subject_key", f"{d}__E_note_state"]].copy()
        tmp[f"{d}__E_priority"] = tmp[f"{d}__E_note_state"].map(E_priority).fillna(0).astype(int)

        pat = tmp.groupby("subject_key", as_index=False)[f"{d}__E_priority"].max()
        inv_priority = {v: k for k, v in E_priority.items()}
        pat[f"{d}__E_state"] = pat[f"{d}__E_priority"].map(inv_priority)

        # one-hot with no_evidence as reference level
        pat[f"{d}__E_observed_no_adverse"] = (pat[f"{d}__E_state"] == "observed_no_adverse").astype(int)
        pat[f"{d}__E_observed_adverse"] = (pat[f"{d}__E_state"] == "observed_adverse").astype(int)
        pat[f"{d}__E_uncertain"] = (pat[f"{d}__E_state"] == "uncertain").astype(int)

        E_parts.append(
            pat[[
                "subject_key",
                f"{d}__E_state",
                f"{d}__E_observed_no_adverse",
                f"{d}__E_observed_adverse",
                f"{d}__E_uncertain",
            ]]
        )

        # last valid aggregated label (debug)
        valid_mask = sdoh[d].map(lambda x: norm_label(x) in VALID_LABELS[d])
        tmpv = sdoh.loc[valid_mask, ["subject_key", "note_order", d]].copy()
        if len(tmpv):
            last_valid = (tmpv.sort_values(["subject_key", "note_order"])
                             .groupby("subject_key", as_index=False)
                             .tail(1)
                             .rename(columns={d: f"{d}__last_valid_label"})
                             [["subject_key", f"{d}__last_valid_label"]])
        else:
            last_valid = pd.DataFrame({"subject_key": sdoh["subject_key"].unique()})
            last_valid[f"{d}__last_valid_label"] = "unknown"
        E_parts.append(last_valid)

    pd.DataFrame(E_state_counts).to_csv(f"{OUT_PREFIX}__sanity_sdoh_E_note_state_counts.csv", index=False)

    # --- Two-component SDOH signal from m1/m2/m3 (note-level) ---
    # adversity_strength_note = adverse votes / 3
    # decision_conf_note      = decisive votes / 3
    # with decisive = valid OR not_mentioned
    for d in DOMAINS:
        for m in ["m1", "m2", "m3"]:
            col = f"{d}__{m}"
            if col not in sdoh.columns:
                raise ValueError(f"Missing per-model column: {col}")

        dec = []
        adv = []
        for m in ["m1", "m2", "m3"]:
            decisive, adverse, _ = zip(*sdoh[f"{d}__{m}"].map(lambda x: model_decisive_and_adverse(d, x)))
            dec.append(pd.Series(decisive, index=sdoh.index))
            adv.append(pd.Series(adverse, index=sdoh.index))

        dec_sum = dec[0] + dec[1] + dec[2]
        adv_sum = adv[0] + adv[1] + adv[2]

        sdoh[f"{d}__decision_conf_note"] = dec_sum / 3.0
        sdoh[f"{d}__adv_strength_note"] = adv_sum / 3.0

        # High-confidence notes: at least 2/3 decisive outputs
        sdoh[f"{d}__adv_strength_note_hiconf"] = np.where(
            sdoh[f"{d}__decision_conf_note"] >= (2.0 / 3.0),
            sdoh[f"{d}__adv_strength_note"],
            np.nan,
        )

        # Legacy majority-vote binary sensitivity
        sdoh[f"{d}__vote_note"] = np.where(
            dec_sum == 0, np.nan,
            np.where((dec_sum >= 2) & (adv_sum >= 2), 1.0, 0.0)
        )

    # aggregate to patient
    def agg_patient(col, out, agg="max", fill_value=0.0):
        tmp = sdoh[["subject_key", col]].copy()
        if agg == "max":
            g = tmp.groupby("subject_key", as_index=False)[col].max()
        elif agg == "mean":
            g = tmp.groupby("subject_key", as_index=False)[col].mean()
        else:
            raise ValueError("agg must be max or mean")
        g = g.rename(columns={col: out})
        g[out] = g[out].fillna(fill_value)
        return g

    P_parts = []
    for d in DOMAINS:
        P_parts.append(agg_patient(f"{d}__adv_strength_note", f"{d}__adv_strength_max", agg="max"))
        P_parts.append(agg_patient(f"{d}__adv_strength_note", f"{d}__adv_strength_mean", agg="mean"))
        P_parts.append(agg_patient(f"{d}__adv_strength_note_hiconf", f"{d}__adv_strength_max_hiconf", agg="max", fill_value=0.0))
        P_parts.append(agg_patient(f"{d}__decision_conf_note", f"{d}__decision_conf_max", agg="max"))
        P_parts.append(agg_patient(f"{d}__decision_conf_note", f"{d}__decision_conf_mean", agg="mean"))
        P_parts.append(agg_patient(f"{d}__vote_note", f"{d}__vote_max", agg="max", fill_value=0.0))

    # merge all parts
    feat = E_parts[0]
    for part in E_parts[1:]:
        feat = feat.merge(part, on="subject_key", how="left", validate="one_to_one")
    for part in P_parts:
        feat = feat.merge(part, on="subject_key", how="left", validate="one_to_one")

    # sanity: patient-level coverage summary
    cov = []
    for d in DOMAINS:
        for state in ["observed_no_adverse", "observed_adverse", "uncertain"]:
            cov.append({"domain": d, "var": f"E_{state}", "mean": float(feat[f"{d}__E_{state}"].mean())})
        cov += [
            {"domain": d, "var": "adv_strength_max_mean", "mean": float(feat[f"{d}__adv_strength_max"].mean())},
            {"domain": d, "var": "adv_strength_mean_mean", "mean": float(feat[f"{d}__adv_strength_mean"].mean())},
            {"domain": d, "var": "decision_conf_mean_mean", "mean": float(feat[f"{d}__decision_conf_mean"].mean())},
            {"domain": d, "var": "decision_conf_max_mean", "mean": float(feat[f"{d}__decision_conf_max"].mean())},
        ]
    pd.DataFrame(cov).to_csv(f"{OUT_PREFIX}__sanity_sdoh_patient_cov.csv", index=False)

    feat.to_csv(f"{OUT_PREFIX}__sdoh_patient_features.csv", index=False)
    print(f"[SDOH] wrote {OUT_PREFIX}__sdoh_patient_features.csv (n={len(feat)})")
    return feat


# =========================
# Main
# =========================
def main():
    print("=== Cox SDOH (clean E + strength/confidence) ===")
    print("SURV_PATH:", SURV_PATH)
    print("SDOH_PATH:", SDOH_PATH)
    print("DOMAINS:", DOMAINS)

    # ---- Load survival ----
    surv = pd.read_csv(SURV_PATH, low_memory=False)
    surv = to_dt(surv, [TIME0_COL, DOD_COL])

    # subject_key for merge/cluster
    assert "subject_id" in surv.columns, "survival df must contain subject_id"
    surv["subject_key"] = surv["subject_id"].astype(str)

    # Stay sanity
    assert {"subject_key", "stay_id"}.issubset(surv.columns), "survival df missing subject_id or stay_id"
    assert not surv.duplicated(["subject_key", "stay_id"]).any(), "Duplicate (subject_id, stay_id) in survival df"

    print("[SURV] shape:", surv.shape)
    print("[SURV] unique stays:", int(surv[["subject_key", "stay_id"]].drop_duplicates().shape[0]))

    # outcome
    surv = build_time_event(surv)
    print("[OUTCOME] death_30d:", surv[EVENT_COL].value_counts(dropna=False).to_dict())
    print("[OUTCOME] event==1 but missing dod_adjusted:", int(surv["event_with_missing_dod"].sum()))

    # DocInt
    if "n_notes_preicu" in surv.columns:
        surv["docint_log"] = np.log1p(pd.to_numeric(surv["n_notes_preicu"], errors="coerce").fillna(0))
        doc_src = "n_notes_preicu"
    elif "n_words_preicu" in surv.columns:
        surv["docint_log"] = np.log1p(pd.to_numeric(surv["n_words_preicu"], errors="coerce").fillna(0))
        doc_src = "n_words_preicu"
    else:
        surv["docint_log"] = 0.0
        doc_src = "none"
    print("[DOCINT] source:", doc_src, "| summary:", surv["docint_log"].describe().to_dict())

    # ---- Load SDOH note-level ----
    sdoh = pd.read_csv(SDOH_PATH, usecols=sdoh_usecols, low_memory=False)
    assert "subject_id" in sdoh.columns and "note_id" in sdoh.columns, "SDOH must have subject_id and note_id"
    sdoh["subject_key"] = sdoh["subject_id"].astype(str)
    print("[SDOH] loaded shape:", sdoh.shape)
    print("[SDOH] unique notes:", int(sdoh[["subject_key", "note_id"]].drop_duplicates().shape[0]))

    # build patient features
    sdoh_pat = build_sdoh_patient_features(sdoh)

    # merge by subject_key only (many stays per subject ok)
    n0 = len(surv)
    df = surv.merge(sdoh_pat, on="subject_key", how="left", validate="many_to_one")
    assert len(df) == n0, "Row count changed after merging SDOH patient features."
    print("[MERGE] df shape:", df.shape)

    # Fill missing SDOH features for patients not present in SDOH file
    for d in DOMAINS:
        state_col = f"{d}__E_state"
        if state_col in df.columns:
            df[state_col] = df[state_col].fillna("no_evidence")
        for e in ["E_observed_no_adverse", "E_observed_adverse", "E_uncertain"]:
            col = f"{d}__{e}"
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        for pcol in [
            f"{d}__adv_strength_max",
            f"{d}__adv_strength_mean",
            f"{d}__adv_strength_max_hiconf",
            f"{d}__decision_conf_max",
            f"{d}__decision_conf_mean",
            f"{d}__vote_max",
        ]:
            if pcol in df.columns:
                df[pcol] = df[pcol].fillna(0.0)

    # baseline covariates
    df = apply_clinical_category_collapses(df)
    for c in CAT_COLS:
        if c not in df.columns:
            continue
        if c in {"race", "admission_type", "icu_type"}:
            # already clinically collapsed above
            df[c] = df[c].fillna("Other")
        else:
            df[c] = cap_categories(df[c], topk=TOPK_CATS)

    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    # save baseline missingness sanity
    base_miss = []
    for c in num_cols + cat_cols:
        base_miss.append({"col": c, "missing_frac": float(pd.to_numeric(df[c], errors="coerce").isna().mean() if c in num_cols else df[c].isna().mean())})
    pd.DataFrame(base_miss).sort_values("missing_frac", ascending=False).to_csv(
        f"{OUT_PREFIX}__sanity_baseline_missingness.csv", index=False
    )

    # ---- Design builder ----
    def make_design(tag: str, strength_kind: str, conf_kind: str = "decision_conf_mean"):
        """
        Build a design matrix with:
          - cleaner mutually exclusive E indicators (reference = no_evidence)
          - one adversity-strength variable per domain
          - one decision-confidence variable per domain

        strength_kind in {"adv_strength_max", "adv_strength_mean", "adv_strength_max_hiconf", "vote_max"}
        conf_kind in {"decision_conf_mean", "decision_conf_max"}
        """
        sdoh_cols = []
        for d in DOMAINS:
            sdoh_cols += [
                f"{d}__E_observed_no_adverse",
                f"{d}__E_observed_adverse",
                f"{d}__E_uncertain",
                f"{d}__{strength_kind}",
                f"{d}__{conf_kind}",
            ]
        sdoh_cols = [c for c in sdoh_cols if c in df.columns]

        keep = ["subject_key", "time_days", "event"] + num_cols + cat_cols + sdoh_cols
        X = df[keep].copy()

        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        X = add_missing_indicators_and_impute(X, num_cols)

        for c in X.columns:
            if c != "subject_key":
                X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        # robust finite check (avoid object dtype issues)
        arr = X.drop(columns=["subject_key"]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
        nonfinite = int((~np.isfinite(arr)).sum())
        if nonfinite:
            print(f"[WARN] {nonfinite} non-finite values in design before fit; rows will be dropped.")

        X.to_csv(f"{OUT_PREFIX}__{tag}__design_pre_drop.csv", index=False)
        return X

    # ---- Models (primary + vital sensitivities) ----
    models = [
        ("primary_strengthmax_confmean", "adv_strength_max", "decision_conf_mean"),
        ("sens_strengthmean_confmean", "adv_strength_mean", "decision_conf_mean"),
        ("sens_strengthmax_hiconf_confmean", "adv_strength_max_hiconf", "decision_conf_mean"),
        ("sens_majority_vote_confmean", "vote_max", "decision_conf_mean"),
    ]

    all_meta = []

    for tag, strength_kind, conf_kind in models:
        print(f"\n=== FIT: {tag} | strength={strength_kind} | confidence={conf_kind} ===")
        X = make_design(tag, strength_kind, conf_kind)
        assert {"subject_key", "time_days", "event"}.issubset(X.columns), "Design missing required columns."
        summ, ph, meta = fit_cox(X, model_name=tag)
        all_meta.append({"model": tag, **meta})

        # forced print: SDOH terms
        sdoh_terms = []
        for d in DOMAINS:
            sdoh_terms += [
                f"{d}__{strength_kind}",
                f"{d}__{conf_kind}",
                f"{d}__E_observed_no_adverse",
                f"{d}__E_observed_adverse",
                f"{d}__E_uncertain",
            ]
        sdoh_terms = [t for t in sdoh_terms if t in summ.index]

        print("\\n[TOP 12 overall]")
        print(summ.head(12)[["coef", "HR", "HR_lower_95", "HR_upper_95", "p"]])

        print("\\n[SDOH terms (forced)]")
        if sdoh_terms:
            print(summ.loc[sdoh_terms, ["coef", "HR", "HR_lower_95", "HR_upper_95", "p"]].sort_values("p").head(60))
        else:
            print("  (none; likely dropped. Check dropped_cols csv.)")

        # PH violations quick summary
        n_ph = int((ph["p"] < 0.05).sum()) if "p" in ph.columns else -1
        ph.head(50).to_csv(f"{OUT_PREFIX}__{tag}__ph_top50.csv")
        print(f"[PH] #covariates with p<0.05: {n_ph} | wrote {OUT_PREFIX}__{tag}__ph_top50.csv")

    pd.DataFrame(all_meta).to_csv(f"{OUT_PREFIX}__all_models_meta.csv", index=False)

    print("\\n✅ DONE. Outputs:")
    print(" -", f"{OUT_PREFIX}__sdoh_patient_features.csv")
    print(" -", f"{OUT_PREFIX}__sanity_*.csv")
    print(" -", f"{OUT_PREFIX}__*__summary.csv")
    print(" -", f"{OUT_PREFIX}__*__ph_test.csv")
    print(" -", f"{OUT_PREFIX}__*__dropped_cols.csv (if any)")


if __name__ == "__main__":
    main()
