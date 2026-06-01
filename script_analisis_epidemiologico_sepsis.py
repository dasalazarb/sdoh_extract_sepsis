#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis epidemiologico/descriptivo para la cohorte de sepsis.

Este script genera un workbook .xlsx listo para manuscrito con:
- Tabla 1 descriptiva global y estratificada por muerte a 30 dias.
- Una fila por cada nivel de las variables categoricas, no solo la categoria mas frecuente.
- Detalle de muertes, censura y tiempo de seguimiento a 30 dias.
- Perfil de datos faltantes.
- Chequeo del supuesto de riesgos proporcionales con residuos de Schoenfeld.
- Revision practica de posible leakage temporal de las notas SDOH.

Variables alineadas con:
- code_build_sdoh_all_notes_with_llm_strata_from_extracted.py
- code_cox_survival_model.py
"""

import argparse
import datetime
from datetime import timezone

if not hasattr(datetime, "UTC"):
    datetime.UTC = timezone.utc

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

HORIZON_DAYS = 30.0
TIME0_COL = "icu_intime"
DOD_COL = "dod_adjusted"
DEFAULT_EVENT_COL = "death_30d"
DEFAULT_TIME_COL = "time_days"

DOMAINS = ["employment_status", "housing_issues", "transportation_issues", "social_support"]
CAT_COLS = ["gender", "race", "admission_type", "icu_type"]
NUM_COLS = [
    "age_at_icu_intime",
    "charlson_12m_prior_with_age",
    "sofa_24hours",
    "lactate",
    "creatinine",
    "bun",
    "bilirubin_total",
    "platelets",
    "wbc",
    "pao2",
    "docint_log",
]
OPTIONAL_DESCRIPTIVE_NUM_COLS = [
    "anchor_age",
    "age_at_admit",
    "n_prior_hadm_12m",
    "pao2fio2ratio_novent_min_24h",
    "pao2fio2ratio_vent_min_24h",
    "rate_epinephrine_max_24h",
    "rate_norepinephrine_max_24h",
    "rate_dopamine_max_24h",
    "rate_dobutamine_max_24h",
    "meanbp_min_24h",
    "gcs_min_24h",
    "uo_24hr_24h",
    "respiration_24hours",
    "coagulation_24hours",
    "liver_24hours",
    "cardiovascular_24hours",
    "cns_24hours",
    "renal_24hours",
    "n_notes_preicu",
    "n_chars_preicu",
    "n_words_preicu",
]

UNKNOWN_LABELS = {"unknown", "", "nan", "null"}
UNSURE_LABEL = "unsure"
NM_LABEL = "not_mentioned"
VALID_LABELS = {
    "employment_status": {"employed", "underemployed", "unemployed", "disability", "retired", "student"},
    "housing_issues": {"financial_status", "undomiciled", "other"},
    "transportation_issues": {"distance", "resources", "other"},
    "social_support": {"plus", "minus"},
}
ADVERSE_LABELS = {
    "employment_status": {"unemployed", "underemployed", "disability"},
    "social_support": {"minus"},
    "housing_issues": VALID_LABELS["housing_issues"],
    "transportation_issues": VALID_LABELS["transportation_issues"],
}
E_PRIORITY = {
    "no_evidence": 0,
    "uncertain": 1,
    "observed_no_adverse": 2,
    "observed_adverse": 3,
}
E_LABELS = {
    "observed_adverse": "Observed adverse SDOH",
    "observed_no_adverse": "Observed no adverse SDOH",
    "uncertain": "Uncertain",
    "no_evidence": "No evidence / not documented",
}
YES_NO_LABELS = {0: "No", 1: "Yes"}


def norm_cat(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    return " ".join(s.split())


def collapse_race(x):
    s = norm_cat(x)
    unknown_tokens = {
        "",
        "unknown",
        "unable to obtain",
        "patient declined to answer",
        "declined",
        "na",
        "n a",
        "other unknown",
        "unobtainable",
        "not specified",
    }
    if s in unknown_tokens:
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


def note_e_state(domain: str, x) -> str:
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
    s = norm_label(x)
    if s in VALID_LABELS[domain]:
        return 1, int(s in ADVERSE_LABELS[domain])
    if s == NM_LABEL:
        return 1, 0
    return 0, 0


def format_n_pct(n, denom) -> str:
    if denom == 0:
        return "0 (0.0%)"
    return f"{int(n)} ({100.0 * n / denom:.1f}%)"


def summarize_num(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return "NA"
    return f"{s.median():.2f} [{s.quantile(0.25):.2f}, {s.quantile(0.75):.2f}]"


def summarize_missing(s: pd.Series) -> str:
    return format_n_pct(int(s.isna().sum()), len(s))


def ordered_levels(s: pd.Series, include_missing=True) -> list:
    sx = s.astype("object").where(s.notna(), "Missing") if include_missing else s.dropna().astype("object")
    vc = sx.value_counts(dropna=False)
    return vc.index.tolist()


def category_cell(s: pd.Series, level) -> str:
    sx = s.astype("object").where(s.notna(), "Missing")
    return format_n_pct(int((sx == level).sum()), len(sx))


def add_numeric_row(rows, label: str, df: pd.DataFrame, g0: pd.DataFrame, g1: pd.DataFrame, col: str):
    rows.append([
        label,
        "Median [IQR]",
        summarize_num(df[col]),
        summarize_num(g0[col]),
        summarize_num(g1[col]),
    ])
    if df[col].isna().any():
        rows.append([
            f"  Missing {label}",
            "n (%)",
            summarize_missing(df[col]),
            summarize_missing(g0[col]),
            summarize_missing(g1[col]),
        ])


def add_categorical_rows(rows, label: str, df: pd.DataFrame, g0: pd.DataFrame, g1: pd.DataFrame, col: str):
    rows.append([label, "n (%)", "", "", ""])
    for level in ordered_levels(df[col], include_missing=df[col].isna().any()):
        rows.append([
            f"  {level}",
            "n (%)",
            category_cell(df[col], level),
            category_cell(g0[col], level),
            category_cell(g1[col], level),
        ])


def make_sdoh_patient_features(sdoh: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "note_id"}
    missing_required = required - set(sdoh.columns)
    if missing_required:
        raise ValueError(f"El archivo SDOH debe incluir columnas {sorted(missing_required)}")

    sdoh = sdoh.copy()
    sdoh["subject_key"] = sdoh["subject_id"].astype(str)
    out = sdoh[["subject_key", "note_id"]].copy()

    if "note_charttime" in sdoh.columns and "icu_intime" in sdoh.columns:
        t_note = pd.to_datetime(sdoh["note_charttime"], errors="coerce")
        t_icu = pd.to_datetime(sdoh["icu_intime"], errors="coerce")
        out["potential_temporal_leak_note"] = (t_note > t_icu).astype("Int64")
    elif "charttime" in sdoh.columns and "icu_intime" in sdoh.columns:
        t_note = pd.to_datetime(sdoh["charttime"], errors="coerce")
        t_icu = pd.to_datetime(sdoh["icu_intime"], errors="coerce")
        out["potential_temporal_leak_note"] = (t_note > t_icu).astype("Int64")
    else:
        out["potential_temporal_leak_note"] = pd.NA

    for d in DOMAINS:
        agg = sdoh[d].map(norm_label) if d in sdoh.columns else pd.Series("", index=sdoh.index)
        state = agg.map(lambda x, domain=d: note_e_state(domain, x))
        out[f"{d}__E_state_score"] = state.map(E_PRIORITY).astype(int)

        model_cols = [f"{d}__m{i}" for i in (1, 2, 3) if f"{d}__m{i}" in sdoh.columns]
        if model_cols:
            decisive = []
            adverse = []
            for _, row in sdoh[model_cols].iterrows():
                vals = [model_decisive_and_adverse(d, row[c]) for c in model_cols]
                decisive.append(sum(v[0] for v in vals) / 3.0)
                adverse.append(sum(v[1] for v in vals) / 3.0)
            out[f"{d}__decision_conf_mean_note"] = decisive
            out[f"{d}__adv_strength_note"] = adverse
        else:
            out[f"{d}__decision_conf_mean_note"] = np.nan
            out[f"{d}__adv_strength_note"] = np.nan

    grp = out.groupby("subject_key", dropna=False)
    patient = grp.agg(
        n_notes_sdoh=("note_id", "size"),
        potential_temporal_leak_note=("potential_temporal_leak_note", "max"),
    ).reset_index()

    for d in DOMAINS:
        score = grp[f"{d}__E_state_score"].max().reset_index(name=f"{d}__E_state_score")
        patient = patient.merge(score, on="subject_key", how="left", validate="one_to_one")
        inv_priority = {v: k for k, v in E_PRIORITY.items()}
        patient[f"{d}__E_state"] = patient[f"{d}__E_state_score"].map(inv_priority).map(E_LABELS)
        patient[f"{d}__E_observed_adverse"] = (patient[f"{d}__E_state_score"] == E_PRIORITY["observed_adverse"]).astype(int)
        patient[f"{d}_adverse"] = patient[f"{d}__E_observed_adverse"].map(YES_NO_LABELS)

        for source, reducer, dest in [
            (f"{d}__adv_strength_note", "max", f"{d}__adv_strength_max"),
            (f"{d}__adv_strength_note", "mean", f"{d}__adv_strength_mean"),
            (f"{d}__decision_conf_mean_note", "mean", f"{d}__decision_conf_mean"),
        ]:
            tmp = getattr(grp[source], reducer)().reset_index(name=dest)
            patient = patient.merge(tmp, on="subject_key", how="left", validate="one_to_one")

    return patient


def build_time_event(df: pd.DataFrame, event_col: str, time_col: str) -> pd.DataFrame:
    df = df.copy()
    if event_col not in df.columns:
        raise ValueError(f"No existe la columna de evento {event_col}")

    df["event"] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    if time_col in df.columns:
        df["time_days"] = pd.to_numeric(df[time_col], errors="coerce")
    elif {TIME0_COL, DOD_COL}.issubset(df.columns):
        t0 = pd.to_datetime(df[TIME0_COL], errors="coerce")
        dod = pd.to_datetime(df[DOD_COL], errors="coerce")
        delta_days = (dod - t0).dt.total_seconds() / 86400.0
        df["time_days"] = np.where(df["event"] == 1, delta_days, HORIZON_DAYS)
    else:
        raise ValueError(f"No existe {time_col} y no se pudo derivar desde {TIME0_COL}/{DOD_COL}")

    df.loc[df["time_days"].isna(), "time_days"] = HORIZON_DAYS
    df.loc[df["time_days"] < 0, "time_days"] = 0.0
    df.loc[df["time_days"] > HORIZON_DAYS, "time_days"] = HORIZON_DAYS
    df["event_with_missing_dod"] = 0
    if DOD_COL in df.columns:
        dod = pd.to_datetime(df[DOD_COL], errors="coerce")
        df["event_with_missing_dod"] = ((df["event"] == 1) & dod.isna()).astype(int)
    return df


def prepare_analysis_df(surv: pd.DataFrame, sdoh: pd.DataFrame, event_col: str, time_col: str) -> pd.DataFrame:
    if "subject_id" not in surv.columns:
        raise ValueError("El archivo de supervivencia debe incluir subject_id para merge con SDOH")
    surv = surv.copy()
    surv["subject_key"] = surv["subject_id"].astype(str)
    if "docint_log" not in surv.columns:
        if "n_notes_preicu" in surv.columns:
            surv["docint_log"] = np.log1p(pd.to_numeric(surv["n_notes_preicu"], errors="coerce").fillna(0))
        elif "n_words_preicu" in surv.columns:
            surv["docint_log"] = np.log1p(pd.to_numeric(surv["n_words_preicu"], errors="coerce").fillna(0))
        else:
            surv["docint_log"] = 0.0

    pat_sdoh = make_sdoh_patient_features(sdoh)
    ana = surv.merge(pat_sdoh, on="subject_key", how="left", validate="many_to_one")
    ana = apply_clinical_category_collapses(ana)
    ana = build_time_event(ana, event_col=event_col, time_col=time_col)

    for d in DOMAINS:
        state_col = f"{d}__E_state"
        if state_col in ana.columns:
            ana[state_col] = ana[state_col].fillna(E_LABELS["no_evidence"])
        adverse_col = f"{d}_adverse"
        if adverse_col in ana.columns:
            ana[adverse_col] = ana[adverse_col].fillna("No")
        for col in [
            f"{d}__E_observed_adverse",
            f"{d}__adv_strength_max",
            f"{d}__adv_strength_mean",
            f"{d}__decision_conf_mean",
        ]:
            if col in ana.columns:
                ana[col] = pd.to_numeric(ana[col], errors="coerce").fillna(0)

    return ana


def table1(df: pd.DataFrame) -> pd.DataFrame:
    g0 = df[df["event"] == 0]
    g1 = df[df["event"] == 1]
    rows = [["N", "n", str(len(df)), str(len(g0)), str(len(g1))]]

    numeric_cols = []
    for col in NUM_COLS + OPTIONAL_DESCRIPTIVE_NUM_COLS:
        if col in df.columns and col not in numeric_cols:
            numeric_cols.append(col)
    for col in numeric_cols:
        add_numeric_row(rows, col, df, g0, g1, col)

    cat_cols = [c for c in CAT_COLS if c in df.columns]
    sdoh_state_cols = [f"{d}__E_state" for d in DOMAINS if f"{d}__E_state" in df.columns]
    sdoh_adverse_cols = [f"{d}_adverse" for d in DOMAINS if f"{d}_adverse" in df.columns]
    for col in cat_cols + sdoh_state_cols + sdoh_adverse_cols:
        add_categorical_rows(rows, col, df, g0, g1, col)

    return pd.DataFrame(
        rows,
        columns=[
            "Variable",
            "Statistic",
            f"Overall (N={len(df)})",
            f"Alive_30d (N={len(g0)})",
            f"Dead_30d (N={len(g1)})",
        ],
    )


def mortality_frame(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
    s = pd.to_numeric(df["time_days"], errors="coerce")
    e = pd.to_numeric(df["event"], errors="coerce")
    return pd.DataFrame([{
        "n_total": len(df),
        "n_deaths_30d": int((e == 1).sum()),
        "pct_deaths_30d": float((e == 1).mean() * 100),
        "n_censored_30d": int((e == 0).sum()),
        "followup_median_days": float(s.median()),
        "followup_iqr_q1": float(s.quantile(0.25)),
        "followup_iqr_q3": float(s.quantile(0.75)),
        "event_with_missing_dod": int(df.get("event_with_missing_dod", pd.Series(dtype=float)).sum()),
        "missing_event_original": int(df[event_col].isna().sum()) if event_col in df.columns else np.nan,
        "missing_time_after_derivation": int(s.isna().sum()),
    }])


def missingness_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "variable": df.columns,
        "n_missing": [int(df[c].isna().sum()) for c in df.columns],
        "pct_missing": [float(df[c].isna().mean() * 100) for c in df.columns],
    }).sort_values(["pct_missing", "variable"], ascending=[False, True])


def run_ph_test(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in NUM_COLS if c in df.columns]
    cat = [c for c in CAT_COLS if c in df.columns]
    sdoh = []
    for d in DOMAINS:
        for col in [f"{d}__E_observed_adverse", f"{d}__adv_strength_max", f"{d}__decision_conf_mean"]:
            if col in df.columns:
                sdoh.append(col)

    keep = ["time_days", "event"] + numeric + cat + sdoh
    dat = df[keep].copy()
    dat = pd.get_dummies(dat, columns=cat, drop_first=True)

    for c in dat.columns:
        dat[c] = pd.to_numeric(dat[c], errors="coerce")
    dat = dat.dropna(subset=["time_days", "event"])
    for c in dat.columns:
        if c in {"time_days", "event"}:
            continue
        if dat[c].notna().sum() == 0 or dat[c].nunique(dropna=True) <= 1:
            dat = dat.drop(columns=c)
            continue
        dat[c] = dat[c].fillna(dat[c].median())

    covariates = [c for c in dat.columns if c not in {"time_days", "event"}]
    if not covariates:
        return pd.DataFrame(columns=["covariate", "test_statistic", "p", "minus_log2_p"])

    cph = CoxPHFitter(penalizer=1.0)
    try:
        cph.fit(dat, duration_col="time_days", event_col="event")
        zph = proportional_hazard_test(cph, dat, time_transform="rank")
        return zph.summary.reset_index().rename(columns={"index": "covariate"})
    except Exception as exc:
        return pd.DataFrame([{
            "covariate": "PH test not estimable",
            "test_statistic": np.nan,
            "p": np.nan,
            "minus_log2_p": np.nan,
            "note": str(exc),
        }])


def temporal_leakage_frame(df: pd.DataFrame) -> pd.DataFrame:
    has_flag = "potential_temporal_leak_note" in df.columns and df["potential_temporal_leak_note"].notna().any()
    if has_flag:
        n = int((pd.to_numeric(df["potential_temporal_leak_note"], errors="coerce") == 1).sum())
        pct = float((pd.to_numeric(df["potential_temporal_leak_note"], errors="coerce") == 1).mean() * 100)
        note = "Se marco leakage potencial si la hora de la nota SDOH fue posterior a icu_intime."
    else:
        n = np.nan
        pct = np.nan
        note = "El archivo SDOH agregado no contiene charttime/note_charttime; usar n_notes_preicu/n_words_preicu del dataframe de supervivencia para confirmar restriccion pre-ICU."
    return pd.DataFrame([{
        "n_with_leak_flag": n,
        "pct_with_leak_flag": pct,
        "note": note,
    }])


def framing_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "mortality_role", "value": "exploratory", "detail": "Reportar como exploratorio salvo que el protocolo/objetivo primario especifique mortalidad a 30 dias como aim central."},
        {"item": "death_definition", "value": "death_30d", "detail": "Evento binario a 30 dias desde icu_intime; el tiempo se trunca/censura a 30 dias."},
        {"item": "covariate_selection", "value": "a priori", "detail": "Covariables clinicas y SDOH siguen code_cox_survival_model.py: edad ICU, Charlson, SOFA, labs, intensidad documental y dominios SDOH."},
        {"item": "missing_data", "value": "reported", "detail": "La hoja Missingness muestra faltantes por variable; la imputacion mediana solo se usa internamente para el chequeo PH."},
        {"item": "ph_assumption", "value": "Schoenfeld rank test", "detail": "p<0.05 sugiere posible violacion del supuesto de riesgos proporcionales y requiere revision/sensibilidad."},
        {"item": "temporal_leakage", "value": "screening", "detail": "Preferir notas estrictamente pre-ICU; la hoja Temporal_Leakage indica si fue posible evaluarlo con timestamps de notas."},
    ])


def write_workbook(path: str, frames: dict):
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        for sheet, frame in frames.items():
            frame.to_excel(xw, sheet_name=sheet, index=False)

        wb = xw.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1, "text_wrap": True})
        section_fmt = wb.add_format({"bold": True, "bg_color": "#E2F0D9"})
        pct_fmt = wb.add_format({"num_format": "0.0"})
        for sheet, frame in frames.items():
            ws = xw.sheets[sheet]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(frame), max(len(frame.columns) - 1, 0))
            for i, col in enumerate(frame.columns):
                ws.write(0, i, col, header_fmt)
                width = max(14, min(55, max([len(str(col))] + [len(str(v)) for v in frame[col].head(100)])))
                ws.set_column(i, i, width + 2)
            if sheet == "Table1_Descriptive" and "Statistic" in frame.columns:
                for row_idx, value in enumerate(frame["Statistic"], start=1):
                    if value == "n (%)" and frame.iloc[row_idx - 1, 2:].eq("").all():
                        ws.set_row(row_idx, None, section_fmt)
            for pct_col in [c for c in frame.columns if c.startswith("pct_") or c.endswith("pct")]:
                ws.set_column(frame.columns.get_loc(pct_col), frame.columns.get_loc(pct_col), 14, pct_fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surv", default="model_ready_sepsis_survival_stayid_plusSOFA.csv")
    ap.add_argument("--sdoh", default="sdoh_all_notes_with_llm_strata.csv")
    ap.add_argument("--out", default="tabla_analisis_epidemiologico_sepsis.xlsx")
    ap.add_argument("--event-col", default=DEFAULT_EVENT_COL)
    ap.add_argument("--time-col", default=DEFAULT_TIME_COL)
    args = ap.parse_args()

    surv = pd.read_csv(args.surv, low_memory=False)
    sdoh = pd.read_csv(args.sdoh, low_memory=False)
    ana = prepare_analysis_df(surv, sdoh, event_col=args.event_col, time_col=args.time_col)

    frames = {
        "Framing": framing_frame(),
        "Table1_Descriptive": table1(ana),
        "Mortality_Events": mortality_frame(ana, args.event_col),
        "Missingness": missingness_frame(ana),
        "PH_Assumption": run_ph_test(ana),
        "Temporal_Leakage": temporal_leakage_frame(ana),
    }
    write_workbook(args.out, frames)
    print(f"Archivo listo: {args.out}")


if __name__ == "__main__":
    main()
