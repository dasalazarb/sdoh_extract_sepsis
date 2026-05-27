#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis epidemiologico/descriptivo para cohorte de sepsis con enfoque en mortalidad.

Genera un archivo .xlsx listo para manuscrito con:
- Tabla 1 descriptiva global y por estado de muerte a 30 dias.
- Resumen de eventos de muerte y censura.
- Perfil de datos faltantes.
- Verificacion de supuesto de riesgos proporcionales (Cox PH).
- Chequeo de posible leakage temporal en variables SDOH.

Variables alineadas con:
- code_build_sdoh_all_notes_with_llm_strata_from_extracted.py
- code_cox_survival_model.py
"""

import argparse
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

DOMAINS = ["employment_status", "housing_issues", "transportation_issues", "social_support"]
CAT_COLS = ["gender", "race", "admission_type", "icu_type"]
NUM_COLS = [
    "age_at_icu_intime",
    "charlson_12m_prior_with_age",
    "sofa_24hours",
    "lactate", "creatinine", "bun", "bilirubin_total", "platelets", "wbc", "pao2",
]

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


def norm_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def summarize_num(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return "NA"
    return f"{s.median():.2f} [{s.quantile(0.25):.2f}, {s.quantile(0.75):.2f}]"


def summarize_cat(s: pd.Series) -> str:
    n = s.notna().sum()
    if n == 0:
        return "NA"
    top = s.astype("object").fillna("Missing").value_counts(dropna=False)
    k = top.index[0]
    v = top.iloc[0]
    return f"{k}: {v} ({100.0*v/len(s):.1f}%)"


def make_sdoh_patient_features(sdoh: pd.DataFrame) -> pd.DataFrame:
    for col in ["subject_id", "note_id"]:
        sdoh[col] = sdoh[col].astype(str)

    out = sdoh[["subject_id", "note_id"]].copy()

    # flags de leakage: nota posterior a ICU
    if "note_charttime" in sdoh.columns and "icu_intime" in sdoh.columns:
        t_note = pd.to_datetime(sdoh["note_charttime"], errors="coerce")
        t_icu = pd.to_datetime(sdoh["icu_intime"], errors="coerce")
        out["potential_temporal_leak_note"] = (t_note > t_icu).astype("Int64")
    else:
        out["potential_temporal_leak_note"] = pd.NA

    for d in DOMAINS:
        agg = sdoh[d].map(norm_label) if d in sdoh.columns else pd.Series("", index=sdoh.index)
        adverse = agg.isin(ADVERSE_LABELS[d]).astype(int)
        observed = (agg.isin(VALID_LABELS[d]) | (agg == "not_mentioned")).astype(int)

        out[f"{d}_adverse"] = adverse
        out[f"{d}_observed"] = observed

    grp = out.groupby("subject_id", dropna=False)
    patient = grp.max(numeric_only=True).reset_index()
    patient["n_notes_sdoh"] = grp.size().values
    return patient


def table1(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
    rows = []
    g0 = df[df[event_col] == 0]
    g1 = df[df[event_col] == 1]

    rows.append(["N", str(len(df)), str(len(g0)), str(len(g1))])
    for c in NUM_COLS:
        if c in df.columns:
            rows.append([c, summarize_num(df[c]), summarize_num(g0[c]), summarize_num(g1[c])])
    for c in CAT_COLS:
        if c in df.columns:
            rows.append([c, summarize_cat(df[c]), summarize_cat(g0[c]), summarize_cat(g1[c])])
    for d in DOMAINS:
        c = f"{d}_adverse"
        if c in df.columns:
            rows.append([c, summarize_cat(df[c]), summarize_cat(g0[c]), summarize_cat(g1[c])])

    return pd.DataFrame(rows, columns=["Variable", "Overall", "Alive_30d", "Dead_30d"])


def mortality_frame(df: pd.DataFrame, event_col: str, time_col: str) -> pd.DataFrame:
    s = pd.to_numeric(df[time_col], errors="coerce")
    e = pd.to_numeric(df[event_col], errors="coerce")
    out = {
        "n_total": len(df),
        "n_deaths_30d": int((e == 1).sum()),
        "pct_deaths_30d": float((e == 1).mean() * 100),
        "n_censored_30d": int((e == 0).sum()),
        "followup_median_days": float(s.median()),
        "followup_iqr_q1": float(s.quantile(0.25)),
        "followup_iqr_q3": float(s.quantile(0.75)),
        "missing_event": int(e.isna().sum()),
        "missing_time": int(s.isna().sum()),
    }
    return pd.DataFrame([out])


def run_ph_test(df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    keep = [c for c in [*NUM_COLS, *(f"{d}_adverse" for d in DOMAINS)] if c in df.columns]
    if len(keep) == 0:
        return pd.DataFrame(columns=["covariate", "p"])

    dat = df[[time_col, event_col] + keep].copy()
    dat = dat.apply(pd.to_numeric, errors="coerce")
    dat = dat.dropna(subset=[time_col, event_col])
    for c in keep:
        dat[c] = dat[c].fillna(dat[c].median())

    cph = CoxPHFitter(penalizer=1.0)
    cph.fit(dat, duration_col=time_col, event_col=event_col)
    zph = proportional_hazard_test(cph, dat, time_transform="rank")
    return zph.summary.reset_index().rename(columns={"index": "covariate"})[["covariate", "p"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surv", default="model_ready_sepsis_survival_stayid_plusSOFA.csv")
    ap.add_argument("--sdoh", default="sdoh_all_notes_with_llm_strata.csv")
    ap.add_argument("--out", default="tabla_analisis_epidemiologico_sepsis.xlsx")
    ap.add_argument("--event-col", default="death_30d")
    ap.add_argument("--time-col", default="time_to_event_days")
    args = ap.parse_args()

    surv = pd.read_csv(args.surv)
    sdoh = pd.read_csv(args.sdoh)

    pat_sdoh = make_sdoh_patient_features(sdoh)
    if "subject_id" not in surv.columns:
        raise ValueError("El archivo de supervivencia debe incluir subject_id para merge con SDOH")

    surv["subject_id"] = surv["subject_id"].astype(str)
    ana = surv.merge(pat_sdoh, on="subject_id", how="left")

    if args.time_col not in ana.columns:
        # fallback reproducible con columnas en code_cox_survival_model
        if {"icu_intime", "dod_adjusted"}.issubset(set(ana.columns)):
            t0 = pd.to_datetime(ana["icu_intime"], errors="coerce")
            dod = pd.to_datetime(ana["dod_adjusted"], errors="coerce")
            ana[args.time_col] = (dod - t0).dt.total_seconds() / 86400.0
        else:
            raise ValueError(f"No existe {args.time_col} y no se pudo derivar")

    miss = pd.DataFrame({
        "variable": ana.columns,
        "n_missing": [ana[c].isna().sum() for c in ana.columns],
        "pct_missing": [ana[c].isna().mean() * 100 for c in ana.columns],
    }).sort_values(["pct_missing", "variable"], ascending=[False, True])

    leak = pd.DataFrame([{
        "n_with_leak_flag": int((ana.get("potential_temporal_leak_note", pd.Series(dtype=float)) == 1).sum()),
        "pct_with_leak_flag": float((ana.get("potential_temporal_leak_note", pd.Series(dtype=float)) == 1).mean() * 100) if "potential_temporal_leak_note" in ana.columns else np.nan,
        "note": "Exploratorio: revisar si note_charttime > icu_intime en notas usadas para SDOH",
    }])

    framing = pd.DataFrame([
        {"item": "mortality_role", "value": "exploratory", "detail": "Analisis de mortalidad recomendado como exploratorio salvo que el protocolo lo defina como objetivo primario."},
        {"item": "covariate_selection", "value": "a_priori + disponibilidad", "detail": "Se usan covariables clinicas de code_cox_survival_model.py y exposiciones SDOH predefinidas."},
        {"item": "ph_assumption", "value": "Schoenfeld rank test", "detail": "p<0.05 sugiere violacion del supuesto de riesgos proporcionales."},
        {"item": "missing_data", "value": "tabla de faltantes", "detail": "Se reporta magnitud de faltantes por variable; imputacion para PH test solo descriptiva interna."},
        {"item": "temporal_leakage", "value": "screening flag", "detail": "Se marca posible leakage cuando timestamp de nota excede icu_intime."},
    ])

    t1 = table1(ana, args.event_col)
    mort = mortality_frame(ana, args.event_col, args.time_col)
    ph = run_ph_test(ana, args.time_col, args.event_col)

    with pd.ExcelWriter(args.out, engine="xlsxwriter") as xw:
        framing.to_excel(xw, sheet_name="Framing", index=False)
        t1.to_excel(xw, sheet_name="Table1_Descriptive", index=False)
        mort.to_excel(xw, sheet_name="Mortality_Events", index=False)
        miss.to_excel(xw, sheet_name="Missingness", index=False)
        ph.to_excel(xw, sheet_name="PH_Assumption", index=False)
        leak.to_excel(xw, sheet_name="Temporal_Leakage", index=False)

        wb = xw.book
        fmt_header = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        fmt_pct = wb.add_format({"num_format": "0.0"})
        for sh, df_ in {
            "Framing": framing,
            "Table1_Descriptive": t1,
            "Mortality_Events": mort,
            "Missingness": miss,
            "PH_Assumption": ph,
            "Temporal_Leakage": leak,
        }.items():
            ws = xw.sheets[sh]
            for i, col in enumerate(df_.columns):
                ws.write(0, i, col, fmt_header)
                ws.set_column(i, i, max(16, min(60, len(col) + 2)))
            if "pct_missing" in df_.columns:
                j = list(df_.columns).index("pct_missing")
                ws.set_column(j, j, 14, fmt_pct)

    print(f"Archivo listo: {args.out}")


if __name__ == "__main__":
    main()
