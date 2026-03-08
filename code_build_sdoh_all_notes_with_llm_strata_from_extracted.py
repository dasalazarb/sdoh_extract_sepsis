import argparse
import glob
import os
import re
from collections import Counter, defaultdict
from functools import reduce

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

MODEL_SPECS = [
    ("m1", "deepseek", ["deepseek"]),
    ("m2", "gptoss", ["gpt-oss", "gptoss"]),
    ("m3", "mistral", ["mistral"]),
]

DOMAIN_MAP = {
    "Relationship status": "relationship_status",
    "Employment status": "employment_status",
    "Housing issues": "housing_issues",
    "Parental status": "parental_status",
    "Social support": "social_support",
    "Transportation issues": "transportation_issues",
}

OUTPUT_COLUMN_ORDER = [
    "subject_id", "hadm_id", "note_id",
    "relationship_status__m1", "relationship_status__m2", "relationship_status__m3",
    "relationship_status", "relationship_status__agree_n", "relationship_status__stratum",
    "employment_status__m1", "employment_status__m2", "employment_status__m3",
    "employment_status", "employment_status__agree_n", "employment_status__stratum",
    "housing_issues__m1", "housing_issues__m2", "housing_issues__m3",
    "housing_issues", "housing_issues__agree_n", "housing_issues__stratum",
    "parental_status__m1", "parental_status__m2", "parental_status__m3",
    "parental_status", "parental_status__agree_n", "parental_status__stratum",
    "social_support__m1", "social_support__m2", "social_support__m3",
    "social_support", "social_support__agree_n", "social_support__stratum",
    "transportation_issues__m1", "transportation_issues__m2", "transportation_issues__m3",
    "transportation_issues", "transportation_issues__agree_n", "transportation_issues__stratum",
]

VALID_LABELS = {
    "employment_status": [
        "employed", "underemployed", "unemployed", "disability", "retired", "student"
    ],
    "parental_status": ["yes", "no"],
    "housing_issues": ["financial_status", "undomiciled", "other"],
    "transportation_issues": ["distance", "resources", "other"],
    "relationship_status": ["married", "partnered", "divorced", "widowed", "single"],
    "social_support": ["plus", "minus"],
}

ID_COLS = ["subject_id", "hadm_id", "note_id"]


# ============================================================
# HELPERS
# ============================================================

def canonicalize_id(x):
    """Convierte 123.0 -> '123', preserva strings alfanuméricos."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def maybe_cast_numeric_series(series):
    """Si una serie es completamente numérica, la convierte a Int64; si no, la deja como string."""
    series = series.astype("string")
    non_null = series.dropna()
    if len(non_null) == 0:
        return series
    if non_null.str.fullmatch(r"-?\d+").all():
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    return series


def normalize_label(x):
    """Normaliza salida de modelos (minúscula, underscore, etc.)."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_+\-]", "", s)

    if s in {"notmentioned", "not_mentioned", "not-mention"}:
        return "not_mentioned"
    if s in {"unknown", "uncertain", "unsure"}:
        return "unsure"
    return s


def safe_mode(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return None, 0
    cnt = Counter(vals)
    max_c = max(cnt.values())
    top = sorted([k for k, v in cnt.items() if v == max_c])[0]
    return top, max_c


def compute_stratum(model_vals, valid_set):
    """
    Regla práctica para reproducir la estructura final:
    - si las 3 predicciones faltan -> NaN/NaN/NaN
    - si falta al menos una, pero existe alguna predicción -> discordant usando la moda disponible
    - si están las 3 -> all3_valid / maj2_valid / all3_nonvalid / discordant
    """
    observed = [v for v in model_vals if v is not None]
    if len(observed) == 0:
        return None, None, None

    mode_label, mode_count = safe_mode(model_vals)

    if any(v is None for v in model_vals):
        return "discordant", mode_label, mode_count

    all3 = mode_count == 3
    maj2 = mode_count == 2

    if all3 and mode_label in valid_set:
        return "all3_valid", mode_label, mode_count
    if maj2 and mode_label in valid_set:
        return "maj2_valid", mode_label, mode_count
    if all3 and mode_label not in valid_set:
        return "all3_nonvalid", mode_label, mode_count
    return "discordant", mode_label, mode_count


def detect_model_slot(filename):
    lower = os.path.basename(filename).lower()
    for slot, _, patterns in MODEL_SPECS:
        if any(p in lower for p in patterns):
            return slot
    return None


def collect_prediction_files(input_dir):
    pattern = os.path.join(input_dir, "sdoh_labels_extraidos_*_v2.csv")
    all_files = sorted(glob.glob(pattern))
    files_by_model = defaultdict(list)

    for f in all_files:
        slot = detect_model_slot(f)
        if slot is None:
            print(f"[WARN] Archivo ignorado (modelo no reconocido): {os.path.basename(f)}")
            continue
        files_by_model[slot].append(f)

    return files_by_model


def ensure_required_columns(df, path):
    required_id_candidates = {
        "subject_id": ["subject_id"],
        "hadm_id": ["hadm_id"],
        "note_id": ["row_id", "note_id"],
    }

    rename_map = {}
    for canonical, candidates in required_id_candidates.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"Falta columna requerida para '{canonical}' en {path}. "
                f"Opciones esperadas: {candidates}"
            )
        rename_map[found] = canonical

    df = df.rename(columns=rename_map)

    missing_domains = [c for c in DOMAIN_MAP.keys() if c not in df.columns]
    if missing_domains:
        raise ValueError(
            f"Faltan columnas de SDOH en {path}: {missing_domains}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    return df


def load_one_file(path, model_slot):
    df = pd.read_csv(path)
    df = ensure_required_columns(df, path).copy()

    # Normalizar ids
    for c in ID_COLS:
        df[c] = df[c].apply(canonicalize_id)

    # Mantener solo IDs + las 6 etiquetas
    keep = ID_COLS + list(DOMAIN_MAP.keys())
    df = df[keep].copy()

    # Renombrar a estructura final por modelo
    rename_map = {
        human_name: f"{machine_name}__{model_slot}"
        for human_name, machine_name in DOMAIN_MAP.items()
    }
    df = df.rename(columns=rename_map)

    # Normalizar labels
    for col in rename_map.values():
        df[col] = df[col].apply(normalize_label)

    df["source_file"] = os.path.basename(path)
    return df


def first_non_null(series):
    for x in series:
        if pd.notna(x):
            return x
    return None


def collapse_model_predictions(df_model, model_slot):
    """
    Si el mismo note_id aparece en varios archivos del mismo modelo,
    colapsa a una sola fila por nota manteniendo el primer valor no nulo
    por dominio y reportando conflictos.
    """
    pred_cols = [c for c in df_model.columns if c.endswith(f"__{model_slot}")]
    conflict_summary = {}

    for col in pred_cols:
        nunique = (
            df_model.groupby(ID_COLS, dropna=False)[col]
            .nunique(dropna=True)
        )
        conflict_summary[col] = int((nunique > 1).sum())

    if any(v > 0 for v in conflict_summary.values()):
        print(f"[WARN] Se detectaron conflictos dentro de {model_slot}:")
        for col, n in conflict_summary.items():
            if n > 0:
                print(f"       - {col}: {n} notas con >1 valor distinto")

    df_model = df_model.sort_values(["source_file"] + ID_COLS).reset_index(drop=True)

    agg_map = {col: first_non_null for col in pred_cols}
    agg_map["source_file"] = lambda s: " | ".join(sorted(set(s.astype(str))))

    collapsed = (
        df_model.groupby(ID_COLS, dropna=False, as_index=False)
        .agg(agg_map)
    )

    return collapsed


def merge_models(model_dfs):
    non_empty = [df for df in model_dfs if df is not None and len(df) > 0]
    if not non_empty:
        raise ValueError("No se cargó ningún archivo válido de predicciones.")
    return reduce(lambda l, r: pd.merge(l, r, on=ID_COLS, how="outer"), non_empty)


def compute_consensus_columns(df):
    for machine_name in DOMAIN_MAP.values():
        model_cols = [f"{machine_name}__m1", f"{machine_name}__m2", f"{machine_name}__m3"]
        valid_set = set(normalize_label(x) for x in VALID_LABELS[machine_name])

        consensus = []
        agree_n = []
        strata = []

        for _, row in df.iterrows():
            vals = [row.get(c, None) for c in model_cols]
            vals = [None if pd.isna(v) else v for v in vals]
            st, label, n = compute_stratum(vals, valid_set)
            consensus.append(label)
            agree_n.append(n)
            strata.append(st)

        df[machine_name] = consensus
        df[f"{machine_name}__agree_n"] = agree_n
        df[f"{machine_name}__stratum"] = strata

    return df


def finalize_output(df):
    # Asegurar que existan todas las columnas aunque falte un modelo completo
    for col in OUTPUT_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[OUTPUT_COLUMN_ORDER].copy()

    # Tipos de ids
    df["subject_id"] = maybe_cast_numeric_series(df["subject_id"])
    df["hadm_id"] = maybe_cast_numeric_series(df["hadm_id"])
    df["note_id"] = df["note_id"].astype("string")

    # Ordenar de forma estable
    subject_sort = pd.to_numeric(df["subject_id"], errors="coerce")
    hadm_sort = pd.to_numeric(df["hadm_id"], errors="coerce")
    note_sort = df["note_id"].fillna("").astype(str)

    df = (
        df.assign(_subject_sort=subject_sort, _hadm_sort=hadm_sort, _note_sort=note_sort)
          .sort_values(["_subject_sort", "_hadm_sort", "_note_sort"], kind="stable")
          .drop(columns=["_subject_sort", "_hadm_sort", "_note_sort"])
          .reset_index(drop=True)
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Construye sdoh_all_notes_with_llm_strata.csv a partir de archivos de predicción por modelo."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directorio donde están los archivos sdoh_labels_extraidos_*_v2.csv"
    )
    parser.add_argument(
        "--output-file",
        default="sdoh_all_notes_with_llm_strata.csv",
        help="Ruta del CSV final"
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=9959,
        help="Total esperado de notas clínicas para validación"
    )
    parser.add_argument(
        "--strict-count-check",
        action="store_true",
        help="Si se activa, lanza error cuando el total final != expected_total"
    )
    args = parser.parse_args()

    files_by_model = collect_prediction_files(args.input_dir)

    print("\n=== Archivos detectados por modelo ===")
    total_files = 0
    for model_slot, _, _ in MODEL_SPECS:
        files = files_by_model.get(model_slot, [])
        total_files += len(files)
        print(f"{model_slot}: {len(files)} archivo(s)")
        for f in files:
            print(f"   - {os.path.basename(f)}")

    if total_files == 0:
        raise FileNotFoundError(
            f"No encontré archivos con patrón sdoh_labels_extraidos_*_v2.csv en: {args.input_dir}"
        )

    model_frames = []
    summary_rows = []

    for model_slot, _, _ in MODEL_SPECS:
        files = files_by_model.get(model_slot, [])
        if not files:
            print(f"[WARN] No se encontraron archivos para {model_slot}")
            model_frames.append(None)
            continue

        loaded = []
        for path in files:
            df_one = load_one_file(path, model_slot)
            loaded.append(df_one)
            summary_rows.append({
                "model_slot": model_slot,
                "source_file": os.path.basename(path),
                "rows": len(df_one),
                "unique_notes": df_one[ID_COLS].drop_duplicates().shape[0],
            })

        df_model = pd.concat(loaded, ignore_index=True)
        before = len(df_model)
        df_model = collapse_model_predictions(df_model, model_slot)
        after = len(df_model)

        print(
            f"\n[{model_slot}] filas concatenadas: {before:,} | "
            f"notas únicas tras colapsar: {after:,}"
        )
        model_frames.append(df_model)

    df_all = merge_models(model_frames)
    print(f"\nTotal de notas únicas tras unir modelos: {len(df_all):,}")

    df_all = compute_consensus_columns(df_all)
    df_all = finalize_output(df_all)

    final_total = len(df_all)
    print(f"Total final de clinical notes: {final_total:,}")

    if args.expected_total is not None and final_total != args.expected_total:
        msg = (
            f"[WARN] El total final ({final_total:,}) no coincide con expected_total "
            f"({args.expected_total:,})."
        )
        if args.strict_count_check:
            raise AssertionError(msg)
        print(msg)
    elif args.expected_total is not None:
        print(f"[OK] El total final coincide con expected_total={args.expected_total:,}")

    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    df_all.to_csv(args.output_file, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "build_sdoh_all_notes_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nGuardado CSV final: {args.output_file}")
    print(f"Guardado resumen de entrada: {summary_path}")


if __name__ == "__main__":
    main()
