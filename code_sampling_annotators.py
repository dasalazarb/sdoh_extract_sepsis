import os
import shutil
import pandas as pd

# ============================
# 0) CONFIG
# ============================

MASTER_CSV = "combo_sampling_outputs/MASTER_all_categories_max3_per_combo.csv"
SRC_NOTES_DIR = "notes_by_patient_deepseek_2048tokens"

# Todos anotan LO MISMO; solo cambia el nombre de carpeta y del excel
annotators = ["Diego", "Daniel", "Pankaj"]

# Mapeo: el master tiene columna "category" en snake_case (como en tu muestreo)
# y acá definimos cómo se llama la hoja + labels válidos
categories = {
    "employment_status": {
        "sheet": "Employment_status",
        "labels": ["employed", "underemployed", "unemployed", "disability", "retired", "student"],
    },
    "parental_status": {
        "sheet": "Parental_status",
        "labels": ["yes", "no"],
    },
    "housing_issues": {
        "sheet": "Housing_issues",
        "labels": ["financial_status", "undomiciled", "other"],
    },
    "transportation_issues": {
        "sheet": "Transportation_issues",
        "labels": ["distance", "resources", "other"],
    },
    "relationship_status": {
        "sheet": "Relationship_status",
        "labels": ["married", "partnered", "divorced", "widowed", "single"],
    },
    "social_support": {
        "sheet": "Social_support",
        "labels": ["plus", "minus"],
    },
}

# Si quieres evitar sesgo por ver preds, deja True 
BLIND_MODELS = True

# Columnas esperadas en MASTER para preds por categoría
# ej: employment_status__m1, employment_status__m2, employment_status__m3, employment_status__combo
def preds_cols_for(cat_key: str):
    return {
        "m1": f"{cat_key}__m1",
        "m2": f"{cat_key}__m2",
        "m3": f"{cat_key}__m3",
        "combo": f"{cat_key}__combo",
    }

# Nombre .txt esperado
def note_filename(subject_id: str, hadm_id: str, note_id: str) -> str:
    return f"{subject_id}_{hadm_id}_{note_id}.txt"

def slugify(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


# ============================
# 1) CARGAR MASTER
# ============================

df = pd.read_csv(MASTER_CSV, dtype={"note_id": str, "subject_id": str, "hadm_id": str})
required = {"subject_id", "hadm_id", "note_id", "category"}
missing_req = required - set(df.columns)
if missing_req:
    raise ValueError(f"MASTER CSV no tiene columnas requeridas: {missing_req}")

# normalizar ids
df["subject_id"] = df["subject_id"].astype(str).str.strip()
df["hadm_id"] = df["hadm_id"].astype(str).str.strip()
df["note_id"] = df["note_id"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

# ============================
# 2) PREPARAR LISTA ÚNICA DE NOTAS A COPIAR
# ============================

df_notes_unique = df[["subject_id", "hadm_id", "note_id"]].drop_duplicates()
note_key_to_fname = {}
missing_files_global = []

for _, r in df_notes_unique.iterrows():
    subj, hadm, nid = r["subject_id"], r["hadm_id"], r["note_id"]
    fname = note_filename(subj, hadm, nid)
    note_key_to_fname[(subj, hadm, nid)] = fname

    src_path = os.path.join(SRC_NOTES_DIR, fname)
    if not os.path.exists(src_path):
        missing_files_global.append(src_path)

if missing_files_global:
    print("⚠ WARNING: Hay .txt faltantes en SRC_NOTES_DIR. Muestra (primeros 15):")
    for p in missing_files_global[:15]:
        print("  -", p)
    if len(missing_files_global) > 15:
        print("  ...", len(missing_files_global) - 15, "más")

# ============================
# 3) CREAR PAQUETES (uno por anotador, pero mismo contenido)
# ============================

for annot in annotators:
    base_dir = f"anotacion_{slugify(annot)}"
    notes_dir = os.path.join(base_dir, "clinical_notes")
    os.makedirs(notes_dir, exist_ok=True)

    # 3.1 Copiar notas .txt (solo las que existan)
    copied = 0
    missing = 0

    for _, r in df_notes_unique.iterrows():
        subj, hadm, nid = r["subject_id"], r["hadm_id"], r["note_id"]
        fname = note_key_to_fname[(subj, hadm, nid)]
        src_path = os.path.join(SRC_NOTES_DIR, fname)
        dst_path = os.path.join(notes_dir, fname)

        if not os.path.exists(src_path):
            missing += 1
            continue
        if not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)
            copied += 1

    print(f"\n[{annot}] Copiadas {copied} notas. Faltantes {missing}.")

    # 3.2 Crear Excel con hojas por categoría
    excel_out = os.path.join(base_dir, f"planilla_{slugify(annot)}_sdoh.xlsx")
    print(f"[{annot}] Creando Excel: {excel_out}")

    with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
        workbook = writer.book

        for cat_key, info in categories.items():
            sheet_name = info["sheet"]
            labels = info["labels"]
            preds = preds_cols_for(cat_key)

            # filtrar filas para esa categoría
            df_cat = df[df["category"] == cat_key].copy()
            if df_cat.empty:
                print(f"  - {sheet_name}: sin filas en MASTER (category == {cat_key}). Hoja vacía.")
                # aún creamos hoja vacía para consistencia
                empty_df = pd.DataFrame(columns=["subject_id","hadm_id","note_id","note_link"])
                empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            # verificar columnas preds en master; si no existen, dejamos pred vacía
            for k, col in preds.items():
                if col not in df_cat.columns:
                    df_cat[col] = ""

            # construir hyperlink
            def make_link(row):
                key = (str(row["subject_id"]), str(row["hadm_id"]), str(row["note_id"]))
                fname = note_key_to_fname.get(key)
                if not fname:
                    return str(row["note_id"])
                return f'=HYPERLINK("clinical_notes/{fname}", "{row["note_id"]}")'

            # armar hoja
            sheet_df = pd.DataFrame({
                "subject_id": df_cat["subject_id"].values,
                "hadm_id": df_cat["hadm_id"].values,
                "note_id": df_cat["note_id"].values,
            })
            sheet_df["note_link"] = df_cat.apply(make_link, axis=1)

            # columnas de predicciones (opcional)
            sheet_df["m1_pred"] = df_cat[preds["m1"]].values
            sheet_df["m2_pred"] = df_cat[preds["m2"]].values
            sheet_df["m3_pred"] = df_cat[preds["m3"]].values
            sheet_df["combo"]   = df_cat[preds["combo"]].values

            # columnas de anotación: labels + unknown
            for lab in labels:
                sheet_df[lab] = ""
            sheet_df["unknown"] = ""

            # evidencia
            sheet_df["evidence_text"] = ""
            sheet_df["section"] = ""

            # escribir
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # formato
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)

            # widths básicas
            ws.set_column("A:A", 12)  # subject_id
            ws.set_column("B:B", 12)  # hadm_id
            ws.set_column("C:C", 18)  # note_id
            ws.set_column("D:D", 45)  # note_link

            # preds cols: E-H
            ws.set_column("E:E", 12)  # m1_pred
            ws.set_column("F:F", 12)  # m2_pred
            ws.set_column("G:G", 12)  # m3_pred
            ws.set_column("H:H", 22)  # combo

            # labels start at I
            start_label_col = 8  # 0-indexed; A=0 ... H=7, I=8
            ws.set_column(start_label_col, start_label_col + len(labels), 16)  # labels + unknown

            # evidence columns at the end
            # evidence_text and section are last 2 columns
            last_cols = sheet_df.columns.tolist()
            ev_idx = last_cols.index("evidence_text")
            sec_idx = last_cols.index("section")
            ws.set_column(ev_idx, ev_idx, 45)
            ws.set_column(sec_idx, sec_idx, 25)

            # ocultar predicciones si BLIND_MODELS
            if BLIND_MODELS:
                ws.set_column("E:H", None, None, {"hidden": True})

    print(f"[{annot}] Listo: {base_dir}/ (Excel + clinical_notes/)")

print("\n✅ Finalizado. Paquetes creados para todos los anotadores.")
