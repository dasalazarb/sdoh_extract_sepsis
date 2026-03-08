import os
import glob
import json
import gzip
import pandas as pd
import re

# -----------------------------
# Config
# -----------------------------
OUTPUTS_DIR = "/data/salazarda/data/sdoh/outputs"
OUT_DIR = "/data/salazarda/data/sdoh"   # donde quieres escribir los CSV
MODEL_SUBSTR = "DeepSeek"               # filtro para gpt-oss DeepSeek

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Label extraction (igual a tu lógica)
# -----------------------------
categories_regex = r'(Employment status|Housing issues|Transportation issues|Parental status|Relationship status|Social support)'

def clean_text(text):
    """Elimina pares tipo `"Category": "algo_con_label"` antes de extraer."""
    if not isinstance(text, str):
        return text
    pattern = rf'"{categories_regex}"\s*:\s*"[^"]*label[^"]*"'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)

categories = [
    "Employment status",
    "Housing issues",
    "Transportation issues",
    "Parental status",
    "Relationship status",
    "Social support"
]

pattern = re.compile(
    rf'"?(?P<cat>{"|".join(categories)})"?\s*:\s*"(?P<val>[^"]+)"'
)

def extract_labels(text):
    """
    Extrae categorías desde texto:
    - Solo pares `"Category": "valor"`.
    - Ignora valores con 'label'.
    - Se queda con el ÚLTIMO valor válido por categoría.
    """
    result = {cat: None for cat in categories}
    if not isinstance(text, str):
        return result

    for m in pattern.finditer(text):
        cat = m.group("cat")
        val = m.group("val").strip()

        if "label" in val.lower():
            continue
        if any(c in val for c in "{}[]"):
            continue

        result[cat] = val

    return result

# -----------------------------
# Helpers
# -----------------------------
def open_text_maybe_gz(path: str):
    """Abre .jsonl o .jsonl.gz en modo texto."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def read_jsonl_to_df(path: str) -> pd.DataFrame:
    """Lee JSONL de forma robusta (salta líneas malas)."""
    rows = []
    with open_text_maybe_gz(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] JSON inválido en {os.path.basename(path)} línea {i}; se salta.")
                continue
    return pd.DataFrame(rows)

def make_output_path(in_path: str) -> str:
    """sdoh_outputs_XXX.jsonl -> sdoh_labels_extraidos_XXX_v2.csv"""
    base = os.path.basename(in_path)

    # quitar extensiones (.jsonl o .jsonl.gz)
    if base.endswith(".jsonl.gz"):
        stem = base[:-len(".jsonl.gz")]
    elif base.endswith(".jsonl"):
        stem = base[:-len(".jsonl")]
    else:
        stem = os.path.splitext(base)[0]

    out_stem = stem.replace("sdoh_outputs_", "sdoh_labels_extraidos_", 1)
    if not out_stem.endswith("_v2"):
        out_stem += "_v2"
    return os.path.join(OUT_DIR, out_stem + ".csv")

# -----------------------------
# Main: scan + process
# -----------------------------
candidates = sorted(
    glob.glob(os.path.join(OUTPUTS_DIR, f"sdoh_outputs_*{MODEL_SUBSTR}*.jsonl")) +
    glob.glob(os.path.join(OUTPUTS_DIR, f"sdoh_outputs_*{MODEL_SUBSTR}*.jsonl.gz"))
)

# Extra: por si tu carpeta tiene muchas cosas, filtramos "natural" (incluye _natural.csv)
def should_skip(path: str) -> bool:
    name = os.path.basename(path).lower()
    return ("_natural" in name) or name.endswith("_natural.csv")

input_files = [p for p in candidates if not should_skip(p)]

print(f"[INFO] Encontrados {len(input_files)} JSONL(s) para {MODEL_SUBSTR} (después de filtros).")

for in_path in input_files:
    out_path = make_output_path(in_path)

    print(f"\n[INFO] Procesando: {os.path.basename(in_path)}")
    print(f"[INFO] -> Salida:   {os.path.basename(out_path)}")

    df = read_jsonl_to_df(in_path)
    if df.empty:
        print("[WARN] Archivo vacío o sin JSON válido; se omite.")
        continue

    if "text" not in df.columns:
        print(f"[WARN] No existe columna 'text' en {os.path.basename(in_path)}; columnas={df.columns.tolist()}. Se omite.")
        continue

    # Extraer labels
    df["text_clean"] = df["text"].apply(clean_text)
    labels_df = df["text_clean"].apply(extract_labels).apply(pd.Series)

    # Construir salida (solo si existen las columnas base)
    base_cols = [c for c in ["subject_id", "hadm_id", "row_id", "charttime"] if c in df.columns]
    final_df = pd.concat([df[base_cols], labels_df], axis=1)

    final_df.to_csv(out_path, index=False)
    print(f"[OK] Guardado: {out_path}")