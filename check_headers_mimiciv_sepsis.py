import re
import pandas as pd
from collections import Counter

# === Configuración ===
INPUT_FILE = "/data/salazarda/data/sdoh/temp_notes_filtered.csv"  # cambia a tu ruta
OUTPUT_HEADERS_FILE = "/data/salazarda/data/sdoh/posibles_headers_mimiciv.csv"
CHUNKSIZE = 500  # número de filas por chunk (ajusta según RAM)

# Si tu archivo es CSV con coma, cambia sep=",".
# Para TSV (tabulado), usa sep="\t".
READ_SEP = ","

# === Regex para encontrar encabezados tipo "TEXTO:" al inicio de línea ===
# Ejemplos que captura:
# "HISTORY OF PRESENT ILLNESS:"
# "Social History:"
# "Discharge Instructions:"
header_pattern = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 ,;()\/\-\']{0,80}):\s*$|^\s*([A-Za-z][A-Za-z0-9 ,;()\/\-\']{0,80}):",
    re.MULTILINE
)

def extract_headers_from_text(text: str):
    """
    Devuelve una lista de posibles headers encontrados en el texto.
    Solo busca patrones 'Texto: ' al inicio de línea (con regex MULTILINE).
    """
    if not isinstance(text, str):
        return []
    matches = header_pattern.findall(text)
    headers = []
    for m1, m2 in matches:
        candidate = m1 or m2
        candidate = candidate.strip()
        if candidate:
            headers.append(candidate)
    return headers

def main():
    header_counter = Counter()

    # Leer en chunks para no reventar memoria
    for chunk in pd.read_csv(
        INPUT_FILE,
        sep=READ_SEP,
        usecols=["subject_id", "hadm_id", "text", "charttime"],
        chunksize=CHUNKSIZE
    ):
        for txt in chunk["text"].dropna():
            headers = extract_headers_from_text(txt)
            # Normalizamos a mayúsculas para agrupar variantes
            headers = [h.upper() for h in headers]
            header_counter.update(headers)

    # Pasar el Counter a DataFrame ordenado por frecuencia
    headers_df = (
        pd.DataFrame(
            [{"header": h, "count": c} for h, c in header_counter.items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # Guardar a CSV para inspección
    headers_df.to_csv(OUTPUT_HEADERS_FILE, index=False)
    print(f"Se guardaron {len(headers_df)} posibles headers en: {OUTPUT_HEADERS_FILE}")
    print(headers_df.head(20))


if __name__ == "__main__":
    main()
