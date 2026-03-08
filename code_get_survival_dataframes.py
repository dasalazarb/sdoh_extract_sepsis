import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
def to_int(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def to_dt(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def assert_unique(df, keys, name):
    if df.duplicated(keys).any():
        ex = df.loc[df.duplicated(keys, keep=False), keys].head(10)
        raise ValueError(f"[{name}] NO único por {keys}. Ejemplos:\n{ex}")

def merge_safe(left, right, **kwargs):
    out = left.merge(right, suffixes=("", "_r"), **kwargs)
    dup_cols = [c for c in out.columns if c.endswith("_r") and c[:-2] in out.columns]
    return out.drop(columns=dup_cols)

# ----------------------------
# Paths (SOFA summary ya listo)
# ----------------------------
paths = {
    "sofa24": "data_sofa_24h_summary.csv",              # <-- NUEVO
    "notes": "data_set0_temp_notes_filtered.csv",
    "charlson": "data_set1_charlson_prior12m_icu.csv",
    "sociodemo": "data_set2_sociodemo_admission_mimic4.csv",
    "labs": "data_set3_labs_baseline_icu.csv",
    "sepsis": "data_sepsis_full.csv",
}

# ----------------------------
# Load
# ----------------------------
sofa24   = pd.read_csv(paths["sofa24"])
charlson = pd.read_csv(paths["charlson"])
sociodemo= pd.read_csv(paths["sociodemo"])
labs     = pd.read_csv(paths["labs"])
sepsis   = pd.read_csv(paths["sepsis"])

# IDs
sofa24   = to_int(sofa24,   ["stay_id"])
charlson = to_int(charlson, ["subject_id", "hadm_id", "index_stay_id"])
sociodemo= to_int(sociodemo,["subject_id", "hadm_id", "icu_stay_id"])
labs     = to_int(labs,     ["subject_id", "hadm_id", "index_stay_id"])
sepsis   = to_int(sepsis,   ["subject_id", "stay_id"])

# datetimes
# (sofa24 tiene start/end; los parseamos si existen)
sofa24   = to_dt(sofa24,   ["starttime_0_24h", "endtime_0_24h"])
charlson = to_dt(charlson, ["index_icu_intime"])
sociodemo= to_dt(sociodemo,["admittime", "dischtime", "icu_intime"])
labs     = to_dt(labs,     ["index_icu_intime", "lactate_time", "creatinine_time", "bun_time",
                            "bilirubin_total_time", "platelets_time", "wbc_time", "pao2_time"])
sepsis   = to_dt(sepsis,   ["suspected_infection_time", "dod", "dod_adjusted"])

# ----------------------------
# 0) Pacientes de interés = subject_id presentes en NOTES
# ----------------------------
notes_subjects = pd.read_csv(paths["notes"], usecols=["subject_id"]).drop_duplicates()
notes_subjects = to_int(notes_subjects, ["subject_id"])
assert_unique(notes_subjects, ["subject_id"], "notes_subjects")

# ----------------------------
# 1) Limpieza / dedup por llaves
# ----------------------------
# sepsis: rn==1 si existe
if "rn" in sepsis.columns:
    sepsis = sepsis.loc[sepsis["rn"].fillna(1).astype(int) == 1].drop(columns=["rn"])
assert_unique(sepsis, ["subject_id", "stay_id"], "sepsis")

# filtra sepsis a SOLO subject_id de interés (notes)
sepsis = merge_safe(sepsis, notes_subjects, on="subject_id", how="inner", validate="many_to_one")

# sociodemo: único por icu_stay_id (si duplicado, deja el más temprano)
if sociodemo.duplicated(["icu_stay_id"]).any():
    sociodemo = (sociodemo
                 .sort_values(["icu_stay_id", "icu_intime"])
                 .drop_duplicates(["icu_stay_id"], keep="first"))
assert_unique(sociodemo, ["icu_stay_id"], "sociodemo")

# charlson: rn==1 si existe; único por subject_id+hadm_id
if "rn" in charlson.columns:
    charlson = charlson.loc[charlson["rn"].fillna(1).astype(int) == 1]
if charlson.duplicated(["subject_id", "hadm_id"]).any():
    charlson = (charlson
                .sort_values(["subject_id","hadm_id","index_icu_intime"])
                .drop_duplicates(["subject_id","hadm_id"], keep="first"))
assert_unique(charlson, ["subject_id","hadm_id"], "charlson")

# labs: único por subject_id+hadm_id
if labs.duplicated(["subject_id","hadm_id"]).any():
    labs = (labs
            .sort_values(["subject_id","hadm_id","index_icu_intime"])
            .drop_duplicates(["subject_id","hadm_id"], keep="first"))
assert_unique(labs, ["subject_id","hadm_id"], "labs")

# sofa24: único por stay_id (ya debería venir así)
if sofa24.duplicated(["stay_id"]).any():
    sofa24 = sofa24.sort_values(["stay_id"]).drop_duplicates(["stay_id"], keep="first")
assert_unique(sofa24, ["stay_id"], "sofa24")

# ----------------------------
# 2) Construye DF base a nivel stay_id
# ----------------------------
df = sepsis.copy()

# 2A) Map stay_id -> hadm_id usando sociodemo (icu_stay_id)
df = merge_safe(
    df,
    sociodemo,
    left_on=["subject_id","stay_id"],
    right_on=["subject_id","icu_stay_id"],
    how="left",
    validate="one_to_one",
)

if df["hadm_id"].isna().any():
    print(f"[WARN] stays sin hadm_id tras mapear con sociodemo: {df['hadm_id'].isna().sum()}")

# 2B) Merge SOFA 24h summary (stay-level)
# (no multiplica filas porque es 1 fila por stay_id)
df = merge_safe(df, sofa24, on="stay_id", how="left", validate="one_to_one")

# 2C) Merge Charlson y Labs (hadm-level del cohorte index)
df = merge_safe(df, charlson.drop(columns=["rn"], errors="ignore"),
                on=["subject_id","hadm_id"], how="left", validate="many_to_one")
df = merge_safe(df, labs, on=["subject_id","hadm_id"], how="left", validate="many_to_one")

# ----------------------------
# 3) Doc-intensity desde notes (pre-ICU, evita leakage)
# ----------------------------
hadm_index = (df.dropna(subset=["hadm_id", "icu_intime"])
               .groupby(["subject_id","hadm_id"], as_index=False)["icu_intime"]
               .min())

def compute_doc_stats_preicu(notes_csv, hadm_index_df, chunksize=50_000):
    """
    Cuenta notas/tokens SOLO si charttime < icu_intime (baseline estricta).
    Devuelve stats por (subject_id, hadm_id) para merge posterior.
    """
    agg = {}
    usecols = ["subject_id","hadm_id","charttime","text"]

    for chunk in pd.read_csv(notes_csv, usecols=usecols, chunksize=chunksize):
        chunk = to_int(chunk, ["subject_id","hadm_id"])
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

        # agrega icu_intime por hadm del cohorte index
        chunk = chunk.merge(hadm_index_df, on=["subject_id","hadm_id"], how="left")

        # solo notas antes del ICU admit
        chunk = chunk[chunk["icu_intime"].notna() & (chunk["charttime"] < chunk["icu_intime"])].copy()

        if chunk.empty:
            continue

        txt = chunk["text"].fillna("")
        chunk["n_notes"] = 1
        chunk["n_chars"] = txt.str.len().astype("int64")
        chunk["n_words"] = txt.str.split(r"\s+").map(len).astype("int64")

        g = chunk.groupby(["subject_id","hadm_id"])[["n_notes","n_chars","n_words"]].sum()

        for key, row in g.iterrows():
            vals = row.values.astype("int64")
            if key not in agg:
                agg[key] = vals
            else:
                agg[key] += vals

    out = pd.DataFrame(
        [(k[0], k[1], v[0], v[1], v[2]) for k, v in agg.items()],
        columns=["subject_id","hadm_id","n_notes_preicu","n_chars_preicu","n_words_preicu"]
    )
    out = to_int(out, ["subject_id","hadm_id"])
    assert_unique(out, ["subject_id","hadm_id"], "doc_stats_preicu")
    return out

doc_stats_preicu = compute_doc_stats_preicu(paths["notes"], hadm_index)

df = merge_safe(df, doc_stats_preicu, on=["subject_id","hadm_id"], how="left", validate="many_to_one")

for c in ["n_notes_preicu","n_chars_preicu","n_words_preicu"]:
    if c in df.columns:
        df[c] = df[c].fillna(0).astype("int64")

# ----------------------------
# 4) Final checks + save
# ----------------------------
assert_unique(df, ["subject_id","stay_id"], "FINAL")

out_name = "model_ready_sepsis_survival_stayid_plusSOFA.csv"
df.to_csv(out_name, index=False)
print(f"✅ Saved {out_name} | shape = {df.shape}")