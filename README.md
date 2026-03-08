# SDOH Extraction for Sepsis Cohorts (MIMIC)

This repository contains scripts to run **large language models (LLMs)** for extracting **Social Determinants of Health (SDOH)** from sepsis-related clinical notes (primarily MIMIC-IV discharge notes), plus utilities for sampling outputs and preparing annotation packages.

The project is designed around local/HPC execution (SLURM + GPU), local model checkpoints, and preprocessed note files.

---

## What this repository does

At a high level, the workflow is:

1. Select subject/admission/note IDs from prebuilt CSVs.
2. Load note text (from PostgreSQL MIMIC schemas or pre-saved `.txt` note files).
3. Build structured SDOH prompts.
4. Run one of several local LLMs:
   - DeepSeek
   - GPT-OSS
   - Mistral
5. Save model outputs in JSONL + postprocessed CSV text exports.
6. Sample disagreement/consensus patterns across models for human review.
7. Build annotator-ready Excel + note bundles.

---

## End-to-end pipeline order

The primary analysis pipeline proceeds in this order:

1. **LLM inference generation**
   - Python inference scripts: `gpt_oss_sdoh.py`, `mistral_sdoh.py`, `deepseek_sdoh.py`
   - SLURM wrappers: `run_gptoss_mimic4.sh`, `run_mistral_mimic4.sh`, `run_deepseek_mimic4.sh`
2. **Extraction + strata build from LLM outputs**
   - Run inference outputs through `utils_llm.py` and `code_extract_results_v2.py`
   - Build strata dataset with `code_build_sdoh_all_notes_with_llm_strata_from_extracted.py`
3. **Clinical covariate extraction (SQL sets)**
   - Execute `sql_set1_charlson_lookback_12mo.sql`, `sql_set2_data_sociodemo_admission.sql`,
     `sql_set3_baseline_labs.sql`, and `sql_set4_early_interventions.sql`
   - Assemble survival-ready dataframes with `code_get_survival_dataframes.py`
4. **Survival modeling**
   - Combine outputs from steps 2 and 3 in `code_cox_survival_model.py`
5. **Results scripts**
   - Run `code_results_*.py` scripts for metrics, descriptives, and figures.

---

## Repository structure

### Core inference scripts

- `deepseek_sdoh.py` — DeepSeek inference pipeline with logging and batched generation.
- `gpt_oss_sdoh.py` — GPT-OSS inference pipeline with custom generate loop.
- `mistral_sdoh.py` — Mistral inference pipeline with HF `pipeline` and `device_map="auto"`.

### Shared utilities

- `utils_llm.py` — utility module for:
  - loading notes from MIMIC-III/MIMIC-IV PostgreSQL;
  - loading notes from pre-extracted `.txt` files;
  - building multiple prompt variants;
  - saving outputs to JSONL and derived CSVs;
  - helper text normalization functions.

### Sampling + annotation tooling

- `code_sampling_MASTER.py` — category-wise stratified sampling over model output combinations.
- `code_sampling_annotators.py` — creates annotator packages (Excel sheets + linked note files).

### Data exploration / SQL / notebooks

- `Description_of_sepsis_clinical_notes.sql` — SQL snippets for cohort inspection and note length stats.
- `check_headers_mimiciv_sepsis.py` — scans note text for probable section headers.
- `00_checking_clinicalnotes_allModels.ipynb` and `01_checking_clinicalnotes_LlamaCare.ipynb` — exploratory notebooks.

### HPC submission scripts

- `run_deepseek.sh`, `run_deepseek_mimic4.sh`
- `run_gptoss.sh`, `run_gptoss_mimic4.sh`
- `run_mistral_mimic4.sh`

These include SLURM resources, module loading, and the Python entrypoints.

---

## SDOH targets

The project includes prompt templates covering SDOH categories such as:

- Employment status
- Housing issues
- Transportation issues
- Parental status
- Relationship status
- Social support

Depending on the prompt version, label sets vary (binary, multi-class, or with `unknown`/`not_mentioned` classes).

---

## Requirements

The scripts assume:

- Python 3.10+
- NVIDIA GPU for model inference (A100-class in SLURM examples)
- Local access to model weights/checkpoints
- Optional PostgreSQL access to MIMIC schemas

Primary Python dependencies used in scripts:

- `torch`
- `transformers`
- `pandas`
- `tqdm`
- `psycopg2`
- `numpy`
- `xlsxwriter` (for annotation package generation)

Example install:

```bash
pip install torch transformers pandas tqdm psycopg2-binary numpy xlsxwriter
```

> Note: exact CUDA/PyTorch build should match your cluster environment.

---

## Environment and path assumptions

Most scripts currently use **hard-coded absolute paths**, for example under:

- `/data/salazarda/data/models/...`
- `/data/salazarda/data/sdoh/...`

Before running, you will likely need to edit:

- model paths in `*_sdoh.py`
- dataset CSV input paths in `*_sdoh.py`
- note-file base directories in `*_sdoh.py` / `utils_llm.py`
- output paths in `utils_llm.py`

The helper functions also assume specific PostgreSQL DB names/ports for MIMIC instances.

---

## Running inference

### Option A: Run directly

```bash
python deepseek_sdoh.py
python gpt_oss_sdoh.py
python mistral_sdoh.py
```

### Option B: Submit through SLURM

```bash
sbatch run_deepseek_mimic4.sh
sbatch run_gptoss_mimic4.sh
sbatch run_mistral_mimic4.sh
```

The SLURM scripts show examples for:

- job resources (`--gres=gpu:a100:1`, RAM, walltime)
- optional PostgreSQL module startup
- project directory setup

---

## Output artifacts

The save routine writes:

1. **JSONL** with per-note metadata + generated text.
2. **CSV** containing cleaned “natural text” versions (square-bracket content removed).

Current output naming pattern (from `save_to_jsonl`) is:

- `sdoh_outputs_<model_name>_<timestamp>.jsonl`
- `sdoh_outputs_<model_name>_<timestamp>_natural.csv`

in the configured output directory.

---

## Sampling and annotation workflow

### 1) Build model-combo samples

Run:

```bash
python code_sampling_MASTER.py
```

This script:

- loads per-category consensus CSVs;
- normalizes labels to valid sets (+ `unknown` fallback);
- computes model triplet combinations;
- samples up to `MAX_PER_COMBO` notes per combination;
- exports per-category sample/stat files plus a master CSV.

### 2) Build annotator packages

Run:

```bash
python code_sampling_annotators.py
```

This script:

- reads the master sample file;
- copies unique note `.txt` files into per-annotator folders;
- creates multi-sheet Excel files with:
  - metadata IDs,
  - clickable note links,
  - model predictions,
  - blank annotation columns,
  - evidence/section fields.

---

## PostgreSQL / MIMIC notes access

`utils_llm.py` includes functions for both:

- SQL-backed retrieval from MIMIC schemas (`mimiciii.noteevents`, `mimiciv_note.discharge`), and
- file-backed retrieval from preprocessed note text files (`{subject_id}_{hadm_id}_{note_id}.txt`).

If you are using local PostgreSQL, update connection parameters (`dbname`, `host`, `port`, credentials) to your environment.

---

## Reproducibility notes

To improve reproducibility across runs:

- pin package versions in an environment file;
- externalize all hard-coded paths into config/env vars;
- save prompt version identifiers with outputs;
- keep input ID lists versioned;
- log model commit/hash when possible.

---

## Data governance and privacy

Clinical notes may contain sensitive patient information.

- Follow your institution’s policies for handling MIMIC and derived artifacts.
- Avoid moving raw note text to unauthorized storage.
- Restrict output sharing if generated files contain identifiable details.

---

## Suggested next improvements

- Add a centralized YAML/JSON config file for all paths and runtime params.
- Add CLI arguments (`argparse`/`typer`) for model selection, batch size, and slices.
- Add schema validation for model JSON outputs.
- Add unit tests for prompt parsing and sampling utilities.
- Add a lightweight evaluation script against adjudicated labels.

---

## License

No license file is currently present in this repository.
If open distribution is intended, add a `LICENSE` file (e.g., MIT, Apache-2.0, or institution-specific terms).
