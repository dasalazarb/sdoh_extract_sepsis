#!/bin/bash
#SBATCH --job-name=jamia_desc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --partition=norm
#SBATCH --output=jamia_desc_%j.out
#SBATCH --error=jamia_desc_%j.err

set -euo pipefail

echo "============================================================"
echo "JAMIA descriptives job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================================"

# ===== RUTA DEL PROYECTO =====
WORKDIR="/data/salazarda/data/sdoh"

# ===== PYTHON A USAR =====
PYTHON_BIN="/usr/local/Anaconda/envs/py3.10/bin/python"

# ===== SCRIPT =====
SCRIPT_NAME="code_make_jamia_descriptives.py"

cd "$WORKDIR"

echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "Script: $SCRIPT_NAME"

# Verificaciones
if [ ! -f "$SCRIPT_NAME" ]; then
  echo "[ERROR] No existe el script: $WORKDIR/$SCRIPT_NAME"
  exit 1
fi

if [ ! -f "sdoh_all_notes_with_llm_strata.csv" ]; then
  echo "[ERROR] No existe el input CSV en el mismo folder del script:"
  echo "        $WORKDIR/sdoh_all_notes_with_llm_strata.csv"
  exit 1
fi

echo "Input CSV encontrado."
echo "Ejecutando..."

"$PYTHON_BIN" "$SCRIPT_NAME"

echo "============================================================"
echo "Job finished successfully"
echo "Date: $(date)"
echo "============================================================"