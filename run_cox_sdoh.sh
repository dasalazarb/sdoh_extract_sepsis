#!/bin/bash
#SBATCH --job-name=cox
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=cox_%j.out
#SBATCH --error=cox_%j.err

set -euo pipefail

cd /data/salazarda/data/sdoh/SDOH_extraction_sepsis

python code_cox_survival_model.py