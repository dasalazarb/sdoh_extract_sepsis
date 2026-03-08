#!/bin/bash
#SBATCH --job-name=ASSOC_
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=ASSOC_%j.out
#SBATCH --error=ASSOC_%j.err

set -euo pipefail

cd /data/salazarda/data/sdoh/SDOH_extraction_sepsis

python code_get_associations_sdoh_surv.py