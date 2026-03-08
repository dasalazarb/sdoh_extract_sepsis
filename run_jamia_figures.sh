#!/bin/bash
#SBATCH --job-name=jamiaFig
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=jamiaFig_%j.out
#SBATCH --error=jamiaFig_%j.err

set -euo pipefail

cd /data/salazarda/data/sdoh/SDOH_extraction_sepsis

python code_results_make_jamia_figures.py