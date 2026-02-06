#!/bin/bash
#SBATCH --job-name=gptoss_job
#SBATCH --cpus-per-task=10                # Núcleos CPU
#SBATCH --mem=120g                       # Memoria RAM total
#SBATCH --gres=gpu:a100:1                # GPU A100
#SBATCH --time=3:00:00                  # Tiempo máximo (24h)
#SBATCH --partition=gpu                  # Cola con GPUs
#SBATCH --output=gptoss_job_slurm-%j.out            # Log de salida
#SBATCH --error=gptoss_job_slurm-%j.err             # Log de error

module load postgresql/16
pg_ctl -D /data/$USER/pg_mimic3 -o "-p 5433" -l /data/$USER/pg_mimic3/pg.log start

cd /data/salazarda/data/sdoh/SDOH_extraction_sepsis/

# source /usr/local/apps/conda/conda.sh
# conda activate /data/salazarda/pytorch_a100

# === Ejecuta el script ===
python deepseek_sdoh.py
