#!/bin/bash
#SBATCH --job-name=gptoss_job
#SBATCH --cpus-per-task=10                # Núcleos CPU
#SBATCH --mem=150g                        # Memoria RAM total
#SBATCH --gres=gpu:a100:1                 # GPU A100
#SBATCH --time=12:00:00                    # Tiempo máximo
#SBATCH --partition=gpu                   # Cola con GPUs
#SBATCH --output=job_slurm_gptoss-%j.out  # Log de salida
#SBATCH --error=job_slurm_gptoss-%j.err   # Log de error

module load postgresql/16

# Levantar PostgreSQL para MIMIC-IV
# pg_ctl -D /data/$USER/pg_mimic4 -o "-p 5434" -l /data/$USER/pg_mimic4/pg.log start

cd /data/salazarda/data/sdoh/SDOH_extraction_sepsis/

# Activa tu entorno si lo necesitas
# source /usr/local/apps/conda/conda.sh
# conda activate /data/salazarda/pytorch_a100

# === Ejecuta el script ===
python gpt_oss_sdoh.py

# (Opcional pero buena práctica) parar el servidor al final:
# pg_ctl -D /data/$USER/pg_mimic4 stop
