#!/bin/bash
#SBATCH --job-name=mistral_sdoh
#SBATCH --cpus-per-task=10
#SBATCH --mem=150g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=job_slurm_mistral-%j.out
#SBATCH --error=job_slurm_mistral-%j.err

# Si necesitas PostgreSQL:
module load postgresql/16

# Ruta del proyecto
cd /data/salazarda/data/sdoh/SDOH_extraction_sepsis/

# Activar entorno Conda (si lo usas)
# source /usr/local/apps/conda/conda.sh
# conda activate /data/salazarda/pytorch_a100

# Para PyTorch en GPUs de Biowulf (opcional pero recomendado)
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Ejecutar script con Mistral
python mistral_sdoh.py
