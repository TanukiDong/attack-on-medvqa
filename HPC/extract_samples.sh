#!/bin/bash
#SBATCH --job-name=extract_samples
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=HPC/output/%x_%j.out
#SBATCH --error=HPC/output/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

export HF_HOME=/mnt/parscratch/users/acp25tw/huggingface_cache

cd "$HOME/dissertation/attack-on-medvqa"

source gpu_env.sh

MODALITY="$1"
SAMPLES="${2:-1000}"

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
python --version
echo "============================================================"

python HPC/extract_samples.py \
    --modality "$MODALITY" \
    --samples "$SAMPLES"

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
