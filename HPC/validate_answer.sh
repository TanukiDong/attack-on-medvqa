#!/bin/bash
#SBATCH --job-name=validate_results
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=HPC/output/%x_%j.out
#SBATCH --error=HPC/output/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

export HF_HOME=/mnt/parscratch/users/acp25tw/huggingface_cache

cd "$HOME/dissertation/attack-on-medvqa"

source gpu_env.sh

CONFIG="${1:-}"
MODALITY="${2:-}"

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: CONFIG is not set"
    exit 1
fi

if [[ -z "$MODALITY" ]]; then
    echo "ERROR: MODALITY is not set"
    exit 1
fi

echo "============================================================"
echo "Config: $CONFIG"
echo "Modality: $MODALITY"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
python --version
echo "============================================================"

python HPC/validate_answer.py \
    --config "$CONFIG" \
    --modality "$MODALITY"

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
