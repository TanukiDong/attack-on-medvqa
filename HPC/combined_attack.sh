#!/bin/bash
#SBATCH --job-name=combined_attack
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

MODALITY="${1:-}"
LOSS="${2:-}"
INITIALIZATION="${3:-}"
CONFIG="${4:-}"

if [[ -z "$MODALITY" ]]; then
    echo "ERROR: MODALITY is not set"
    exit 1
fi

if [[ -z "$LOSS" ]]; then
    echo "ERROR: LOSS is not set"
    exit 1
fi

if [[ -z "$INITIALIZATION" ]]; then
    echo "ERROR: INITIALIZATION is not set"
    exit 1
fi

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: CONFIG is not set"
    exit 1
fi

echo "============================================================"
echo "Combined Bias Field + Answer Negation Attack"
echo "============================================================"
echo "Modality:             ${MODALITY}"
echo "Loss:                 ${LOSS}"
echo "Initialization:       ${INITIALIZATION}"
echo "Config:               ${CONFIG}"
echo "Job ID:               ${SLURM_JOB_ID:-N/A}"
echo "Node:                 $(hostname)"
echo "Started:              $(date)"
echo "Working directory:    $(pwd)"
python --version
echo "============================================================"

python HPC/combined_attack.py \
    --modality "${MODALITY}" \
    --loss "${LOSS}" \
    --initialization "${INITIALIZATION}" \
    --config "${CONFIG}"

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
