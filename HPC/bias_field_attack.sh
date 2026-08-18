#!/bin/bash
#SBATCH --job-name=bias_field_attack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --array=0-19
#SBATCH --output="HPC/output/bias_field_attack_%x_%A_%a.out"
#SBATCH --error="HPC/output/bias_field_attack_%x_%A_%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

export HF_HOME=/mnt/parscratch/users/acp25tw/huggingface_cache

cd "$HOME/dissertation/attack-on-medvqa"

source gpu_env.sh

if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG is not set"
    exit 1
fi

echo "============================================================"
echo "Config: $CONFIG"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
python --version
echo "============================================================"

python HPC/bias_field_attack.py \
    --config "$CONFIG"

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
