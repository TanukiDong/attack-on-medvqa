#!/bin/bash
#SBATCH --job-name=validate_results
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=HPC/output/%x_%j.out
#SBATCH --error=HPC/output/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

export HF_HOME=/mnt/parscratch/users/acp25tw/huggingface_cache

cd "$HOME/dissertation/attack-on-medvqa"

source gpu_env.sh

CONFIG="$1"

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "Configuration: ${CONFIG}"
python --version
echo "============================================================"

python HPC/validate_answer.py --config "$CONFIG"

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
