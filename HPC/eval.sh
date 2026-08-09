#!/bin/bash
#SBATCH --job-name=validate_results
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-8
#SBATCH --output=./HPC/output/validate_results_%A_%a.out
#SBATCH --error=./HPC/output/validate_results_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

cd "$HOME/dissertation/attack-on-medvqa"

source env.sh

CONFIGS=(
    "cps_8_eps_0p1"
    "cps_8_eps_0p3"
    "cps_8_eps_0p5"
    "cps_16_eps_0p1"
    "cps_16_eps_0p3"
    "cps_16_eps_0p5"
    "cps_24_eps_0p1"
    "cps_24_eps_0p3"
    "cps_32_eps_0p3"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "Configuration: ${CONFIG}"
python --version
echo "============================================================"

python HPC/eval_batches_config.py --config "$CONFIG"

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
