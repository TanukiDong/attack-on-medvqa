#!/bin/bash
#SBATCH --job-name=bias_field_attack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=./HPC/output/bias_field_attack_%A_%a.out
#SBATCH --error=./HPC/output/bias_field_attack_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

cd "$HOME/dissertation/attack-on-medvqa"

source env.sh

BATCH_SIZE=50
START_INDEX=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END_INDEX=$((START_INDEX + BATCH_SIZE))
OUTPUT_PATH="result/MedVLM-R1/bias_field_attack/cps_X_eps_0pX/batch_${SLURM_ARRAY_TASK_ID}"

echo "============================================================"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "Sample range: ${START_INDEX}:${END_INDEX}"
echo "Output path: ${OUTPUT_PATH}"
python --version
echo "============================================================"

python HPC/bias_field_attack_cps_X_eps_0pX.py \
    --start-index "$START_INDEX" \
    --end-index "$END_INDEX" \
    --output-path "$OUTPUT_PATH"

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
