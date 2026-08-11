#!/bin/bash
#SBATCH --job-name=bias_field_attack
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=5:30:00
#SBATCH --output=./HPC/output/bias_field_candidates_generation_output.txt
#SBATCH --error=./HPC/output/bias_field_candidates_generation_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

cd "$HOME/dissertation/attack-on-medvqa"

source gpu_env.sh

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
python --version
echo "============================================================"

python -u HPC/bias_field_candidates_generation.py

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"

