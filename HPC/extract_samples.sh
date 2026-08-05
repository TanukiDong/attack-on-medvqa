#!/bin/bash
#SBATCH --job-name=extract_samples

# SBATCH --partition=gpu
# SBATCH --qos=gpu
# SBATCH --gres=gpu:1
# SBATCH --nodes=1
# SBATCH --cpus-per-task=4
# SBATCH --mem=32G

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

#SBATCH --array=0-2
# SBATCH --array=0-18

# SBATCH --time=00:30:00
#SBATCH --output=./HPC/output/extract_ct_%A_%a.out
#SBATCH --error=./HPC/output/extract_ct_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

cd "$HOME/dissertation/attack-on-medvqa"

source env.sh

samples=$((100 + SLURM_ARRAY_TASK_ID * 50))

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
    --modality ct \
    --samples "$samples" \
    --cpu

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"