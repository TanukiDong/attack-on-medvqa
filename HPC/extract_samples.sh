#!/bin/bash
#SBATCH --job-name=extract_samples
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=./output/extract_mri_output.txt
#SBATCH --error=./output/extract_mri_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

cd "$HOME/dissertation/attack-on-medvqa"

source env.sh

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
python --version
echo "============================================================"

python HPC/extract_samples.py \
    --data-root /mnt/parscratch/users/acp25tw/datasets/OmniMedVQA-V2 \
    --results HPC/output/clean_results_mri.csv \
    --output-root data/OmniMedVQA/sample_mri \
    --selection-results HPC/output/correct_mri_samples.csv \
    --samples-per-modality 100 \
    --overwrite
    
spark-submit --driver-memory 32g ./OmniMedVQA_viz.py

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"