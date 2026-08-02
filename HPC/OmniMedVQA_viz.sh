#!/bin/bash
#SBATCH --job-name=omni_viz
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=./output/OmniMedVQA_viz_output.txt
#SBATCH --error=./output/OmniMedVQA_viz_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twanavit1@sheffield.ac.uk

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit --driver-memory 32g ./OmniMedVQA_viz.py
