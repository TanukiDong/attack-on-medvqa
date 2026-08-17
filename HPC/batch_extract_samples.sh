#!/bin/bash

SAMPLES="${1:-1000}"

for modality in mri ct us; do
    echo "Submitting ${modality}: ${SAMPLES} samples"

    sbatch \
        --job-name="extract_${modality}" \
        HPC/extract_samples.sh \
        "$modality" \
        "$SAMPLES"
done