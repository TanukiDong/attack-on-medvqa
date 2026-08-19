#!/bin/bash

cd "$HOME/dissertation/attack-on-medvqa"

MODALITIES=(
    "mri"
    "ct"
    "us"
)

CONFIGS=(
    "cps_8_eps_0p1"
    "cps_8_eps_0p3"
    "cps_8_eps_0p5"

    "cps_16_eps_0p1"
    "cps_16_eps_0p3"
    "cps_16_eps_0p5"

    "cps_32_eps_0p1"
    "cps_32_eps_0p3"
    "cps_32_eps_0p5"

    "cps_64_eps_0p1"
    "cps_64_eps_0p3"
    "cps_64_eps_0p5"

    "cps_128_eps_0p1"
    "cps_128_eps_0p3"
    "cps_128_eps_0p5"

    "cps_256_eps_0p1"
    "cps_256_eps_0p3"
    "cps_256_eps_0p5"
)

for modality in "${MODALITIES[@]}"; do
    for config in "${CONFIGS[@]}"; do
        echo "Validating modality=$modality config=$config"

        sbatch \
            --job-name="val_${modality}_${config}" \
            HPC/validate_answer.sh "$config" "$modality"
    done
done