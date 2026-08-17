#!/bin/bash

cd "$HOME/dissertation/attack-on-medvqa"

CONFIGS=(
    "configs/cps_8_eps_0p1.yaml"
    "configs/cps_8_eps_0p3.yaml"
    "configs/cps_8_eps_0p5.yaml"

    "configs/cps_16_eps_0p1.yaml"
    "configs/cps_16_eps_0p3.yaml"
    "configs/cps_16_eps_0p5.yaml"

    "configs/cps_32_eps_0p1.yaml"
    "configs/cps_32_eps_0p3.yaml"
    "configs/cps_32_eps_0p5.yaml"

    "configs/cps_64_eps_0p1.yaml"
    "configs/cps_64_eps_0p3.yaml"
    "configs/cps_64_eps_0p5.yaml"
)

for config in "${CONFIGS[@]}"
do
    job_name="$(basename "$config" .yaml)"

    echo "Submitting $config as $job_name"

    sbatch \
        --job-name="$job_name" \
        --export=CONFIG="$config" \
        HPC/bias_field_attack.sh
done
