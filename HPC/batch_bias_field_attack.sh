#!/bin/bash

cd "$HOME/dissertation/attack-on-medvqa"

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
)

for config in "${CONFIGS[@]}"
do
    echo "Submitting $config"

    sbatch \
        --job-name="$config" \
        --export=CONFIG="$config" \
        HPC/bias_field_attack.sh
done
