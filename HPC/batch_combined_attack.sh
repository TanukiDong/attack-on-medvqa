#!/bin/bash

cd "$HOME/dissertation/attack-on-medvqa"

MODALITIES=(
    "mri"
    # "ct"
    # "us"
)
LOSSES=(
    "cross_entropy"
    # "kl"
    # "entropy"
)

INITIALIZATIONS=(
    "random"
    "identity"
)


for MODALITY in "${MODALITIES[@]}"; do

    for LOSS in "${LOSSES[@]}"; do

        for INITIALIZATION in "${INITIALIZATIONS[@]}"; do

            RESULT_DIR="result/MedVLM-R1/bias_field_attack/${MODALITY}/${LOSS}/${INITIALIZATION}"

            if [[ ! -d "${RESULT_DIR}" ]]; then
                echo "Skipping missing directory:"
                echo "  ${RESULT_DIR}"
                continue
            fi

            for CONFIG_DIR in "${RESULT_DIR}"/cps_*; do

                # Skip if no cps_* directories matched
                if [[ ! -d "${CONFIG_DIR}" ]]; then
                    continue
                fi

                CONFIG="$(basename "${CONFIG_DIR}")"

                echo "Submitting:"
                echo "  ${MODALITY} ${LOSS} ${INITIALIZATION} ${CONFIG}"

                sbatch \
                    --job-name="combined_${LOSS}_${INITIALIZATION}_${CONFIG}" \
                    HPC/combined_attack.sh \
                    "${MODALITY}" \
                    "${LOSS}" \
                    "${INITIALIZATION}" \
                    "${CONFIG}"

            done

        done

    done

done