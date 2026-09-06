import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm


# ============================================================
# Constants
# ============================================================

LOSS_CHOICES = (
    "cross_entropy",
    "kl",
    "entropy",
)

INITIALIZATION_CHOICES = (
    "random",
    "identity",
)


# ============================================================
# Project setup
# ============================================================

def find_project_root():
    """Find project root by searching parent directories for .git."""
    for path in Path(__file__).resolve().parents:
        if (path / ".git").exists():
            return path

    raise FileNotFoundError(
        "Could not find project root. "
        "Expected a parent directory containing .git."
    )


PROJECT_ROOT = find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from common.io import append_jsonl
from common.model import (
    VALID_ANSWERS,
    extract_answer,
    load_model,
    run_model,
)


# ============================================================
# Model configuration
# ============================================================

MODEL_CONFIG = {
    "path": "JZPeterPan/MedVLM-R1",
    "hf_cache": "/mnt/parscratch/users/acp25tw/huggingface_cache",
}


# ============================================================
# JSON helpers
# ============================================================

def load_jsonl(path):
    """Load records from a JSONL file."""
    path = Path(path)

    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}."
                ) from error

    return records


def load_question_map(modality):
    """
    Load answer-negated questions.

    Expected:
        data/OmniMedVQA/sample_mri/question_an.json
        data/OmniMedVQA/sample_ct/question_an.json
        data/OmniMedVQA/sample_us/question_an.json
    """

    question_path = (
        PROJECT_ROOT
        / "data"
        / "OmniMedVQA"
        / f"sample_{modality}"
        / "question_an.json"
    )

    if not question_path.exists():
        raise FileNotFoundError(
            f"Answer-negation question file not found:\n"
            f"{question_path}"
        )

    with question_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    return {
        sample["id"]: sample
        for sample in questions
    }


def load_completed_ids(result_path, overwrite=False):
    """
    Load already processed question IDs.

    Allows an interrupted HPC job to resume without
    repeating completed inference.
    """

    result_path = Path(result_path)

    if overwrite or not result_path.exists():
        return set()

    return {
        record["question_id"]
        for record in load_jsonl(result_path)
    }


# ============================================================
# Batch sorting
# ============================================================

def batch_sort_key(path):
    """Sort batch_0, batch_1, ..., batch_19 numerically."""

    match = re.fullmatch(
        r"batch_(\d+)",
        path.name,
    )

    if match is None:
        return float("inf")

    return int(match.group(1))


# ============================================================
# Bias-field helpers
# ============================================================

def get_attacked_image_path(
    batch_directory,
    question_id,
):
    """
    Get the saved bias-field attacked image.

    Example:

        mod-mri:000000

    becomes:

        attacked_images/mod-mri_000000_biased.png
    """

    safe_id = str(question_id).replace(
        ":",
        "_",
    )

    attacked_image_path = (
        batch_directory
        / "attacked_images"
        / f"{safe_id}_biased.png"
    )

    if not attacked_image_path.exists():
        raise FileNotFoundError(
            f"Attacked image not found for {question_id}:\n"
            f"{attacked_image_path}"
        )

    return attacked_image_path


def collect_bias_field_samples(
    source_config_directory,
):
    """
    Collect attack results from all batch directories
    belonging to one bias-field configuration.

    Returns:
        [
            {
                "batch_directory": Path(...),
                "bias_result": {...},
            },
            ...
        ]
    """

    batch_directories = sorted(
        [
            path
            for path in source_config_directory.glob("batch_*")
            if path.is_dir()
        ],
        key=batch_sort_key,
    )

    if not batch_directories:
        raise RuntimeError(
            f"No batch directories found under:\n"
            f"{source_config_directory}"
        )

    samples = []
    seen_ids = set()

    for batch_directory in batch_directories:

        attack_results_path = (
            batch_directory
            / "attack_results.jsonl"
        )

        if not attack_results_path.exists():
            raise FileNotFoundError(
                f"Missing attack results:\n"
                f"{attack_results_path}"
            )

        batch_results = load_jsonl(
            attack_results_path
        )

        print(
            f"  {batch_directory.name}: "
            f"{len(batch_results)} samples"
        )

        for bias_result in batch_results:

            question_id = bias_result["question_id"]

            if question_id in seen_ids:
                raise RuntimeError(
                    f"Duplicate question ID found "
                    f"across batches: {question_id}"
                )

            seen_ids.add(question_id)

            samples.append(
                {
                    "batch_directory": batch_directory,
                    "bias_result": bias_result,
                }
            )

    return samples


# ============================================================
# Experiment discovery
# ============================================================

def discover_experiments(
    source_modality_root,
    loss=None,
    initialization=None,
    config=None,
):
    """
    Discover bias-field experiment configurations.

    Examples:

        cross_entropy/random/cps_32_eps_0p3
        cross_entropy/identity/cps_64_eps_0p1
        kl/random/cps_16_eps_0p5
        entropy/identity/cps_8_eps_0p1
    """

    experiments = []

    # ========================================================
    # Loss selection
    # ========================================================

    if loss is not None:
        losses = [loss]
    else:
        losses = LOSS_CHOICES

    for loss_name in losses:

        loss_directory = (
            source_modality_root
            / loss_name
        )

        # When processing everything, a loss may simply
        # not have been generated yet.
        if not loss_directory.exists():

            if loss is not None:
                raise FileNotFoundError(
                    f"Loss directory not found:\n"
                    f"{loss_directory}"
                )

            print(
                f"Skipping unavailable loss: "
                f"{loss_name}"
            )

            continue

        # ====================================================
        # Initialization selection
        # ====================================================

        if initialization is not None:
            initializations = [initialization]
        else:
            initializations = INITIALIZATION_CHOICES

        for initialization_name in initializations:

            initialization_directory = (
                loss_directory
                / initialization_name
            )

            if not initialization_directory.exists():

                if initialization is not None:
                    print(
                        "Skipping unavailable combination: "
                        f"{loss_name}/{initialization_name}"
                    )

                continue

            # ================================================
            # Config selection
            # ================================================

            if config is not None:

                config_directories = [
                    initialization_directory
                    / config
                ]

            else:

                # No sorting required for configurations.
                config_directories = [
                    path
                    for path in initialization_directory.iterdir()
                    if (
                        path.is_dir()
                        and path.name.startswith("cps_")
                    )
                ]

            for config_directory in config_directories:

                if not config_directory.exists():

                    print(
                        "Skipping missing configuration: "
                        f"{config_directory}"
                    )

                    continue

                batch_directories = [
                    path
                    for path in config_directory.glob("batch_*")
                    if path.is_dir()
                ]

                if not batch_directories:

                    print(
                        "Skipping configuration with no batches: "
                        f"{config_directory}"
                    )

                    continue

                experiments.append(
                    config_directory
                )

    return experiments


# ============================================================
# Summary
# ============================================================

def print_config_summary(result_path):
    """
    Print summary based on the complete result file.

    Reading the complete file means the summary remains
    correct after resuming an interrupted job.
    """

    results = load_jsonl(
        result_path
    )

    total = len(results)

    successful = sum(
        bool(result["attack_success"])
        for result in results
    )

    failed = total - successful

    attack_success_rate = (
        successful / total
        if total > 0
        else None
    )

    print()
    print("-" * 60)

    print(
        f"Result:     {result_path}"
    )

    print(
        f"Evaluated:  {total}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    if attack_success_rate is not None:

        print(
            f"ASR:        "
            f"{attack_success_rate:.2%}"
        )

    print("-" * 60)


# ============================================================
# Run one configuration
# ============================================================

def run_config(
    source_config_directory,
    output_config_directory,
    question_map,
    model,
    processor,
    generation_config,
    overwrite=False,
    verbose=1,
):
    """
    Run answer-negation inference using bias-field attacked
    images from every batch of one configuration.

    All batches are combined into:

        <config>/inference_results.jsonl
    """

    config_name = (
        source_config_directory.name
    )

    output_config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_config_directory
        / "inference_results.jsonl"
    )

    # ========================================================
    # Overwrite
    # ========================================================

    if overwrite:

        result_path.unlink(
            missing_ok=True
        )

        print(
            f"Overwrite: removed {result_path}"
        )

    # ========================================================
    # Resume
    # ========================================================

    completed_ids = load_completed_ids(
        result_path=result_path,
        overwrite=overwrite,
    )

    # ========================================================
    # Collect samples from all batches
    # ========================================================

    print()
    print(
        f"Collecting samples for {config_name}..."
    )

    samples = collect_bias_field_samples(
        source_config_directory
    )

    total_samples = len(samples)

    remaining_samples = [
        item
        for item in samples
        if (
            item["bias_result"]["question_id"]
            not in completed_ids
        )
    ]

    print()
    print(
        f"Total samples:     {total_samples}"
    )

    print(
        f"Already completed: {len(completed_ids)}"
    )

    print(
        f"Remaining:         {len(remaining_samples)}"
    )

    # ========================================================
    # Already complete
    # ========================================================

    if not remaining_samples:

        print(
            f"{config_name} is already complete."
        )

        print_config_summary(
            result_path
        )

        return

    # ========================================================
    # Inference
    # ========================================================

    for item in tqdm(
        remaining_samples,
        desc=config_name,
    ):

        batch_directory = (
            item["batch_directory"]
        )

        bias_result = (
            item["bias_result"]
        )

        question_id = (
            bias_result["question_id"]
        )

        # ----------------------------------------------------
        # Answer-negated question
        # ----------------------------------------------------

        sample = question_map.get(
            question_id
        )

        if sample is None:

            raise KeyError(
                f"{question_id} exists in the "
                f"bias-field results but was not found "
                f"in question_an.json."
            )

        problem = (
            sample["problem"]
        )

        correct_answer = (
            sample["solution"]
            .strip()
            .upper()
        )

        if correct_answer not in VALID_ANSWERS:

            raise ValueError(
                f"Invalid correct answer "
                f"for {question_id}: "
                f"{correct_answer!r}"
            )

        # ----------------------------------------------------
        # Bias-field attacked image
        # ----------------------------------------------------

        attacked_image_path = (
            get_attacked_image_path(
                batch_directory=batch_directory,
                question_id=question_id,
            )
        )

        # ----------------------------------------------------
        # Combined inference:
        #
        # bias-field attacked image
        #           +
        # answer-negated question
        # ----------------------------------------------------

        model_output = run_model(
            question=problem,
            image=attacked_image_path,
            model=model,
            processor=processor,
            generation_config=generation_config,
        )

        predicted_answer = extract_answer(
            model_output,
            tag="answer",
        )

        # ----------------------------------------------------
        # Attack success
        # ----------------------------------------------------

        attack_success = (
            predicted_answer in VALID_ANSWERS
            and predicted_answer != correct_answer
        )

        # ----------------------------------------------------
        # Save minimal result
        # ----------------------------------------------------

        result = {
            "question_id": question_id,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "attack_success": attack_success,
        }

        append_jsonl(
            result_path,
            result,
        )

        # ----------------------------------------------------
        # Verbose output
        # ----------------------------------------------------

        if verbose > 1:

            print()
            print(
                f"ID:         {question_id}"
            )

            print(
                f"Batch:      {batch_directory.name}"
            )

            print(
                f"Correct:    {correct_answer}"
            )

            print(
                f"Prediction: {predicted_answer}"
            )

            print(
                f"Success:    {attack_success}"
            )

            if verbose > 2:

                print(
                    f"Question: {problem}"
                )

                print(
                    f"Image: {attacked_image_path}"
                )

                print(
                    f"Output:\n{model_output}"
                )

    # ========================================================
    # Final summary
    # ========================================================

    print_config_summary(
        result_path
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run combined bias-field + answer-negation "
            "inference using MedVLM-R1."
        )
    )

    # ========================================================
    # Arguments
    # ========================================================

    parser.add_argument(
        "--modality",
        required=True,
        choices=[
            "mri",
            "ct",
            "us",
        ],
        help="Imaging modality.",
    )

    parser.add_argument(
        "--loss",
        choices=LOSS_CHOICES,
        default=None,
        help=(
            "Optional loss function. "
            "If omitted, all available losses are processed."
        ),
    )

    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default=None,
        help=(
            "Optional initialization method. "
            "If omitted, all available initializations "
            "are processed."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional bias-field configuration, "
            "e.g. cps_32_eps_0p3. "
            "If omitted, all available configurations "
            "are processed."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing inference_results.jsonl "
            "before running."
        ),
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help=(
            "Verbosity: "
            "1=progress, "
            "2=sample results, "
            "3=full model output."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Root directories
    # ========================================================

    source_modality_root = (
        PROJECT_ROOT
        / "result"
        / "MedVLM-R1"
        / "bias_field_attack"
        / args.modality
    )

    output_modality_root = (
        PROJECT_ROOT
        / "result"
        / "MedVLM-R1"
        / "multimodal_attack"
        / args.modality
    )

    if not source_modality_root.exists():

        raise FileNotFoundError(
            f"Bias-field modality directory not found:\n"
            f"{source_modality_root}"
        )

    # ========================================================
    # Discover experiments
    # ========================================================

    experiments = discover_experiments(
        source_modality_root=source_modality_root,
        loss=args.loss,
        initialization=args.initialization,
        config=args.config,
    )

    if not experiments:

        raise RuntimeError(
            "No matching bias-field experiments were found."
        )

    # ========================================================
    # Print discovered experiments
    # ========================================================

    print()
    print(
        f"Found {len(experiments)} experiment(s):"
    )

    for experiment in experiments:

        relative_path = (
            experiment.relative_to(
                source_modality_root
            )
        )

        print(
            f"  {relative_path}"
        )

    # ========================================================
    # Load answer-negated questions
    # ========================================================

    question_map = load_question_map(
        modality=args.modality
    )

    print()
    print(
        f"Loaded {len(question_map)} "
        f"answer-negated questions."
    )

    # ========================================================
    # GPU
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable. "
            "Run this script on a GPU node."
        )

    # ========================================================
    # Load model once
    # ========================================================

    print()
    print("Loading MedVLM-R1...")

    (
        model,
        processor,
        generation_config,
    ) = load_model(
        MODEL_CONFIG
    )

    # ========================================================
    # Process experiments
    # ========================================================

    for experiment_index, source_config_directory in enumerate(
        experiments,
        start=1,
    ):

        # Preserve:
        #
        # loss/
        # initialization/
        # config/

        relative_experiment_path = (
            source_config_directory.relative_to(
                source_modality_root
            )
        )

        output_config_directory = (
            output_modality_root
            / relative_experiment_path
        )

        print()
        print("=" * 70)

        print(
            f"Experiment "
            f"{experiment_index}/{len(experiments)}"
        )

        print(
            f"Experiment: "
            f"{relative_experiment_path}"
        )

        print(
            f"Source: "
            f"{source_config_directory}"
        )

        print(
            f"Output: "
            f"{output_config_directory}"
        )

        print("=" * 70)

        run_config(
            source_config_directory=source_config_directory,
            output_config_directory=output_config_directory,
            question_map=question_map,
            model=model,
            processor=processor,
            generation_config=generation_config,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

    # ========================================================
    # Done
    # ========================================================

    print()
    print(
        "Finished combined bias-field + "
        "answer-negation inference."
    )


if __name__ == "__main__":
    main()