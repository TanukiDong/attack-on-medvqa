import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from transformers import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add src/ to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.io import (
    append_jsonl,
    find_project_root,
    load_completed_ids,
)
from common.model import (
    VALID_ANSWERS,
    extract_answer,
    load_model,
    run_model,
)


MODEL_CONFIG = {
    "path": "JZPeterPan/MedVLM-R1",
    "hf_cache": "/mnt/parscratch/users/acp25tw/huggingface_cache",
}

VALID_MODALITIES = ("mri", "ct", "us")


def run_answer_negation_inference(
    modality,
    overwrite=False,
    verbose=1,
):
    """Run inference on answer-negated OmniMedVQA samples."""
    start_time = perf_counter()

    # Project paths
    project_root = find_project_root()

    sample_root = (
        project_root
        / "data"
        / "OmniMedVQA"
        / f"sample_{modality}"
    )

    question_path = sample_root / "question_an.json"

    output_directory = (
        project_root
        / "result"
        / "MedVLM-R1"
        / "answer_negation"
        / modality
    )

    result_path = output_directory / "inference_results.jsonl"

    # Check input
    if not question_path.exists():
        raise FileNotFoundError(
            f"Answer-negation question file not found: {question_path}"
        )

    # Initialize output
    output_directory.mkdir(parents=True, exist_ok=True)

    if overwrite:
        result_path.unlink(missing_ok=True)
        print(f"Overwrite: Removed {result_path}")

    # Seed
    set_seed(42, deterministic=True)

    # Load model
    model, processor, generation_config = load_model(
        MODEL_CONFIG
    )

    # Load answer-negated samples
    with question_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    # Load already completed samples
    completed_ids = load_completed_ids(
        result_path=result_path,
        overwrite=overwrite,
    )

    total_evaluated = 0
    total_successful = 0
    total_failed = 0
    total_invalid = 0
    total_skipped = 0

    # Inference loop
    for index, sample in enumerate(samples):

        question_id = sample["id"]

        # Skip previously processed samples
        if question_id in completed_ids:
            total_skipped += 1
            continue

        problem = sample["problem"]
        expected_answer = sample["solution"]

        image_path = sample_root / sample["image"][0]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found for {question_id}: {image_path}"
            )

        # Run model on original image + answer-negated question
        model_output = run_model(
            question=problem,
            image=image_path,
            model=model,
            processor=processor,
            generation_config=generation_config,
        )

        # Extract predicted answer letter
        predicted_answer = extract_answer(
            model_output,
            tag="answer",
        )

        valid_answer = predicted_answer in VALID_ANSWERS

        # The expected answer letter stays the same after answer negation.
        #
        # Example:
        #
        # Original:
        #   B) MRI
        #
        # Answer-negated:
        #   B) No correct answer
        #
        # Therefore B is still the expected answer.
        #
        # Attack succeeds if the model changes to another valid answer.
        attack_success = (
            valid_answer
            and predicted_answer != expected_answer
        )

        # Save result
        result = {
            "question_id": question_id,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "valid_answer": valid_answer,
            "attack_success": attack_success,
            "model_output": model_output,
        }

        append_jsonl(
            result_path,
            result,
        )

        # Statistics
        total_evaluated += 1

        if not valid_answer:
            total_invalid += 1
        elif attack_success:
            total_successful += 1
        else:
            total_failed += 1

        # Print progress
        if verbose:
            print(
                f"{question_id} | "
                f"expected={expected_answer} | "
                f"prediction={predicted_answer} | "
                f"attack_success={attack_success} | "
                f"{index + 1}/{len(samples)}"
            )

        if verbose > 1:
            print(f"Question: {problem}")
            print(f"Model output:\n{model_output}")
            print()

    # Summary
    total_time = perf_counter() - start_time

    if total_evaluated > 0:
        attack_success_rate = (
            total_successful / total_evaluated
        )
    else:
        attack_success_rate = None

    print("\n========== ANSWER NEGATION RESULTS ==========")
    print(f"Evaluated:           {total_evaluated}")
    print(f"Successful attacks:  {total_successful}")
    print(f"Failed attacks:      {total_failed}")
    print(f"Invalid answers:     {total_invalid}")
    print(f"Skipped:             {total_skipped}")

    if attack_success_rate is not None:
        print(
            f"Attack success rate: {attack_success_rate:.2%}"
        )

    print(
        f"Total runtime: "
        f"{total_time:.2f}s "
        f"({total_time / 60:.2f}m)"
    )

    return {
        "evaluated": total_evaluated,
        "successful": total_successful,
        "failed": total_failed,
        "invalid": total_invalid,
        "skipped": total_skipped,
        "attack_success_rate": attack_success_rate,
        "total_time": total_time,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MedVLM-R1 inference on "
            "answer-negated OmniMedVQA questions."
        )
    )

    parser.add_argument(
        "--modality",
        required=True,
        choices=VALID_MODALITIES,
        help="Modality to evaluate: mri, ct, or us.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference results.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_answer_negation_inference(
        modality=args.modality,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()