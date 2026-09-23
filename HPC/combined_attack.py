import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

LOSS_CHOICES = ("cross_entropy", "entropy", "kl")
INITIALIZATION_CHOICES = ("random", "identity")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.io import (
    append_jsonl,
    load_completed_ids,
    get_attacked_image_path,
    combine_batch,
    find_config)
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


def load_question_an(modality):
    """
    Load answer-negated questions.
    """
    question_path = PROJECT_ROOT / "data" / "OmniMedVQA" / f"sample_{modality}" / "question_an.json"

    if not question_path.exists():
        raise FileNotFoundError(f"Answer-negation question file not found: {question_path}")

    with question_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    return {
        sample["id"]: sample
        for sample in questions
    }


def multimodal_attack(
    source_dir,
    output_dir,
    question_map,
    model,
    processor,
    generation_config,
    overwrite=False,
):
    """
    Run bias field and answer negation attack
    """

    config_name = source_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "inference_results.jsonl"


    if overwrite:
        result_path.unlink(missing_ok=True)
        print(f"Overwrite: removed {result_path}")

    completed_ids = load_completed_ids(result_path=result_path, overwrite=overwrite)

    samples = combine_batch(source_dir)

    remaining_samples = [
        item
        for item in samples
        if item["bias_result"]["question_id"] not in completed_ids
    ]


    if not remaining_samples:
        print(f"{config_name} is already complete.")
        return

    # ========================================================
    # Inference
    # ========================================================

    for item in tqdm(remaining_samples, desc=config_name):

        batch_directory = item["batch_directory"]
        bias_result = item["bias_result"]
        question_id = bias_result["question_id"]

        # Answer negation
        sample = question_map.get(question_id)
        
        if sample is None:
            raise KeyError(f"{question_id} not found in question_an.json.")

        problem = sample["problem"]
        correct_answer = sample["solution"]


        # Bias-field attack
        attacked_image_path = get_attacked_image_path(batch_directory=batch_directory, question_id=question_id)

        # Combined attack
        model_output = run_model(
            question=problem,
            image=attacked_image_path,
            model=model,
            processor=processor,
            generation_config=generation_config,
        )

        predicted_answer = extract_answer(model_output, tag="answer")

        attack_success = (
            predicted_answer in VALID_ANSWERS
            and predicted_answer != correct_answer
        )

        result = {
            "question_id": question_id,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "attack_success": attack_success,
        }

        append_jsonl(result_path, result)

def main():

    parser = argparse.ArgumentParser(
        description=("Run combined bias field + answer negation attack.")
    )

    parser.add_argument(
        "--modality",
        required=True,
        choices=["mri", "ct", "us"],
        help="Imaging modality.",
    )

    parser.add_argument(
        "--loss",
        choices=LOSS_CHOICES,
        default=None,
        help=("Loss function. If omitted, all available losses are processed."),
    )

    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default=None,
        help=("Initialization method. If omitted, all available initializations are processed."),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Bias field configuration, e.g. cps_32_eps_0p3. If omitted, all available configurations are processed."),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Delete existing inference_results.jsonl before running."),
    )

    args = parser.parse_args()

    source_modality_root = PROJECT_ROOT / "result" / "MedVLM-R1" / "bias_field_attack" / args.modality
    output_modality_root = PROJECT_ROOT / "result" / "MedVLM-R1" / "multimodal_attack" / args.modality

    if not source_modality_root.exists():
        raise FileNotFoundError(f"Bias-field modality directory not found: {source_modality_root}")

    experiments = find_config(
        source_modality_root=source_modality_root,
        loss=args.loss,
        initialization=args.initialization,
        config=args.config,
    )


    # Load answer-negated questions
    question_map = load_question_an(modality=args.modality)


    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.Run this script on a GPU node.")

    model, processor, generation_config = load_model(MODEL_CONFIG)

    # Run attack
    for experiment_index, source_dir in enumerate(experiments, start=1):

        relative_experiment_path = (source_dir.relative_to(source_modality_root))
        output_dir = (output_modality_root / relative_experiment_path)

        multimodal_attack(
            source_dir=source_dir,
            output_dir=output_dir,
            question_map=question_map,
            model=model,
            processor=processor,
            generation_config=generation_config,
            overwrite=args.overwrite,
        )

    print("Finished combined multimodal attack.")

if __name__ == "__main__":
    main()