import json
from time import perf_counter

import torch

from common.io import (
    append_jsonl,
    find_project_root,
    load_validation_data,
    resolve_project_path,
)
from common.model import (
    VALID_ANSWERS,
    extract_answer,
    run_model,
)
from common.preprocess import load_image_tensor

def validate_answers(
    result_directory,
    modality,
    model,
    processor,
    generation_config,
    overwrite=False,
    verbose=1,
):
    start_time = perf_counter()
    
    total_evaluated = 0
    total_successful = 0
    total_failed = 0
    total_skipped = 0
    
    project_root = find_project_root()
    attack_results_path = result_directory / "attack_results.jsonl"
    validated_results_path = result_directory / "validated_attack_results.jsonl"
    
    if overwrite:
        validated_results_path.unlink(missing_ok=True)
        print(f"Overwrite: Removed {validated_results_path}")
    
    # Load validation data
    problems_by_id, completed_ids = load_validation_data(
        modality=modality,
        validated_results_path=validated_results_path,
        overwrite=overwrite,
    )

    # Load attack results
    with attack_results_path.open(encoding="utf-8") as file:
        attack_results = [json.loads(line) for line in file if line.strip()]

    # Validation loop
    for result in attack_results:
        question_id = result["question_id"]
        problem = problems_by_id.get(question_id)
    
        # Skip processed samples
        if question_id in completed_ids:
            total_skipped += 1
            continue
        
        # Skip unsuccessful attacks
        if not result.get("attack_success", False):
            validation_result = {
                **result,
                "validated": False,
                "validated_answer": None,
                "validated_attack_success": None,
            }

            append_jsonl(validated_results_path, validation_result)
            total_skipped += 1
            continue
        
        correct_answer = result["correct_answer"]
        
        # Path
        original_image_path = resolve_project_path(result["original_image_path"], project_root)
        bias_field_path = resolve_project_path(result["bias_field_path"], project_root)
        
        # Create adversarial 
        image = load_image_tensor(original_image_path)
        bias_field = torch.load(bias_field_path, map_location=image.device,).float()
        adversarial_image = torch.clamp(image * bias_field, min=0, max=1)

        model_output = run_model(
            question=problem,
            image=adversarial_image,
            model=model,
            processor=processor,
            generation_config=generation_config,
        )

        validated_answer = extract_answer(model_output, tag="answer")

        validated_attack_success = (
            validated_answer in VALID_ANSWERS
            and validated_answer != correct_answer
        )

        total_evaluated += 1

        if validated_attack_success:
            total_successful += 1
        else:
            total_failed += 1

        validation_result = {
            **result,
            "validated": True,
            "validated_answer": validated_answer,
            "validated_attack_success": validated_attack_success,
        }

        append_jsonl(validated_results_path, validation_result)

        if verbose:
            print(f"ID: {question_id}")
            if verbose > 1:
                print(f"Question: {problem}")
                print(f"Correct answer: {correct_answer}")
                print(f"Validated answer: {validated_answer}")
            print(f"Attack success: {validated_attack_success}")

            if verbose > 1:
                print(f"Model output:\n{model_output}")

    total_time = perf_counter() - start_time
    
    if total_evaluated > 0:
        attack_success_rate = total_successful / total_evaluated
    else:
        attack_success_rate = None

    summary = {
        "evaluated": total_evaluated,
        "successful": total_successful,
        "failed": total_failed,
        "attack_success_rate": attack_success_rate,
        "total_time": total_time,
    }

    if verbose:
        if total_evaluated:
            print(f"Evaluated:          {total_evaluated}")
            print(f"Successful attacks: {total_successful}")
            print(f"Failed attacks:     {total_failed}")
            print(f"Skipped:            {total_skipped}")
            print(f"Attack success rate: {attack_success_rate:.2%}")
            print(f"Total evaluation time: {total_time:.2f}s ({total_time / 60:.2f}m)")
        else:
            print("No samples were evaluated.")

    return summary