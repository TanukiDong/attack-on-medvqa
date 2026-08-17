import json
import re
from time import perf_counter

from common.io import append_jsonl, load_samples
from common.model import (
    VALID_ANSWERS,
    extract_answer,
    run_model,
)


def validate_answers(
    question_path,
    result_directory,
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
    
    attack_results_path = result_directory / "attack_results.jsonl"
    validated_results_path = result_directory / "validated_attack_results.jsonl"
    if overwrite:
        validated_results_path.unlink(missing_ok=True)
        print(f"Overwrite: Removed {validated_results_path}")
    
    start_index, end_index = get_batch_range(result_directory)
    
    # Load samples (for the problem text)
    samples = load_samples(
        question_path=question_path,
        result_path=validated_results_path,
        start_index=start_index,
        end_index=end_index,
        overwrite=False,
        verbose=0
    )
    problems_by_id = {sample["id"]: sample["problem"] for sample in samples}
    
    # Load attack results
    with attack_results_path.open(encoding="utf-8") as file:
        attack_results = [json.loads(line) for line in file if line.strip()]

    # Validation loop
    for result in attack_results:
        question_id = result["question_id"]
        problem = problems_by_id.get(question_id)
        # Skip processed samples
        if problem is None:
            continue
        correct_answer = result["correct_answer"]
        attacked_image_path = result["attacked_image_path"]

        model_output = run_model(
            question=problem,
            image=attacked_image_path,
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
            "validated_answer": validated_answer,
            "validated_attack_success": validated_attack_success,
        }

        append_jsonl(
            validated_results_path,
            validation_result,
        )

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
            print(f"Attack success rate: {attack_success_rate:.2%}")
            print(f"Total evaluation time: {total_time:.2f}s ({total_time / 60:.2f}m)")
        else:
            print("No samples were evaluated.")

    return summary


def get_batch_range(result_directory, batch_size=50):
    """Derive dataset index range from a batch_N directory name."""

    match = re.fullmatch(r"batch_(\d+)", result_directory.name)

    if match is None:
        raise ValueError(f"Expected result directory named 'batch_N', but got: {result_directory.name}")

    batch_index = int(match.group(1))

    start_index = batch_index * batch_size
    end_index = start_index + batch_size

    return start_index, end_index