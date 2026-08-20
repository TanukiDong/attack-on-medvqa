import sys
from time import perf_counter

import torch

from tqdm.auto import tqdm
from transformers import set_seed

from attacks.bias_field.attack import attack_bf
from common.preprocess import MODALITY, load_image_tensor, tensor_to_pil
from common.io import (
    find_project_root,
    append_attack_history,
    append_jsonl,
    get_output_paths,
    initialize_output,
    load_samples,
    relative_to_project,
    resolve_project_path,
    load_config,
)
from common.model import (
    extract_answer_token_ids,
    load_model,
    run_model,
)

def run_bias_field_attack(
    config_path,
    modality,
    start_index=0,
    end_index=None,
    output_path=None,
    output_subpath=None,
    overwrite=False):
        
    # Path
    project_root = find_project_root()
    sample_root = project_root / "data" / "OmniMedVQA" / f"sample_{modality}"
    question_path = sample_root / "question.json"
    
    config_path = resolve_project_path(config_path, project_root)
    config = load_config(config_path)

    experiment_config = config["experiment"]
    model_config = config["model"]
    attack_config = config["attack"]
    bias_config = config["bias_field"]

    verbose = experiment_config.get("verbose", 1)
    
    experiment_name = config_path.stem
    initialization = "random" if bias_config["random_start"] else "identity"

    # Override default output path
    if output_path is not None:
        output_directory = resolve_project_path(output_path, project_root)
    else:
        output_directory = project_root / "result" / "MedVLM-R1" / "bias_field_attack" / initialization
        
    output_directory = output_directory / modality / experiment_name / initialization
    
    # Bacth output for HPC
    if output_subpath is not None:
        output_directory = output_directory / output_subpath

    output_paths = initialize_output(
        output_directory=output_directory,
        config_path=config_path,
        overwrite=overwrite,
    )

    # Load samples
    samples = load_samples(
        question_path=question_path,
        result_path=output_paths["result_path"],
        modality=MODALITY[modality],
        start_index=start_index,
        end_index=end_index,
        overwrite=overwrite
    )

    # Device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    # Seed
    set_seed(experiment_config.get("seed", 42), deterministic=True)

    # Model
    model, processor, generation_config = load_model(model_config)

    answer_token_ids = extract_answer_token_ids(processor.tokenizer, verbose=verbose)

    # Attack loop
    torch.cuda.synchronize()
    total_start = perf_counter()

    for sample_index, sample in enumerate(tqdm(samples, desc=f"{modality.upper()} VQA tasks", disable=not sys.stdout.isatty())):

        torch.cuda.synchronize()
        sample_start = perf_counter()

        question_id = sample["id"]
        image_path = sample_root / sample["image"][0]

        image = load_image_tensor(image_path)
        problem = sample["problem"]
        solution = sample["solution"]

        if verbose:
            print(f"\nProcessing sample {sample_index + 1}/{len(samples)}: ID={question_id}")

        # Clean inference
        clean_output = run_model(
            question=problem,
            image=image_path,
            model=model,
            processor=processor,
            generation_config=generation_config,
        )

        # Attack
        adversarial_output, attack_history = attack_bf(
            image=image,
            problem=problem,
            target=solution,
            reference_output=clean_output,
            model=model,
            processor=processor,
            generation_config=generation_config,
            answer_token_ids=answer_token_ids,
            attack_config=attack_config,
            bias_config=bias_config,
            verbose=verbose,
        )

        # Output paths
        attacked_image_path, bias_field_path = get_output_paths(
            question_id=question_id,
            attacked_image_directory=output_paths["attacked_image_directory"],
            bias_field_directory=output_paths["bias_field_directory"],
        )

        # Save adversarial image and bias field
        tensor_to_pil(adversarial_output["image"]).save(attacked_image_path)
        torch.save(adversarial_output["bias_field"], bias_field_path)

        # Print outputs
        if verbose > 1:
            print("\n========== CLEAN MODEL OUTPUT ==========")
            print(repr(clean_output))
            print("========================================")
            print("======= ADVERSARIAL MODEL OUTPUT =======")
            print(repr(adversarial_output["output"]))
            print("========================================")
            print(f"Attack Success: {adversarial_output['attack_success']}")

        # Result
        attack_result = {
            "question_id": question_id,
            "correct_answer": solution,
            "adversarial_answer": adversarial_output["answer"],
            "attack_success": adversarial_output["attack_success"],
            "steps": len(attack_history),

            # Best candidate
            "best_step": adversarial_output["step"],
            "best_loss": adversarial_output["loss"],
            "image_loss": adversarial_output["image_loss"],

            # Files
            "original_image_path": relative_to_project(image_path,project_root,),
            "attacked_image_path": relative_to_project(attacked_image_path,project_root),
            "bias_field_path": relative_to_project(bias_field_path, project_root),
        }

        # Save results
        append_jsonl(output_paths["result_path"], attack_result)
        append_attack_history(output_paths["history_path"], question_id, attack_history)


        torch.cuda.synchronize()
        sample_time = perf_counter() - sample_start

        print(f"Saved results for ID:{question_id} | {sample_index + 1}/{len(samples)} | Time: {sample_time:.2f}s ({sample_time / 60:.2f}m)")


    torch.cuda.synchronize()
    total_time = perf_counter() - total_start

    print("\nFinished running attack.")
    print(f"Total runtime: {total_time:.2f}s ({total_time / 60:.2f}m, {total_time / 3600:.2f}h)")
