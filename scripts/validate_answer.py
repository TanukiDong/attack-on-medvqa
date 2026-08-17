import argparse
from pathlib import Path

from transformers import set_seed

from common.io import find_project_root, get_batch_directories, load_config
from common.model import load_model
from common.validate import validate_answers

PROJECT_ROOT = find_project_root()
QUESTION_PATH = PROJECT_ROOT / "data" / "OmniMedVQA" / "sample_mri" / "question.json"
RESULTS_PATH = PROJECT_ROOT / "result" / "MedVLM-R1" / "bias_field_attack"

def parse_args():
    parser = argparse.ArgumentParser(description="Validate bias-field attacked images.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment config name, e.g. 'cps_8_eps_0p3'",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing validation results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    experiment_directory = RESULTS_PATH / args.config
    if not experiment_directory.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_directory}")
    
    batch_directories = get_batch_directories(experiment_directory)
    config_path = batch_directories[0] / "config.yaml"

    config = load_config(config_path)
    experiment_config = config["experiment"]
    model_config = config["model"]

    set_seed(experiment_config.get("seed", 42), deterministic=True)

    # load model
    model, processor, generation_config = load_model(model_config)

    # Loop over batches
    for i, result_directory in enumerate(batch_directories, start=1):
        print(f"Validating config {args.config} batch {i}/{len(batch_directories)}: {result_directory}")
    
        validate_answers(
            question_path=QUESTION_PATH,
            result_directory=result_directory,
            model=model,
            processor=processor,
            generation_config=generation_config,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()