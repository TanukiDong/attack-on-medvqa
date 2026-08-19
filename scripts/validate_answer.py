import argparse
from transformers import set_seed

from common.io import find_project_root, load_config
from common.model import load_model
from attacks.bias_field.validate import validate_answers

PROJECT_ROOT = find_project_root()
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
        "--modality",
        type=str,
        required=True,
        choices=["mri", "ct", "us"],
        help="Medical imaging modality to attack.",
    )
        
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing validation results",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    experiment_directory = RESULTS_PATH / args.modality / args.config
    if not experiment_directory.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_directory}")
    
    config_path = experiment_directory / "config.yaml"

    config = load_config(config_path)
    experiment_config = config["experiment"]
    model_config = config["model"]

    set_seed(experiment_config.get("seed", 42), deterministic=True)

    # load model
    model, processor, generation_config = load_model(model_config)

    # Validation
    print(f"Validating modality {args.modality} config {args.config}: {experiment_directory}")

    validate_answers(
        result_directory=experiment_directory,
        modality=args.modality,
        model=model,
        processor=processor,
        generation_config=generation_config,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()