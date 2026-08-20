import argparse
import sys
from pathlib import Path

from transformers import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs"
RESULTS_PATH = PROJECT_ROOT / "result" / "MedVLM-R1" / "bias_field_attack"

# Add src/ to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.io import get_batch_directories, load_config
from common.model import load_model
from attacks.bias_field.validate import validate_answers

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

    config_path = CONFIG_PATH / f"{args.config}.yaml"
    config = load_config(config_path)
    
    initialization = "random" if config["bias_field"]["random_start"] else "identity"
    experiment_directory = RESULTS_PATH / args.modality / args.config / initialization
    if not experiment_directory.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_directory}")
    
    batch_directories = get_batch_directories(experiment_directory)
    
    set_seed(config["experiment"].get("seed", 42), deterministic=True)

    # load model
    model, processor, generation_config = load_model(config["model"])

    # Loop over batches
    for result_directory in batch_directories:
        print(f"Validating modality {args.modality} config {args.config} : {result_directory}")
    
        validate_answers(
            result_directory=result_directory,
            modality=args.modality,
            model=model,
            processor=processor,
            generation_config=generation_config,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()