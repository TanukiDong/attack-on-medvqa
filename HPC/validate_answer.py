import argparse
import sys
from pathlib import Path

from transformers import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTION_PATH = PROJECT_ROOT / "data" / "OmniMedVQA" / "sample_mri" / "question.json"

# Add src/ to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.io import load_config, resolve_project_path
from common.model import load_model
from common.validate import validate_answers

def parse_args():
    parser = argparse.ArgumentParser(description="Validate bias-field attacked images.")

    parser.add_argument(
        "--result-directory",
        type=str,
        required=True,
        help="Path to the attack result directory",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing validation results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result_directory = resolve_project_path(args.result_directory, PROJECT_ROOT)
    config_path = result_directory / "config.yaml"

    config = load_config(config_path)
    experiment_config = config["experiment"]
    model_config = config["model"]

    set_seed(experiment_config.get("seed", 42), deterministic=True)

    # load model
    model, processor, generation_config = load_model(model_config)

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