import argparse
from pathlib import Path

from attacks.bias_field.runner import run_bias_field_attack
from common.io import find_project_root

PROJECT_ROOT = find_project_root()

def parse_args():
    parser = argparse.ArgumentParser(description="Bias Field Attack.")

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
        "--start-index",
        type=int,
        default=0,
        help="Start index of the selected samples.",
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=10,
        help="End index of the selected samples.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override output directory. Default is result/MedVLM-R1/bias_field_attack/<modality>/<experiment_name>.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.",
    )
    
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start-index must be at least 0.")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("--end-index must be greater than or equal to --start-index.")

    return args


def main():
    args = parse_args()
    
    experiment_name = args.config
    config_path = PROJECT_ROOT / "configs" / f"{experiment_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    run_bias_field_attack(
        config_path=config_path,
        modality=args.modality,
        start_index=args.start_index,
        end_index=args.end_index,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()