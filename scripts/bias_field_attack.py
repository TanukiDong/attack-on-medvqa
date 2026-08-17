import argparse
from pathlib import Path

from attacks.bias_field.runner import run_bias_field_attack

def parse_args():
    parser = argparse.ArgumentParser(description="Bias Field Attack.")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration.",
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
        help="Override output directory. Default is result/MedVLM-R1/bias_field_attack/<experiment_name>.",
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
    
    run_bias_field_attack(
        config_path=args.config,
        start_index=args.start_index,
        end_index=args.end_index,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()