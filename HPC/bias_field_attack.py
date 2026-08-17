import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ["HF_HOME"] = "/mnt/parscratch/users/acp25tw/huggingface_cache"


from attacks.bias_field.runner import run_bias_field_attack


def parse_args():
    parser = argparse.ArgumentParser(description="Bias Field Attack on HPC.")

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
        "--batch-size",
        type=int,
        default=50,
        help="Number of samples processed by each Slurm array task.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override the experiment output directory.",
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
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0.")

    return args


def main():
    args = parse_args()

    batch_id_string = os.environ.get("SLURM_ARRAY_TASK_ID")

    if batch_id_string is not None:
        # Running as a Slurm array task
        batch_id = int(batch_id_string)

        start_index = batch_id * args.batch_size
        end_index = start_index + args.batch_size

    else:
        # Running directly
        batch_id = None

        start_index = args.start_index
        end_index = args.end_index

    experiment_name = args.config.stem

    if args.output_path is not None:
        output_path = args.output_path

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path 

    else:
        output_path = PROJECT_ROOT / "result" / "MedVLM-R1" / "bias_field_attack"

    if batch_id is not None:
        output_path = output_path / experiment_name / f"batch_{batch_id}"
    else:
        output_path = output_path / experiment_name

    run_bias_field_attack(
        config_path=args.config,
        start_index=start_index,
        end_index=end_index,
        output_path=output_path,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
