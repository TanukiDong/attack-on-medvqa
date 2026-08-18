import argparse

from common.io import find_project_root
from common.model import load_model
from extract.extractor import extract_samples
from extract.loader import MODALITIES, load_data


MODEL_CONFIG = {
    "path": "JZPeterPan/MedVLM-R1",
    "max_new_tokens": 1024,
    "pad_token_id": 151643,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        choices=MODALITIES,
        required=True,
        help="Modality to extract samples from.",
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="Number of samples to extract.",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.",
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    return args


def main():
    args = parse_args()

    project_root = find_project_root()

    model, processor, generation_config = load_model(MODEL_CONFIG)

    dataset = load_data(args.modality)

    extract_samples(
        dataset=dataset,
        modality=args.modality,
        num_samples=args.samples,
        model=model,
        processor=processor,
        generation_config=generation_config,
        project_root=project_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()