import csv
import json
from pathlib import Path

from datasets import load_dataset


DATA_ROOT = Path("/mnt/parscratch/users/acp25tw/datasets/OmniMedVQA-V2")

MODALITIES = {
    "mri": ("MRI", "mod-mri"),
    "ct": ("CT", "mod-ct"),
    "us": ("Ultrasound", "mod-us"),
}

def load_data(modality):
    """Load the OmniMedVQA test parquet."""
    _, subset = MODALITIES[modality]
    parquet_path = DATA_ROOT / subset / "test-00000-of-00001.parquet"

    return load_dataset(
        "parquet",
        data_files={"test": str(parquet_path)},
        split="test",
    )

def load_questions(path):
    """Load previously selected correct samples."""
    if not path.exists():
        return []

    return json.loads(path.read_text(encoding="utf-8"))


def load_processed_ids(path):
    """Load IDs that have already been evaluated."""
    if not path.exists():
        return set()

    with path.open(encoding="utf-8", newline="") as file:
        return {
            row["sample_id"]
            for row in csv.DictReader(file)
        }