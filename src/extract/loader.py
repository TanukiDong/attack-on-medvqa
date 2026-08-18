import csv
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from common.io import find_project_root

PROJECT_ROOT = find_project_root()
HPC_DATA_ROOT = Path("/mnt/parscratch/users/acp25tw/datasets/OmniMedVQA-V2")
LOCAL_DATA_ROOT = PROJECT_ROOT / "data" / "OmniMedVQA" / "parquet"

MODALITIES = {
    "mri": ("MRI", "mod-mri"),
    "ct": ("CT", "mod-ct"),
    "us": ("Ultrasound", "mod-us"),
}

SUBSET_TO_CACHE = {
    "mod-ct": "default-a272ac07a5ea5697",
    "mod-mri": "default-5e6fee2c6158fda4",
    "mod-us": "default-82c4edbb04c261dd",
}

def load_data(modality):
    """Load the OmniMedVQA test parquet."""
    
    # HPC
    _, subset = MODALITIES[modality]
    parquet_path = HPC_DATA_ROOT / subset / "test-00000-of-00001.parquet"
    
    if parquet_path.exists():
        print(f"Loading parquet dataset from {parquet_path}")
    
        return load_dataset(
            "parquet",
            data_files={"test": str(parquet_path)},
            split="test",
        )
    
    # Local
    cache_dir = LOCAL_DATA_ROOT / SUBSET_TO_CACHE[subset]
    # Search for the arrow files
    arrow_files = sorted(cache_dir.rglob("parquet-test*.arrow"))
    
    if arrow_files:
        print(f"Loading arrow cache from {cache_dir}")
        
        datasets = [Dataset.from_file(str(path)) for path in arrow_files]
        if len(datasets) == 1:
            return datasets[0]
        else:
            return concatenate_datasets(datasets)
        
    raise FileNotFoundError(f"Could not find dataset for modality '{modality}' in HPC or local cache.")
        
    

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