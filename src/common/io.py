import csv
import json
import shutil
import yaml
from pathlib import Path


HISTORY_FIELDS = [
    "question_id",
    "step",
    "loss",
    "prob_A",
    "prob_B",
    "prob_C",
    "prob_D",
    "predicted_answer",
    "attack_success",
    "evaluated",
]

def find_project_root():
    """Find the root directory of the project by looking for a .git file."""
    for path in Path(__file__).resolve().parents:
        if (path / ".git").exists():
            return path

    raise FileNotFoundError("Could not find project root.")

def relative_to_project(path, project_root):
    """Return the relative path with respect to project_root."""
    return str(Path(path).resolve().relative_to(Path(project_root).resolve()))

def resolve_project_path(path, project_root):
    """Resolve a path relative to the project root."""
    path = Path(path)
    
    if path.is_absolute():
        return path

    return Path(project_root) / path

def get_batch_directories(experiment_directory):
    """Get all batch directories containing attack_results.jsonl."""
    return [
        path.parent
        for path in experiment_directory.glob("batch_*/attack_results.jsonl")
    ]

def load_config(config_path):
    """Load a YAML configuration file."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration is not correct.")

    return config

def load_completed_ids(result_path, overwrite=False):
    """Load completed question IDs from a JSONL file."""
    result_path = Path(result_path)
    if overwrite or not result_path.exists():
        return set()

    with result_path.open(encoding="utf-8") as file:
        return {
            json.loads(line)["question_id"]
            for line in file
            if line.strip()
        }

def load_samples(question_path, result_path, modality=None, start_index=0, end_index=None, overwrite=False, verbose=1):
    """Load unprocessed samples."""

    with question_path.open(encoding="utf-8") as file:
        all_samples = json.load(file)

    if modality is not None:
        samples = [sample for sample in all_samples if sample.get("modality") == modality]
    else:
        samples = all_samples

    selected_samples = samples[start_index:end_index]

    completed_question_ids = load_completed_ids(result_path, overwrite=overwrite)

    remaining_samples = [sample for sample in selected_samples if str(sample["id"]) not in completed_question_ids]

    if verbose:
        print(
            f"Loaded {len(samples)} {modality or 'total'} samples, "
            f"selected indices [{start_index}:{end_index}], "
            f"{len(remaining_samples)} samples remaining to be processed."
        )

    return remaining_samples


def initialize_output(output_directory, config_path, overwrite=False):
    """Initialize output directories and files for the attack."""
    output_directory = Path(output_directory)

    if overwrite:
        shutil.rmtree(output_directory, ignore_errors=True)
        print(f"Overwrite: Removed {output_directory}")

    attacked_image_directory = output_directory / "attacked_images"
    bias_field_directory = output_directory / "bias_fields"
    result_path = output_directory / "attack_results.jsonl"
    history_path = output_directory / "attack_history.csv"

    for directory in (output_directory, attacked_image_directory, bias_field_directory):
        directory.mkdir(parents=True, exist_ok=True)

    if overwrite or not result_path.exists():
        result_path.touch()

    if overwrite or not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=HISTORY_FIELDS).writeheader()

    shutil.copy2(config_path, output_directory / "config.yaml")

    return {
        "output_directory": output_directory,
        "result_path": result_path,
        "history_path": history_path,
        "attacked_image_directory": attacked_image_directory,
        "bias_field_directory": bias_field_directory,
    }


def get_output_paths(question_id, attacked_image_directory, bias_field_directory):
    """Format file name and return paths for attacked image and bias field."""
    safe_id = str(question_id).replace(":", "_")
    attacked_image_path = Path(attacked_image_directory) / f"{safe_id}_biased.png"
    bias_field_path = Path(bias_field_directory) / f"{safe_id}_bias_field.pt"
    return attacked_image_path, bias_field_path


def append_jsonl(path, record):
    """Append a record to a JSONL file."""
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def append_attack_history(path, question_id, history):
    """Append attack history to a CSV file."""
    with Path(path).open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)

        for entry in history:
            probabilities = entry.get("answer_probabilities", {})

            writer.writerow(
                {
                    "question_id": question_id,
                    "step": entry["step"],
                    "loss": entry["loss"],
                    "prob_A": probabilities.get("A"),
                    "prob_B": probabilities.get("B"),
                    "prob_C": probabilities.get("C"),
                    "prob_D": probabilities.get("D"),
                    "predicted_answer": entry["predicted_answer"],
                    "attack_success": entry["attack_success"],
                    "evaluated": entry["evaluated"],
                }
            )
