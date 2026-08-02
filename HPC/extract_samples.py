import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)

# TARGET_MODALITIES = ("MRI", "CT", "Ultrasound")
TARGET_MODALITIES = ("MRI",)
# SUBSET_TO_MODALITY = {
#     "mod-ct": "CT",
#     "mod-mri": "MRI",
#     "mod-us": "Ultrasound",
# }
SUBSET_TO_MODALITY = {
    "mod-mri": "MRI",
}

MODEL_PATH = "JZPeterPan/MedVLM-R1"

CLEAN_COLUMNS = [
    "sample_id", "category", "subset", "split", "question",
    "correct_answer", "num_choices", "wrong_targets",
    "clean_prediction", "clean_correct", "clean_raw_output", "error",
]

QUESTION_TEMPLATE = """
    {question}
    Your task:
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """

GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=1024,
    do_sample=False,
    temperature=1,
    num_return_sequences=1,
    pad_token_id=151643,
)


def log(message: str) -> None:
    """Print a timestamped message immediately, including in SLURM logs."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def progress_text(counts: Counter[str], target: int) -> str:
    """Return progress for all target modalities."""
    return " | ".join(
        f"{modality}: {counts[modality]}/{target}"
        for modality in TARGET_MODALITIES
    )


def is_correct(prediction: dict[str, str]) -> bool:
    """Return whether a saved clean prediction matches the ground truth."""
    if prediction.get("clean_correct", "").strip().lower() == "true":
        return True
    return (
        prediction.get("clean_prediction", "").strip().upper()
        == prediction.get("correct_answer", "").strip().upper()
    )


def read_predictions(path: Path) -> list[dict[str, str]]:
    """Read saved clean predictions; a missing file means no work has been done."""
    if not path.exists():
        log(f"No existing prediction file found at: {path}")
        return []

    log(f"Reading existing predictions from: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        predictions = list(csv.DictReader(file))

    correct_count = sum(is_correct(row) for row in predictions)
    log(
        f"Loaded {len(predictions)} previous predictions "
        f"({correct_count} correct)."
    )
    return predictions


def append_prediction(path: Path, prediction: dict[str, str]) -> None:
    """Append one resumable clean-inference result, creating the CSV if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CLEAN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(prediction)


def load_subset(data_root: Path, subset: str) -> Dataset:
    """Load the local test Parquet shards for one OmniMedVQA subset."""
    subset_root = data_root / subset
    parquet_files = sorted(subset_root.glob("test-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No test Parquet files found for {subset} under {subset_root}"
        )

    log(
        f"Loading {subset} from {len(parquet_files)} Parquet file(s) "
        f"under {subset_root}"
    )
    dataset = load_dataset(
        "parquet",
        data_files={"test": [str(path) for path in parquet_files]},
        split="test",
    )
    log(f"Loaded {subset}: {len(dataset):,} test samples")
    return dataset


def choices_for(record: dict[str, Any]) -> dict[str, str]:
    """Return the non-empty multiple-choice answers in one record."""
    return {
        letter: record[f"choice_{letter.lower()}"]
        for letter in "ABCD"
        if record.get(f"choice_{letter.lower()}", "").strip()
    }

def extract_answer(output_text, tag="answer"):
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", output_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

@torch.inference_mode()
def predict(
    model,
    processor,
    image,
    question: str,
    choices: dict[str, str],
) -> tuple[str | None, str]:
    image = image.convert("RGB")

    message = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(question=question),
            },
        ],
    }]

    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(message)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        generation_config=GENERATION_CONFIG,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    log(f"Raw generated output: {output!r}")

    answer_text = extract_answer(output, tag="answer")

    letter_match = (
        re.search(r"\b([A-D])\b", answer_text, re.IGNORECASE)
        if answer_text
        else None
    )

    letter = letter_match.group(1).upper() if letter_match else None

    return (letter if letter in choices else None), output


def select_samples(
    data_root: Path,
    predictions: list[dict[str, str]],
    samples_per_modality: int,
) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    """Return the first correct records for each requested modality."""
    log("Selecting correct samples from the saved prediction records...")

    datasets: dict[str, Dataset] = {}
    counts: Counter[str] = Counter()
    selected = []

    for prediction in predictions:
        if not is_correct(prediction):
            continue

        try:
            subset, index_text = prediction["sample_id"].split(":", maxsplit=1)
            index = int(index_text)
        except (KeyError, ValueError):
            continue

        if subset not in SUBSET_TO_MODALITY:
            continue

        modality = SUBSET_TO_MODALITY[subset]
        if counts[modality] >= samples_per_modality:
            continue

        if subset not in datasets:
            datasets[subset] = load_subset(data_root, subset)

        record = datasets[subset][index]
        if (
            record["problem"] != prediction["question"]
            or record["answer_letter"].strip().upper()
            != prediction["correct_answer"].strip().upper()
        ):
            raise ValueError(f"Cached row does not match {prediction['sample_id']}")

        selected.append((modality, prediction, record))
        counts[modality] += 1
        log(
            f"Selected cached sample {prediction['sample_id']} for {modality}. "
            f"Progress: {progress_text(counts, samples_per_modality)}"
        )

    log(f"Cached selection complete. {progress_text(counts, samples_per_modality)}")
    return selected


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device_name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    log(f"CUDA device: {device_name}")
    log(f"Model dtype: {dtype}")
    log(f"Loading model: {MODEL_PATH}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    model.requires_grad_(False)
    log("Model loaded successfully.")

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    log("Processor loaded successfully.")

    return model, processor


def create_missing_predictions(
    data_root: Path,
    results_path: Path,
    predictions: list[dict[str, str]],
    selected: list[tuple[str, dict[str, str], dict[str, Any]]],
    samples_per_modality: int,
) -> None:
    """Evaluate unseen records until every modality has enough correct samples."""
    counts = Counter(modality for modality, _, _ in selected)
    completed_ids = {row.get("sample_id", "") for row in predictions}

    log(f"Required progress: {progress_text(counts, samples_per_modality)}")

    if all(
        counts[modality] >= samples_per_modality
        for modality in TARGET_MODALITIES
    ):
        log("Enough correct cached samples already exist. Inference is not needed.")
        return

    log("Some modalities need more correct samples. Starting model inference.")
    model, processor = load_model()

    log("Loading target modality datasets...")
    datasets = {
        subset: load_subset(data_root, subset)
        for subset in SUBSET_TO_MODALITY
    }

    evaluated_count = 0

    for subset, dataset in datasets.items():
        modality = SUBSET_TO_MODALITY[subset]

        if counts[modality] >= samples_per_modality:
            log(f"Skipping {subset}: {modality} target is already complete.")
            continue

        log(
            f"Scanning {subset} ({modality}): {len(dataset):,} available samples. "
            f"Current progress: {counts[modality]}/{samples_per_modality}"
        )

        for index in range(len(dataset)):
            if all(
                counts[target] >= samples_per_modality
                for target in TARGET_MODALITIES
            ):
                log("All modality targets have been reached.")
                return

            if counts[modality] >= samples_per_modality:
                log(
                    f"Completed {modality}: "
                    f"{counts[modality]}/{samples_per_modality} correct samples."
                )
                break

            sample_id = f"{subset}:{index:06d}"
            if sample_id in completed_ids:
                continue

            record = dataset[index]
            choices = choices_for(record)

            log(
                f"Evaluating {sample_id} ({index + 1:,}/{len(dataset):,}) "
                f"for {modality}..."
            )

            predicted_letter, raw_output = predict(
                model,
                processor,
                record["image"],
                record["problem"],
                choices,
            )

            correct_answer = record["answer_letter"].strip().upper()
            prediction = {
                "sample_id": sample_id,
                "category": "modality",
                "subset": subset,
                "split": "test",
                "question": record["problem"],
                "correct_answer": correct_answer,
                "num_choices": str(len(choices)),
                "wrong_targets": "|".join(
                    letter for letter in choices if letter != correct_answer
                ),
                "clean_prediction": predicted_letter or "",
                "clean_correct": str(predicted_letter == correct_answer),
                "clean_raw_output": raw_output,
                "error": "",
            }

            append_prediction(results_path, prediction)
            predictions.append(prediction)
            completed_ids.add(sample_id)
            evaluated_count += 1

            if is_correct(prediction):
                counts[modality] += 1

            displayed_prediction = predicted_letter or "NONE"
            log(
                f"Result {sample_id}: predicted={displayed_prediction}, "
                f"correct={correct_answer}, "
                f"match={prediction['clean_correct']}"
            )
            log(
                f"Overall progress after {evaluated_count} new inference(s): "
                f"{progress_text(counts, samples_per_modality)}"
            )
            log(f"Prediction saved to: {results_path}")

    missing = {
        modality: samples_per_modality - counts[modality]
        for modality in TARGET_MODALITIES
        if counts[modality] < samples_per_modality
    }

    if missing:
        log(f"Warning: not enough correct samples were found: {missing}")
    else:
        log("Inference stage completed successfully.")


def write_samples(
    selected: list[tuple[str, dict[str, str], dict[str, Any]]],
    output_root: Path,
    selection_results: Path,
    overwrite: bool,
) -> None:
    """Write PNG images, question.json, and a CSV describing selected rows."""
    if not selected:
        raise RuntimeError("No correct samples were selected, so nothing can be saved.")

    images_root = output_root / "images"
    question_path = output_root / "question_mri.json"

    log(f"Preparing to save {len(selected)} selected sample(s) to: {output_root}")

    if overwrite:
        log(f"Overwrite enabled. Removing existing image directory: {images_root}")
        shutil.rmtree(images_root, ignore_errors=True)
    elif images_root.exists() or question_path.exists():
        raise FileExistsError(
            f"{output_root} already contains sample data; use --overwrite to replace it."
        )

    images_root.mkdir(parents=True, exist_ok=True)
    questions = []
    manifest = []

    for position, (modality, prediction, record) in enumerate(selected, start=1):
        image_name = prediction["sample_id"].replace(":", "_") + ".png"
        image_path = images_root / image_name
        image = record["image"]

        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        image.save(image_path, format="PNG")
        log(
            f"Saved image {position}/{len(selected)}: "
            f"{image_path} ({modality})"
        )

        questions.append({
            "id": prediction["sample_id"],
            "image": [f"images/{image_name}"],
            "problem": record["problem"],
            "solution": record["answer_letter"],
            "answer": record["answer_text"],
            "modality": modality,
            "question_type": record["question_type"],
        })
        manifest.append({
            "sample_id": prediction["sample_id"],
            "modality": modality,
            "question_type": record["question_type"],
            "correct_answer": prediction["correct_answer"],
            "clean_prediction": prediction["clean_prediction"],
            "image": f"images/{image_name}",
        })

    output_root.mkdir(parents=True, exist_ok=True)
    question_path.write_text(
        json.dumps(questions, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Saved question file: {question_path}")

    selection_results.parent.mkdir(parents=True, exist_ok=True)
    with selection_results.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    log(f"Saved selection manifest: {selection_results}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-root",
        type=Path,
        help="Directory containing the downloaded OmniMedVQA-V2 dataset.",
        default=Path("/mnt/parscratch/users/acp25tw/datasets/OmniMedVQA-V2"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("HPC/output/clean_results_mri.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/OmniMedVQA/sample_mri"),
    )
    parser.add_argument(
        "--selection-results",
        type=Path,
        default=Path("HPC/output/correct_samples_mri.csv"),
    )
    parser.add_argument("--samples-per-modality", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.samples_per_modality < 1:
        raise ValueError("--samples-per-modality must be at least 1")

    log("=" * 70)
    log("Starting OmniMedVQA correct-sample extraction")
    log(f"Model: {MODEL_PATH}")
    log(f"Data root: {args.data_root}")
    log(f"Predictions CSV: {args.results}")
    log(f"Output root: {args.output_root}")
    log(f"Selection CSV: {args.selection_results}")
    log(f"Samples per modality: {args.samples_per_modality}")
    log(f"Target total: {args.samples_per_modality * len(TARGET_MODALITIES)}")
    log(f"Overwrite: {args.overwrite}")
    log("=" * 70)

    log("Stage 1/4: Reading previous prediction results")
    predictions = read_predictions(args.results)

    log("Stage 2/4: Searching cached results for correct samples")
    selected = select_samples(
        args.data_root,
        predictions,
        args.samples_per_modality,
    )

    log("Stage 3/4: Generating any missing predictions")
    create_missing_predictions(
        args.data_root,
        args.results,
        predictions,
        selected,
        args.samples_per_modality,
    )

    log("Refreshing the selected samples after inference...")
    selected = select_samples(
        args.data_root,
        predictions,
        args.samples_per_modality,
    )

    log("Stage 4/4: Saving selected samples")
    write_samples(
        selected,
        args.output_root,
        args.selection_results,
        args.overwrite,
    )

    counts = Counter(modality for modality, _, _ in selected)
    log("=" * 70)
    log("Extraction completed")
    log(f"Final counts: {progress_text(counts, args.samples_per_modality)}")
    log(f"Total samples saved: {len(selected)}")
    log(f"Images: {args.output_root / 'images'}")
    log(f"Questions: {args.output_root / 'question_mri.json'}")
    log(f"Predictions: {args.results}")
    log(f"Selection manifest: {args.selection_results}")
    log("=" * 70)


if __name__ == "__main__":
    main()