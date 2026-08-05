import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)

VERBOSE = False

MODALITY_CHOICES = {
    "mri": ("MRI", "mod-mri"),
    "ct": ("CT", "mod-ct"),
    "ultrasound": ("Ultrasound", "mod-us"),
}

SUBSET_TO_MODALITY = {
    "mod-ct": "CT",
    "mod-mri": "MRI",
    "mod-us": "Ultrasound",
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


def progress_text(
    counts: Counter[str],
    target: int,
    target_modality: str,
) -> str:
    """Return progress for the selected modality."""
    return f"{target_modality}: {counts[target_modality]}/{target}"


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

    with path.open(encoding="utf-8", newline="") as file:
        predictions = list(csv.DictReader(file))

    correct_count = sum(is_correct(row) for row in predictions)
    if VERBOSE:
        log(f"Loaded {len(predictions)} previous predictions ({correct_count} correct).")
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

    dataset = load_dataset(
        "parquet",
        data_files={"test": [str(path) for path in parquet_files]},
        split="test",
    )
    if VERBOSE:
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
    device,
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
    ).to(device)

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
    target_modality: str,
    target_subset: str,
) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    """Return the first correct records for the selected modality."""
    if VERBOSE:
        log("Selecting correct samples from the saved prediction records...")

    dataset: Dataset | None = None
    selected = []

    for prediction in predictions:
        if not is_correct(prediction):
            continue

        try:
            subset, index_text = prediction["sample_id"].split(":", maxsplit=1)
            index = int(index_text)
        except (KeyError, ValueError):
            continue

        if subset != target_subset:
            continue

        if len(selected) >= samples_per_modality:
            break

        if dataset is None:
            dataset = load_subset(data_root, target_subset)

        record = dataset[index]
        if (
            record["problem"] != prediction["question"]
            or record["answer_letter"].strip().upper()
            != prediction["correct_answer"].strip().upper()
        ):
            raise ValueError(f"Cached row does not match {prediction['sample_id']}")

        selected.append((target_modality, prediction, record))

    return selected


def load_model(use_cpu: bool = False):
    if use_cpu:
        device = "cpu"
        dtype = torch.float32
        if VERBOSE:
            log("Running on CPU.")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use --cpu to run on CPU."
            )

        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        if VERBOSE:
            log(f"CUDA device: {device_name}")

    if VERBOSE:
        log(f"Device: {device}")
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
    if VERBOSE:
        log("Model loaded successfully.")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    if VERBOSE:
        log("Processor loaded successfully.")

    return model, processor, device


def create_missing_predictions(
    data_root: Path,
    results_path: Path,
    predictions: list[dict[str, str]],
    selected: list[tuple[str, dict[str, str], dict[str, Any]]],
    samples_per_modality: int,
    target_modality: str,
    target_subset: str,
    use_cpu: bool = False,
) -> None:
    """Evaluate unseen records until the selected modality has enough samples."""
    correct_count = len(selected)
    completed_ids = {row.get("sample_id", "") for row in predictions}

    if VERBOSE:
        log(f"Required progress: {target_modality}: {correct_count}/{samples_per_modality}")

    if correct_count >= samples_per_modality:
        log("Enough correct cached samples already exist. Inference is not needed.")
        return

    model, processor, device = load_model(use_cpu=use_cpu)

    dataset = load_subset(data_root, target_subset)

    evaluated_count = 0
    if VERBOSE:
        log(f"{len(dataset):,} available samples. Current progress: {correct_count}/{samples_per_modality}")

    for index in range(len(dataset)):
        if correct_count >= samples_per_modality:
            log(f"Completed {target_modality}: {correct_count}/{samples_per_modality} correct samples.")
            return

        sample_id = f"{target_subset}:{index:06d}"
        if sample_id in completed_ids:
            continue

        record = dataset[index]
        choices = choices_for(record)

        if VERBOSE:
            log(
                f"Evaluating {sample_id} ({index + 1:,}/{len(dataset):,}) "
                f"for {target_modality}..."
            )

        predicted_letter, raw_output = predict(
            model,
            processor,
            device,
            record["image"],
            record["problem"],
            choices,
        )

        correct_answer = record["answer_letter"].strip().upper()
        prediction = {
            "sample_id": sample_id,
            "category": "modality",
            "subset": target_subset,
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
            correct_count += 1

        displayed_prediction = predicted_letter or "NONE"
        if VERBOSE:
            log(
                f"Result {sample_id}: predicted={displayed_prediction}, "
                f"correct={correct_answer}, "
                f"match={prediction['clean_correct']}"
            )
            log(f"Overall progress {target_modality}: {correct_count}/{samples_per_modality}")

    missing = samples_per_modality - correct_count
    if missing > 0:
        log(f"Warning: not enough correct {target_modality} samples were found. Missing: {missing}")
    else:
        log("Inference stage completed successfully.")


def count_processed_samples(question_path: Path) -> int:
    """Return the number of samples already written to question.json."""
    if not question_path.exists():
        return 0

    try:
        questions = json.loads(question_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read existing sample file: {question_path}"
        ) from error

    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON list in: {question_path}")

    return len(questions)


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
    question_path = output_root / "question.json"

    if overwrite:
        log(f"Overwrite enabled. Removing existing image directory: {images_root}")
        shutil.rmtree(images_root, ignore_errors=True)

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

    selection_results.parent.mkdir(parents=True, exist_ok=True)
    with selection_results.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

def main() -> None:
    total_start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--modality",
        choices=MODALITY_CHOICES,
        default="mri",
        help="Modality to extract: mri, ct, or ultrasound.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Directory containing the downloaded OmniMedVQA-V2 dataset.",
        default=Path("/mnt/parscratch/users/acp25tw/datasets/OmniMedVQA-V2"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Prediction CSV path. Defaults to a modality-specific path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to a modality-specific path.",
    )
    parser.add_argument(
        "--selection-results",
        type=Path,
        default=None,
        help="Selection CSV path. Defaults to a modality-specific path.",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU instead of CUDA.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and model output.",
    )

    args = parser.parse_args()
    
    global VERBOSE
    VERBOSE = args.verbose

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    target_modality, target_subset = MODALITY_CHOICES[args.modality]

    if args.results is None:
        args.results = Path(
            "result/MedVLM-R1/extract_samples/"
            f"clean_results_{args.modality}.csv"
        )

    if args.output_root is None:
        args.output_root = Path(
            f"data/OmniMedVQA/sample_{args.modality}"
        )

    if args.selection_results is None:
        args.selection_results = Path(
            "result/MedVLM-R1/extract_samples/"
            f"correct_samples_{args.modality}.csv"
        )

    processed_count = count_processed_samples(
        args.output_root / "question.json"
    )

    if args.samples < processed_count:
        parser.error(
            f"--samples ({args.samples}) cannot be "
            f"less than the {processed_count} samples already processed in "
            f"{args.output_root}."
        )

    log("=" * 70)
    log("Starting OmniMedVQA correct-sample extraction")
    log(f"Model: {MODEL_PATH}")
    log(f"Modality: {target_modality}")
    log(f"Subset: {target_subset}")
    log(f"Data root: {args.data_root}")
    log(f"Predictions CSV: {args.results}")
    log(f"Output root: {args.output_root}")
    log(f"Selection CSV: {args.selection_results}")
    log(f"Already processed: {processed_count}")
    log(f"Samples per modality: {args.samples}")
    log(f"Target total: {args.samples}")
    log(f"Overwrite: {args.overwrite}")
    log("=" * 70)

    log("Stage 1/4: Reading previous prediction results")
    predictions = read_predictions(args.results)

    log("Stage 2/4: Searching cached results for correct samples")
    selected = select_samples(
        args.data_root,
        predictions,
        args.samples,
        target_modality,
        target_subset,
    )

    log("Stage 3/4: Generating any missing predictions")
    create_missing_predictions(
        args.data_root,
        args.results,
        predictions,
        selected,
        args.samples,
        target_modality,
        target_subset,
        use_cpu=args.cpu,
    )

    selected = select_samples(
        args.data_root,
        predictions,
        args.samples,
        target_modality,
        target_subset,
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
    log(f"Final count: {progress_text(counts, args.samples, target_modality)}")
    log(f"Total samples saved: {len(selected)}")
    log(f"Images: {args.output_root / 'images'}")
    log(f"Questions: {args.output_root / 'question.json'}")
    log(f"Predictions: {args.results}")
    log(f"Selection manifest: {args.selection_results}")
    log(f"Total runtime: {perf_counter() - total_start:.2f} seconds")
    log("=" * 70)


if __name__ == "__main__":
    main()