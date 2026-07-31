"""Create 10 correct MRI, CT, and Ultrasound OmniMedVQA samples.

The script resumes clean MedVLM-R1 inference in result/MedVLM-R1/clean_results.csv
until it has ten correct examples of each modality. It then writes the images and
question metadata to data/OmniMedVQA/sample.
"""

import argparse
import csv
import json
import re
import shutil
import tempfile
from importlib.util import find_spec
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration


MODEL_PATH = "JZPeterPan/MedVLM-R1"
TARGET_MODALITIES = ("MRI", "CT", "Ultrasound")
CLEAN_COLUMNS = [
    "sample_id", "category", "subset", "split", "question",
    "correct_answer", "num_choices", "wrong_targets",
    "clean_prediction", "clean_correct", "clean_raw_output", "error",
]

# Verified in src/MedVLM-R1/ta.ipynb. The local cache directory names are hashes.
CACHE_TO_SUBSET = {
    "default-a272ac07a5ea5697": "mod-ct",
    "default-5e6fee2c6158fda4": "mod-mri",
    "default-82c4edbb04c261dd": "mod-us",
}
SUBSET_TO_CACHE = {subset: cache for cache, subset in CACHE_TO_SUBSET.items()}
QUESTION_TEMPLATE = """{question}
Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice inside <answer>...</answer> tags.
3. No extra information or text outside of these tags.
"""


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
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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
    """Load the local Arrow shards for one OmniMedVQA modality subset."""
    cache_root = data_root / SUBSET_TO_CACHE[subset]
    arrow_files = sorted(cache_root.rglob("parquet-test*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No cached Arrow shards found for {subset} under {cache_root}")
    shards = [Dataset.from_file(str(path)) for path in arrow_files]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def normalise_modality(raw_modality: str) -> str | None:
    """Map OmniMedVQA modality labels to the three requested names."""
    modality = raw_modality.casefold().replace("-", "")
    if "magnetic resonance" in modality or modality.strip() == "mri":
        return "MRI"
    if "computed tomography" in modality or modality.strip() == "ct":
        return "CT"
    if "ultrasound" in modality:
        return "Ultrasound"
    return None


def choices_for(record: dict[str, Any]) -> dict[str, str]:
    """Return the non-empty multiple-choice answers in one record."""
    return {
        letter: record[f"choice_{letter.lower()}"]
        for letter in "ABCD"
        if record.get(f"choice_{letter.lower()}", "").strip()
    }


def predict(
    model,
    processor,
    generation_config: GenerationConfig,
    image,
    question: str,
    choices: dict[str, str],
) -> tuple[str | None, str]:
    """Run one clean MedVLM-R1 prediction using the demo notebook workflow."""
    with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
        image.convert("RGB").save(image_file.name, format="PNG")
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_file.name}"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(question=question)},
            ],
        }]
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=1024,
                do_sample=False,
                generation_config=generation_config,
            )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", output, re.DOTALL)
    letter = answer.group(1).strip().upper() if answer else None
    return (letter if letter in choices else None), output


def select_samples(
    data_root: Path,
    predictions: list[dict[str, str]],
    samples_per_modality: int,
) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    """Return the first correct records for each requested modality."""
    datasets: dict[str, Dataset] = {}
    counts: Counter[str] = Counter()
    selected = []

    for prediction in predictions:
        if not is_correct(prediction):
            continue
        try:
            subset, index_text = prediction["sample_id"].split(":", maxsplit=1)
            index = int(index_text)
        except ValueError:
            continue
        if subset not in SUBSET_TO_CACHE:
            continue
        if subset not in datasets:
            datasets[subset] = load_subset(data_root, subset)
        record = datasets[subset][index]
        modality = normalise_modality(record["modality"])
        if modality not in TARGET_MODALITIES or counts[modality] >= samples_per_modality:
            continue
        if (
            record["problem"] != prediction["question"]
            or record["answer_letter"].strip().upper()
            != prediction["correct_answer"].strip().upper()
        ):
            raise ValueError(f"Cached row does not match {prediction['sample_id']}")
        selected.append((modality, prediction, record))
        counts[modality] += 1
    return selected


def load_model() -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig]:
    """Load MedVLM-R1 with the same settings as demo.ipynb."""
    cuda_available = torch.cuda.is_available()
    attention_implementation = (
        "flash_attention_2" if find_spec("flash_attn") is not None else "sdpa"
    )
    dtype = torch.bfloat16 if cuda_available else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        attn_implementation=attention_implementation,
        device_map="auto",
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    generation_config = GenerationConfig(
        max_new_tokens=1024,
        do_sample=False,
        temperature=1,
        num_return_sequences=1,
        pad_token_id=151643,
    )
    return model, processor, generation_config


def create_missing_predictions(
    data_root: Path,
    results_path: Path,
    predictions: list[dict[str, str]],
    selected: list[tuple[str, dict[str, str], dict[str, Any]]],
    samples_per_modality: int,
) -> None:
    """Evaluate unseen target-modality records until every target has enough correct rows."""
    counts = Counter(modality for modality, _, _ in selected)
    completed_ids = {row.get("sample_id", "") for row in predictions}
    if all(counts[modality] >= samples_per_modality for modality in TARGET_MODALITIES):
        return

    model, processor, generation_config = load_model()
    datasets = {subset: load_subset(data_root, subset) for subset in SUBSET_TO_CACHE}

    for subset, dataset in datasets.items():
        for index in range(len(dataset)):
            if all(counts[modality] >= samples_per_modality for modality in TARGET_MODALITIES):
                return

            sample_id = f"{subset}:{index:06d}"
            if sample_id in completed_ids:
                continue
            record = dataset[index]
            modality = normalise_modality(record["modality"])
            if modality not in TARGET_MODALITIES or counts[modality] >= samples_per_modality:
                continue

            choices = choices_for(record)
            predicted_letter, raw_output = predict(
                model, processor, generation_config, record["image"], record["problem"], choices
            )
            prediction = {
                "sample_id": sample_id,
                "category": "modality",
                "subset": subset,
                "split": "test",
                "question": record["problem"],
                "correct_answer": record["answer_letter"],
                "num_choices": str(len(choices)),
                "wrong_targets": "|".join(letter for letter in choices if letter != record["answer_letter"]),
                "clean_prediction": predicted_letter or "",
                "clean_correct": str(predicted_letter == record["answer_letter"]),
                "clean_raw_output": raw_output,
                "error": "",
            }
            append_prediction(results_path, prediction)
            predictions.append(prediction)
            completed_ids.add(sample_id)
            if is_correct(prediction):
                counts[modality] += 1
            print(f"{sample_id}: {modality}, correct={prediction['clean_correct']}")

    missing = ", ".join(
        modality for modality in TARGET_MODALITIES if counts[modality] < samples_per_modality
    )
    raise RuntimeError(f"Ran out of records before finding enough correct samples for: {missing}")


def write_samples(
    selected: list[tuple[str, dict[str, str], dict[str, Any]]],
    output_root: Path,
    selection_results: Path,
    overwrite: bool,
) -> None:
    """Write PNG images, question.json, and a CSV describing the selected rows."""
    images_root = output_root / "images"
    question_path = output_root / "question.json"
    if overwrite:
        shutil.rmtree(images_root, ignore_errors=True)
    elif images_root.exists() or question_path.exists():
        raise FileExistsError(
            f"{output_root} already contains sample data; use --overwrite to replace it."
        )
    images_root.mkdir(parents=True, exist_ok=True)
    questions = []
    manifest = []

    for modality, prediction, record in selected:
        image_name = prediction["sample_id"].replace(":", "_") + ".png"
        image = record["image"]
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        image.save(images_root / image_name, format="PNG")
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
        json.dumps(questions, indent=2) + "\n", encoding="utf-8"
    )
    selection_results.parent.mkdir(parents=True, exist_ok=True)
    with selection_results.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=project_root / "data/OmniMedVQA/parquet",
        help="Directory containing the local OmniMedVQA Arrow cache.",
    )
    parser.add_argument(
        "--results", type=Path,
        default=project_root / "result/MedVLM-R1/clean_results.csv",
        help="Resumable clean MedVLM-R1 prediction CSV.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=project_root / "data/OmniMedVQA/sample",
        help="Destination for images/ and question.json.",
    )
    parser.add_argument(
        "--selection-results", type=Path,
        default=project_root / "result/MedVLM-R1/correct_modality_samples.csv",
        help="CSV manifest for the selected correct samples.",
    )
    parser.add_argument("--samples-per-modality", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing sample images and question.json (default: true).",
    )
    args = parser.parse_args()
    if args.samples_per_modality < 1:
        raise ValueError("--samples-per-modality must be at least 1")

    predictions = read_predictions(args.results)
    selected = select_samples(args.data_root, predictions, args.samples_per_modality)
    create_missing_predictions(
        args.data_root, args.results, predictions, selected, args.samples_per_modality
    )
    selected = select_samples(args.data_root, predictions, args.samples_per_modality)
    if len(selected) != args.samples_per_modality * len(TARGET_MODALITIES):
        raise RuntimeError("Could not collect the requested number of correct samples.")
    write_samples(selected, args.output_root, args.selection_results, args.overwrite)

    counts = Counter(modality for modality, _, _ in selected)
    print(", ".join(f"{modality}: {counts[modality]}" for modality in TARGET_MODALITIES))
    print(f"Predictions: {args.results}")
    print(f"Questions: {args.output_root / 'question.json'}")


if __name__ == "__main__":
    main()
