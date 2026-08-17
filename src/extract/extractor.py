import csv
import json
import shutil

from common.model import extract_answer, run_model
from extract.loader import MODALITIES, load_processed_ids, load_questions


PROGRESS_FIELDS = [
    "sample_id",
    "modality",
    "question_type",
    "correct_answer",
    "clean_prediction",
    "clean_correct",
]


def append_progress(path, row):
    """Append one completed inference result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=PROGRESS_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_sample(
    record,
    sample_id,
    modality,
    questions,
    images_root,
    question_path,
):
    """Save correctly predicted sample."""
    modality_name, _ = MODALITIES[modality]

    image_name = sample_id.replace(":", "_") + ".png"
    image_path = images_root / image_name
    relative_image_path = f"images/{image_name}"

    image = record["image"].convert("RGB")
    image.save(image_path)

    questions.append(
        {
            "id": sample_id,
            "image": [relative_image_path],
            "problem": record["problem"],
            "solution": record["answer_letter"],
            "answer": record["answer_text"],
            "modality": modality_name,
            "question_type": record["question_type"],
        }
    )

    question_path.write_text(
        json.dumps(questions, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_samples(
    dataset,
    modality,
    num_samples,
    model,
    processor,
    generation_config,
    project_root,
    overwrite=False,
):
    """Extract correctly predicted samples."""
    modality_name, subset = MODALITIES[modality]

    output_dir = project_root / "data" / "OmniMedVQA" / f"sample_{modality}"
    images_root = output_dir / "images"
    question_path = output_dir / "question.json"

    progress_path = project_root / "result" / "MedVLM-R1" / "extract_samples" / f"progress_{modality}.csv"

    # Remove due to overwrite flag
    if overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
        progress_path.unlink(missing_ok=True)
        print(f"Overwrite: Removed {output_dir} and {progress_path}")

    images_root.mkdir(parents=True, exist_ok=True)

    questions = load_questions(question_path)
    processed_ids = load_processed_ids(progress_path)

    # Already enough samples
    if len(questions) >= num_samples:
        print(f"Already have {len(questions)}/{num_samples} correct samples.")
        return

    for index, record in enumerate(dataset):
        
        # Exit extract loop
        if len(questions) >= num_samples:
            break

        sample_id = f"{subset}:{index:06d}"

        # Skip processed
        if sample_id in processed_ids:
            continue

        correct_answer = record["answer_letter"].strip().upper()

        output = run_model(
            question=record["problem"],
            image=record["image"],
            model=model,
            processor=processor,
            generation_config=generation_config,
        )

        prediction = extract_answer(output)

        correct = (prediction == correct_answer)

        if correct:
            save_sample(
                record=record,
                sample_id=sample_id,
                modality=modality,
                questions=questions,
                images_root=images_root,
                question_path=question_path,
            )

        append_progress(
            progress_path,
            {
                "sample_id": sample_id,
                "modality": modality_name,
                "question_type": record["question_type"],
                "correct_answer": correct_answer,
                "clean_prediction": prediction or "",
                "clean_correct": correct,
            },
        )

        print(
            f"{sample_id} | "
            f"prediction={prediction} | "
            f"correct={correct} | "
            f"{len(questions)}/{num_samples}"
        )

    if len(questions) < num_samples:
        raise RuntimeError(f"Only found {len(questions)}/{num_samples} correct samples.")