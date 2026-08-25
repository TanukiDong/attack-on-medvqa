import argparse
import json
import re
from pathlib import Path

from common.io import find_project_root


VALID_MODALITIES = ("mri", "ct", "us")
NEGATED_ANSWER = "No correct answer"


def negate_correct_option(problem, solution):
    """Replace the correct answer option text with 'No correct answer'."""
    solution = solution.strip().upper()

    if solution not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unsupported solution: {solution}")

    # Match:
    # B)MRI
    # B) MRI
    # B.MRI
    # B. MRI
    #
    # Stop when the next answer choice begins or at end of string.
    pattern = re.compile(
        rf"(\b{solution}\s*[\)\.]\s*)"
        rf"(.*?)"
        rf"(?=(?:,\s*|\s+)[A-D]\s*[\)\.]|$)",
        flags=re.IGNORECASE,
    )

    negated_problem, count = pattern.subn(
        rf"\1{NEGATED_ANSWER}",
        problem,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"Could not find option {solution} in problem:\n{problem}"
        )

    return negated_problem


def create_answer_negation_questions(modality):
    """Create question_an.json for answer-negation testing."""
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Unsupported modality: {modality}. "
            f"Choose from {VALID_MODALITIES}."
        )

    project_root = find_project_root()

    sample_directory = (
        project_root
        / "data"
        / "OmniMedVQA"
        / f"sample_{modality}"
    )

    input_path = sample_directory / "question.json"
    output_path = sample_directory / "question_an.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Question file not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    negated_samples = []

    for sample in samples:
        modified_sample = sample.copy()

        modified_sample["problem"] = negate_correct_option(
            problem=sample["problem"],
            solution=sample["solution"],
        )

        # The correct letter stays the same, because that option now means
        # "No correct answer".
        modified_sample["solution"] = sample["solution"]

        # Update the text representation of the answer.
        modified_sample["answer"] = NEGATED_ANSWER

        negated_samples.append(modified_sample)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            negated_samples,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Created {len(negated_samples)} answer-negated questions.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create answer-negated OmniMedVQA questions."
    )

    parser.add_argument(
        "--modality",
        required=True,
        choices=VALID_MODALITIES,
        help="Modality to process: mri, ct, or us.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    create_answer_negation_questions(
        modality=args.modality,
    )


if __name__ == "__main__":
    main()