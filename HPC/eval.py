import argparse
import json
import random
import re

from importlib.util import find_spec
from pathlib import Path
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)


# ---------------------------- Settings ----------------------------

SEED = 42
MODEL_PATH = "JZPeterPan/MedVLM-R1"
VALID_ANSWERS = ("A", "B", "C", "D")


def find_project_root():
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find project root starting from '{current}'"
    )


PROJECT_ROOT = find_project_root()
QUESTION_PATH = PROJECT_ROOT / "data" / "OmniMedVQA" / "sample_mri" / "question.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Configuration folder name, e.g. cps_8_eps_0p3",
    )
    return parser.parse_args()


ARGS = parse_args()

CONFIG_DIRECTORY = PROJECT_ROOT / "result" / "MedVLM-R1" / "bias_field_attack" / ARGS.config
NUM_BATCHES = 20


QUESTION_TEMPLATE = """
    {Question}
    Your task:
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_answer(output_text, tag="answer"):
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def build_message(question, image_path):
    image_uri = Path(image_path).resolve().as_uri()

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question),
                },
            ],
        }
    ]


@torch.inference_mode()
def run_model(question, image_path):
    message = build_message(question, image_path)

    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(message)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        generation_config=generation_config,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# ---------------------------- Setup ----------------------------

set_seed()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Submit this script to a GPU node.")

device = "cuda"
dtype = torch.bfloat16

flash_attn_available = find_spec("flash_attn") is not None
attn_implementation = (
    "flash_attention_2" if flash_attn_available else "sdpa"
)

print(f"Using attention implementation: {attn_implementation}")
print(f"Using device: {device}")
print(f"Using dtype: {dtype}")
print(f"Evaluating configuration: {CONFIG_DIRECTORY}")
print()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=dtype,
    attn_implementation=attn_implementation,
    device_map="auto",
)

model.eval()
model.requires_grad_(False)
model.config.use_cache = False

processor = AutoProcessor.from_pretrained(MODEL_PATH)

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=False,
    num_return_sequences=1,
    pad_token_id=151643,
)

print("Model and processor loaded successfully.")
print()


# ---------------------------- Evaluation ----------------------------

with QUESTION_PATH.open(encoding="utf-8") as file:
    all_samples = json.load(file)

samples_by_id = {
    str(sample["id"]): sample
    for sample in all_samples
}

total_evaluated = 0
total_successful = 0
total_failed = 0

for batch_index in range(NUM_BATCHES):
    batch_directory = CONFIG_DIRECTORY / f"batch_{batch_index}"
    result_path = batch_directory / "attack_results.jsonl"
    validated_result_path = batch_directory / "validated_attack_results.jsonl"

    if not result_path.exists():
        print(f"Skipping batch_{batch_index}: {result_path} not found.")
        continue

    with result_path.open(encoding="utf-8") as file:
        attack_results = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    batch_successful = 0
    batch_failed = 0
    validated_results = []

    print()
    print("=" * 70)
    print(f"VALIDATING batch_{batch_index}")
    print("=" * 70)

    for result in attack_results:
        question_id = str(result["question_id"])

        if question_id not in samples_by_id:
            raise KeyError(
                f"Question ID {question_id!r} from {result_path} "
                "was not found in question.json."
            )

        sample = samples_by_id[question_id]
        question = sample["problem"]
        correct_answer = result["correct_answer"]

        attacked_image_path = Path(result["attacked_image_path"])
        if not attacked_image_path.is_absolute():
            attacked_image_path = PROJECT_ROOT / attacked_image_path

        if not attacked_image_path.exists():
            raise FileNotFoundError(
                f"Attacked image not found: {attacked_image_path}"
            )

        model_output = run_model(
            question=question,
            image_path=attacked_image_path,
        )

        predicted_answer = extract_answer(model_output)

        validated = (
            predicted_answer in VALID_ANSWERS
            and predicted_answer != correct_answer
        )

        # Keep every original field unchanged and append validation result.
        result["validated"] = validated
        validated_results.append(result)

        if validated:
            batch_successful += 1
        else:
            batch_failed += 1

        print(
            f"{question_id} | "
            f"correct={correct_answer} | "
            f"predicted={predicted_answer} | "
            f"validated={validated}"
        )

    with validated_result_path.open("w", encoding="utf-8") as file:
        for result in validated_results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    batch_evaluated = len(validated_results)
    total_evaluated += batch_evaluated
    total_successful += batch_successful
    total_failed += batch_failed

    print("-" * 70)
    print(
        f"batch_{batch_index}: "
        f"{batch_successful}/{batch_evaluated} validated successful"
    )
    print(f"Saved: {validated_result_path}")


# ---------------------------- Summary ----------------------------

print()
print("=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
print(f"Evaluated:          {total_evaluated}")
print(f"Validated success:  {total_successful}")
print(f"Validated failure:  {total_failed}")

if total_evaluated > 0:
    print(
        f"Validated attack success rate: "
        f"{total_successful / total_evaluated:.2%}"
    )
else:
    print("No attack results were found.")
