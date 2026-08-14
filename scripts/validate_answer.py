import json
import random
import re
from time import perf_counter
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
start_time = perf_counter()
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

OUTPUT_DIRECTORY = PROJECT_ROOT / "result" / "MedVLM-R1" / "bias_field_attack" / "cps_8_eps_0p3" / "batch_0"
ADVERSARIAL_IMAGE_DIRECTORY = OUTPUT_DIRECTORY / "attacked_images"


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
print(f"Evaluating images from: {ADVERSARIAL_IMAGE_DIRECTORY}")
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

all_mri_samples = [
    sample for sample in all_samples
    if sample.get("modality") == "MRI"
]

evaluated = 0
successful = 0
failed = 0

for sample in all_mri_samples:
    sample_id = sample["id"]
    question = sample["problem"]
    correct_answer = sample["solution"]

    safe_id = sample_id.replace(":", "_")
    image_path = ADVERSARIAL_IMAGE_DIRECTORY / f"{safe_id}_biased.png"

    if not image_path.exists():
        continue

    model_output = run_model(
        question=question,
        image_path=image_path,
    )

    predicted_answer = extract_answer(model_output)

    attack_success = (
        predicted_answer in VALID_ANSWERS
        and predicted_answer != correct_answer
    )

    evaluated += 1

    if attack_success:
        successful += 1
    else:
        failed += 1

    print("=" * 70)
    print(f"ID:               {sample_id}")
    print(f"Correct answer:   {correct_answer}")
    print(f"Predicted answer: {predicted_answer}")
    print(f"Attack success:   {attack_success}")
    print(f"Model output:     {model_output}")


# ---------------------------- Summary ----------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Evaluated:          {evaluated}")
print(f"Successful attacks: {successful}")
print(f"Failed attacks:     {failed}")

if evaluated > 0:
    print(f"Attack success rate: {successful / evaluated:.2%}")
else:
    print("No attacked images were found.")
    
print(f"Total evaluation time: {perf_counter() - start_time:.2f} seconds")