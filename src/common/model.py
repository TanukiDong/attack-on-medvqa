import os
import re
from importlib.util import find_spec
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)

from common.preprocess import process_image, tensor_to_pil


VALID_ANSWERS = ("A", "B", "C", "D")

QUESTION_TEMPLATE = """
    {Question}
    Your task:
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """

def load_model(model_config):
    """Load the model and processor"""
    model_path = model_config["path"]
    hf_cache = model_config.get("hf_cache")
    dtype = torch.bfloat16

    if hf_cache:
        os.environ["HF_HOME"] = str(hf_cache)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Submit this script to a GPU node.")

    flash_attn_available = find_spec("flash_attn") is not None
    attn_implementation = "flash_attention_2" if flash_attn_available else "sdpa"

    print(f"Using attention implementation: {attn_implementation}")
    print("Using device: cuda")
    print(f"Using dtype: {dtype}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_path)

    generation_config = GenerationConfig(
        max_new_tokens=model_config.get("max_new_tokens", 1024),
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=model_config.get("pad_token_id", 151643),
    )

    print("Model and processor loaded successfully.")

    return model, processor, generation_config


def extract_answer(output_text, tag="answer"):
    """Extract answer letter from the <answer> tags """
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", output_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def build_message(question, image_source):
    """Build a message for the model with the question and image source"""
    if isinstance(image_source, (str, Path)):
        image_source = Path(image_source).resolve().as_uri()
    elif isinstance(image_source, torch.Tensor):
        image_source = tensor_to_pil(image_source)

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_source},
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question),
                },
            ],
        }
    ]


def build_attack_input(tensor_image, problem, target, processor, answer_token_ids, loss_scope, device="cuda", reference_output=None, verbose=1):
    """"Prepare the input and labels for the model based on the loss scope."""
    if loss_scope not in {"choice_answer", "vocab_answer", "full_output"}:
        raise ValueError(f"Unsupported loss_scope: {loss_scope}")

    if tensor_image.ndim == 4:
        tensor_image = tensor_image.squeeze(0)

    message = build_message(problem, "file://placeholder.png")
    
    pil_image = tensor_to_pil(tensor_image)
    prompt_text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    if loss_scope in {"choice_answer", "vocab_answer"}:
        if target is None:
            raise ValueError("target is required for answer-only loss.")
        target_text = f"<answer>{target}</answer>"
    else: # loss_scope == "full_output"
        if reference_output is None:
            raise ValueError("reference_output is required for full-output loss.")
        target_text = reference_output.strip()

    inputs = processor(
        text=[prompt_text + target_text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    target_ids = processor.tokenizer(
        target_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    target_start = inputs.input_ids.size(1) - target_ids.size(1)
    labels = torch.full_like(inputs.input_ids, -100)

    if loss_scope in {"choice_answer", "vocab_answer"}:
    # Find the position of the answer token in the input_ids
        answer_token_id = answer_token_ids[target]
        matches = torch.where(inputs.input_ids[0, target_start:] == answer_token_id)[0]
        answer_position = target_start + matches.item()
        labels[0, answer_position] = answer_token_id
    else: # loss_scope == "full_output"
        labels[0, target_start:] = inputs.input_ids[0, target_start:]

    if verbose > 1:
        labelled_ids = labels[labels != -100]
        print("\n========== LOSS TARGET ==========")
        print(f"Loss scope: {loss_scope}")
        print(f"Token IDs: {labelled_ids.tolist()}")
        print("Decoded target:", repr(processor.tokenizer.decode(labelled_ids.tolist(), skip_special_tokens=False)))
        print("=================================\n")

    return inputs, labels


@torch.inference_mode()
def run_model(question, image, model, processor, generation_config, device="cuda"):
    message = build_message(question, image)
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, use_cache=True, generation_config=generation_config)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]


def extract_answer_token_ids(tokenizer, verbose=1):
    """Extract the token IDs corresponding to the answer letters (A, B, C, D) from the tokenizer."""
    answer_token_ids = {}

    for letter in VALID_ANSWERS:
        target_text = f"<answer>{letter}</answer>"
        encoding = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)

        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        letter_start = len("<answer>")
        letter_end = letter_start + 1

        # Find the token index that corresponds to the letter
        answer_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < letter_end and end > letter_start
        ]

        # Check that only one token is found
        if len(answer_indices) != 1:
            raise RuntimeError(f"Expected one answer-bearing token for {letter}, but found {len(answer_indices)}.")

        answer_index = answer_indices[0]
        token_id = token_ids[answer_index]
        
        # Check that the token ID decodes to a string containing the letter
        decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)

        # Check that found correct token
        if letter not in decoded_token:
                    raise RuntimeError(f"Token {token_id} decoded as {decoded_token!r}, which does not contain {letter!r}.")

        answer_token_ids[letter] = token_id

        if verbose > 1:
            print(f"{letter}: token ID={token_id}, decoded={decoded_token!r}")

    return answer_token_ids


def select_labelled_tokens(logits, labels):
    """Select the logits and labels for the labelled tokens."""
    # Align the logits and labels
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Select the logits and labels using mask
    labelled_mask = shift_labels != -100
    selected_logits = shift_logits[labelled_mask]
    selected_labels = shift_labels[labelled_mask]

    return selected_logits, selected_labels


def get_selected_logits(image, inputs, labels, model, processor, device="cuda"):
    """Get the logits corresponding to the labelled tokens in the input."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Only batch size 1 is currently supported.")
        image = image.squeeze(0)

    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W] or [1, C, H, W].")
    if image.shape[0] != 3:
        raise ValueError("MedVLM-R1 expects a three-channel RGB image.")

    pixel_values, image_grid_thw = process_image(image, processor.image_processor)

    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key
        not in {
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "labels",
        }
    }
    model_inputs["pixel_values"] = pixel_values
    model_inputs["image_grid_thw"] = image_grid_thw.to(device)

    labels = labels.to(device)
    outputs = model(**model_inputs, labels=labels, use_cache=False, return_dict=True)
    
    selected_logits, selected_labels = select_labelled_tokens(outputs.logits, labels)

    return selected_logits, selected_labels
