import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
from time import perf_counter
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_pil_image
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

# ---------------------------- Experiment settings ----------------------------
SEED = 42
VERBOSE = 1
LEARNING_RATE = 0.01
NUM_STEPS = 2000
EVAL_STEP = 10
PRINT_STEP = 100
OVERWRITE = True
VALID_ANSWERS = ("A", "B", "C", "D")
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 0

# Bias field
EPSILON = 0.3
CONTROL_POINT_SPACING = (16, 16)
DOWNSCALE = 8
INTERPOLATION_ORDER = 3
SPACE = "log"

# Attack
LOSS_TYPE = "cross_entropy"     # Options: "cross_entropy"
LOSS_SCOPE = "choice_answer"      # Options: "full_output", "choice_answer", "vocab_answer"
TARGETED = False

EXPERIMENT_NAME = f"cps_{CONTROL_POINT_SPACING[0]}_eps_{str(EPSILON).replace(".", "p")}"

# Path
MODEL_PATH = "JZPeterPan/MedVLM-R1"
HF_CACHE = Path(
    os.environ.get(
        "HF_HOME",
        "/mnt/parscratch/users/acp25tw/huggingface_cache",
    )
)
HF_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")

def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root starting from '{current}'")

PROJECT_ROOT = find_project_root()
SAMPLE_ROOT = PROJECT_ROOT / "data" / "OmniMedVQA" / "sample_mri"
QUESTION_PATH = SAMPLE_ROOT / "question.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--end-index", type=int, default=None)
parser.add_argument(
    "--output-path",
    type=Path,
    default=Path("result") / "MedVLM-R1" / "bias_field_attack" / EXPERIMENT_NAME,
)
args = parser.parse_args()

if args.start_index < 0:
    parser.error("--start-index must be at least 0.")
if args.end_index is not None and args.end_index < args.start_index:
    parser.error("--end-index must be greater than or equal to --start-index.")

OUTPUT_DIRECTORY = args.output_path
if not OUTPUT_DIRECTORY.is_absolute():
    OUTPUT_DIRECTORY = PROJECT_ROOT / OUTPUT_DIRECTORY

RESULT_FILE_PATH = OUTPUT_DIRECTORY / "attack_results.jsonl"
HISTORY_FILE_PATH = OUTPUT_DIRECTORY / "attack_history.csv"
ADVERSARIAL_IMAGE_DIRECTORY  = OUTPUT_DIRECTORY / "attacked_images"
BIAS_FIELD_IMAGE_DIRECTORY = OUTPUT_DIRECTORY / "bias_fields"

def get_output_paths(sample_id):
    sample_id = sample_id.replace(":", "_")
    attacked_image_path = (ADVERSARIAL_IMAGE_DIRECTORY  / f"{sample_id}_biased.png")
    bias_field_path = (BIAS_FIELD_IMAGE_DIRECTORY / f"{sample_id}_bias_field.pt")
    return attacked_image_path, bias_field_path

def relative_to_project(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve()))

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

if OVERWRITE:
    if VERBOSE:
        print(f"Deleting existing output directory due to OVERWRITE flag: {relative_to_project(OUTPUT_DIRECTORY)}")
    shutil.rmtree(OUTPUT_DIRECTORY, ignore_errors=True)
    
for directory in (OUTPUT_DIRECTORY, ADVERSARIAL_IMAGE_DIRECTORY , BIAS_FIELD_IMAGE_DIRECTORY):
    directory.mkdir(parents=True, exist_ok=True)
    
if OVERWRITE or not RESULT_FILE_PATH.exists():
    RESULT_FILE_PATH.touch()
    
if OVERWRITE or not HISTORY_FILE_PATH.exists():
    with HISTORY_FILE_PATH.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=HISTORY_FIELDS).writeheader()

def load_completed_ids(result_path):
    if not result_path.exists():
        return set()

    with result_path.open(encoding="utf-8") as file:
        return {
            str(json.loads(line)["question_id"])
            for line in file
            if line.strip()
        }
        
with QUESTION_PATH.open(encoding="utf-8") as file:
    all_samples = json.load(file)

all_mri_samples = [sample for sample in all_samples if sample.get("modality") == "MRI"]
selected_mri_samples = all_mri_samples[args.start_index:args.end_index]
completed_question_ids = load_completed_ids(RESULT_FILE_PATH)
mri_samples = (
    selected_mri_samples
    if OVERWRITE
    else [sample for sample in all_mri_samples
        if str(sample["id"]) not in completed_question_ids])
    
print(
    f"Loaded {len(all_mri_samples)} MRI samples, "
    f"selected indices [{args.start_index}:{args.end_index}], "
    f"{len(mri_samples)} samples remaining to be processed."
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Submit this script to a GPU node."
    )

device = "cuda"
dtype = torch.bfloat16
computation_dtype = torch.float32
flash_attn_available = find_spec("flash_attn") is not None
attn_implementation = "flash_attention_2" if flash_attn_available else "sdpa"
print(f"Using attention implementation: {attn_implementation}")
print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

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
image_processor = processor.image_processor
tokenizer = processor.tokenizer

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=False,
    num_return_sequences=1,
    pad_token_id=151643,
)

print("Model and processor loaded successfully.")

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
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", output_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def load_image_tensor(image_path):
    image = read_image(image_path).float().div(255.0)

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] > 3:
        image = image[:3]
        
    return image.clamp(0, 1).to(device)

def tensor_to_pil(image_tensor):
    if image_tensor.ndim == 4:
        if image_tensor.shape[0] != 1:
            raise ValueError("Only batch size one is supported")
        image_tensor = image_tensor.squeeze(0)
    if image_tensor.ndim != 3:
        raise ValueError(
            f"Expected image shape [C, H, W] or [1, C, H, W], but got {tuple(image_tensor.shape)}.")
    return to_pil_image(image_tensor.detach().cpu().float().clamp(0, 1))

def get_image_info(image_tensor):
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size
    resized_h, resized_w = smart_resize(
        image_tensor.shape[-2],
        image_tensor.shape[-1],
        factor=patch_size * merge_size,
        min_pixels=image_processor.size["shortest_edge"],
        max_pixels=image_processor.size["longest_edge"],
    )
    
    grid_t = 1
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    
    image_grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]],
        device=image_tensor.device,
        dtype=torch.long,
    )
    
    return resized_h, resized_w, image_grid_thw

def quantize_uint8_ste(image):
    image = image.clamp(0, 1)
    quantized = torch.floor(image * 255.0) / 255.0
    return image + (quantized - image).detach()

def clamp_ste(tensor, min_value=0.0, max_value=1.0):
    clipped = tensor.clamp(min_value, max_value)
    return tensor + (clipped - tensor).detach()

def process_image(image):
    if image.ndim == 4:
        image = image.squeeze(0)
    
    # Quantize image
    image = quantize_uint8_ste(image)
    
    # Resize image
    resized_h, resized_w, grid = get_image_info(image)
    resized = resize(
        image, [resized_h, resized_w],
        interpolation=InterpolationMode.BICUBIC, antialias=True,
    ).clamp(0, 1)
    
    patches = resized.unsqueeze(0)
    
    # Normalize
    mean = patches.new_tensor(image_processor.image_mean).view(1, 3, 1, 1)
    std = patches.new_tensor(image_processor.image_std).view(1, 3, 1, 1)
    patches = (patches - mean) / std

    # Patch size
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size
    temporal_patch_size = image_processor.temporal_patch_size
    if patches.shape[0] % temporal_patch_size:
        repeat_count = temporal_patch_size - patches.shape[0] % temporal_patch_size
        patches = torch.cat([patches, patches[-1:].repeat(repeat_count, 1, 1, 1)], dim=0)
        
    # Grid size
    channels = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    
    # Reshape
    patches = patches.reshape(
        grid_t, temporal_patch_size, channels,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
   
    pixel_values = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    
    return pixel_values.to(dtype=dtype), grid

def build_message(question, image_source):
    
    if isinstance(image_source, (str, Path)):
        image_source = Path(image_source).resolve().as_uri()
    elif isinstance(image_source, torch.Tensor):
        image_source = tensor_to_pil(image_source)
        
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_source},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)},
        ],
    }]

def build_attack_input(tensor_image, problem, target, reference_output=None, loss_scope=LOSS_SCOPE):
    
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
    
    target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device)
          
    target_start = inputs.input_ids.size(1) - target_ids.size(1)
    labels = torch.full_like(inputs.input_ids, -100)
    
    if loss_scope in {"choice_answer", "vocab_answer"}:
    # Find the position of the answer token in the input_ids
        answer_token_id = ANSWER_TOKEN_IDS[target]
        matches = torch.where(inputs.input_ids[0, target_start:] == answer_token_id)[0]
        answer_position = target_start + matches.item()
        labels[0, answer_position] = answer_token_id
    else: # loss_scope == "full_output"
        labels[0, target_start:] = inputs.input_ids[0, target_start:]
    
    if VERBOSE > 1:
        labelled_ids = labels[labels != -100]

        print("\n========== LOSS TARGET ==========")
        print(f"Loss scope: {loss_scope}")
        print(f"Token IDs: {labelled_ids.tolist()}")
        print(
            "Decoded target:",
            repr(
                tokenizer.decode(
                    labelled_ids.tolist(),
                    skip_special_tokens=False,
                )
            ),
        )
        print("=================================\n")
        
    return inputs, labels

@torch.inference_mode()
def run_model(question, image):

    message = build_message(question, image)
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) 
    image_inputs, video_inputs = process_vision_info(message)
    
    inputs_adv = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    generated_ids = model.generate(**inputs_adv, use_cache=True, generation_config=generation_config)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_adv.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text[0]

def bspline(spacing, order=INTERPOLATION_ORDER):
    
    # Validation checks
    if len(spacing) != 2:
        raise ValueError("spacing must contain two values: (height, width).")
    
    spacing_height, spacing_width = map(int, spacing)
    
    if spacing_height < 1 or spacing_width < 1:
        raise ValueError("Spacing values must be at least 1.")
    
    if order < 1:
        raise ValueError("B-spline order must be at least 1.")
    
    # Initialize kernel
    mean_kernel = torch.ones(1,1, spacing_height, spacing_width, device=device, dtype=computation_dtype)
    kernel = mean_kernel.clone()
    normalization = spacing_height * spacing_width
    
    # Iterative convolution
    
    for iteration in range(1, order + 1):
        padding = (iteration * spacing_height, iteration * spacing_width)
        kernel = F.conv2d(kernel, mean_kernel, padding=padding)
        kernel /= normalization
    
    return kernel

def generate_bias_field(image_tensor, control_points=None):
    
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Validation checks
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have shape [batch_size, channels, height, width] or [channels, height, width].")
    
    if not 0 <= EPSILON < 1:
        raise ValueError("EPSILON must be within [0, 1).")
    
    if DOWNSCALE < 1:
        raise ValueError("DOWNSCALE must be at least 1.")
        
    batch_size, channels, height, width = image_tensor.shape
    
    # Downscale image
    low_resolution_height = max(1, math.ceil(height / DOWNSCALE))
    low_resolution_width = max(1, math.ceil(width / DOWNSCALE))
    
    # Control point spacing
    spacing_height = max(1, CONTROL_POINT_SPACING[0] // DOWNSCALE)
    spacing_width = max(1, CONTROL_POINT_SPACING[1] // DOWNSCALE)
    spacing = (spacing_height, spacing_width)
    
    # Control points initialization
    if control_points is None:
        control_points_height = math.ceil(low_resolution_height / spacing_height) + 2
        control_points_width = math.ceil(low_resolution_width / spacing_width) + 2
        control_points_shape = (batch_size, 1, control_points_height, control_points_width)
        # Identity initialization
        control_points = torch.nn.Parameter(
            torch.zeros(control_points_shape, device=device, dtype=computation_dtype),
            requires_grad=True)
        # # Bound initialization
        # lower_bound = math.log1p(-EPSILON)
        # upper_bound = math.log1p(EPSILON)
        # control_points = torch.nn.Parameter(
        #     torch.zeros(control_points_shape, device=device, dtype=computation_dtype).uniform_(lower_bound, upper_bound),
        #     requires_grad=True)
        if VERBOSE > 1:
            print(f"Control points initialized : {control_points_height * control_points_width}.")

    else:
        if control_points.ndim != 4:
            raise ValueError("control_points must have shape [B, 1, Hcp, Wcp].")

        if control_points.shape[0] != batch_size:
            raise ValueError("The control-point batch size must match the image batch size.")

    # Bspline interpolation
    bspline_kernel = bspline(
        spacing=spacing,
        order=INTERPOLATION_ORDER,
    )
    padding = ((bspline_kernel.shape[-2] - 1) // 2, (bspline_kernel.shape[-1] - 1) // 2)

    interpolated_field = F.conv_transpose2d(
        control_points,
        bspline_kernel,
        stride=spacing,
        padding=padding,
    )

    # Crop at the center
    crop_start_h = (interpolated_field.shape[-2] - low_resolution_height) // 2
    crop_start_w = (interpolated_field.shape[-1] - low_resolution_width) // 2
    low_resolution_field = interpolated_field[
        :,
        :,
        crop_start_h : crop_start_h + low_resolution_height,
        crop_start_w : crop_start_w + low_resolution_width,
    ]
    
    # Upscale to original image size
    upscaled_field = F.interpolate(
        low_resolution_field,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    
    # Smooth tanh bound
    unit_field = torch.tanh(upscaled_field)
    if SPACE == "log":
        pos_scale = math.log1p(EPSILON)
        neg_scale = -math.log1p(-EPSILON)
        
        log_field = torch.where(
            unit_field >= 0,
            unit_field * pos_scale,
            unit_field * neg_scale,
        )
        
        bias_field = torch.exp(log_field)

    elif SPACE == "linear": # SPACE == "linear"
        bias_field = 1.0 + unit_field * EPSILON
    else:
        raise ValueError(f"Unsupported SPACE: {SPACE}. Use 'log' or 'linear'.")
    
    # Expand to other channels if necessary
    if channels > 1:
        bias_field = bias_field.expand(-1, channels, -1, -1,)
        
    return control_points, bias_field
    
def eval_image(image1, image2):
    if image1.ndim == 3:
        image1 = image1.unsqueeze(0)

    if image2.ndim == 3:
        image2 = image2.unsqueeze(0)

    if image1.shape != image2.shape:
        raise ValueError(
            f"Shape mismatch: image1={image1.shape}, image2={image2.shape}")

    # Use float32 for stable metric calculation
    image1 = image1.detach().float()
    image2 = image2.detach().float()

    mse = F.mse_loss(image2, image1)

    psnr = peak_signal_noise_ratio(
        image2,
        image1,
        data_range=1.0,
    )

    ssim = structural_similarity_index_measure(
        image2,
        image1,
        data_range=1.0,
    )

    return {
        "mse": mse.item(),
        "psnr": psnr.item(),
        "ssim": ssim.item(),
    }
    
def extract_answer_token_ids(tokenizer):
    answer_token_ids = {}

    for letter in VALID_ANSWERS:
        target_text = f"<answer>{letter}</answer>"

        encoding = tokenizer(
            target_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

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
        decoded_token = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        # Check that found correct token
        if letter not in decoded_token:
            raise RuntimeError(f"Token {token_id} decoded as {decoded_token!r}, which does not contain {letter!r}.")

        answer_token_ids[letter] = token_id

        if VERBOSE > 1:
            print(
                f"{letter}: token ID={token_id}, "
                f"decoded={decoded_token!r}"
            )

    return answer_token_ids

def compute_loss(image, inputs, labels, target, loss_type=LOSS_TYPE, loss_scope=LOSS_SCOPE, targeted=TARGETED, clean_probs=None):
    
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("Only batch size 1 is currently supported.")
            image = image.squeeze(0)
        if image.ndim != 3:
            raise ValueError("image must have shape [C, H, W] or [1, C, H, W].")

        if image.shape[0] != 3:
            raise ValueError("MedVLM-R1 expects a three-channel RGB image.")
        
        pixel_values, image_grid_thw = process_image(image)
        
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {
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
        
        if loss_type == "cross_entropy":
            if loss_scope in {"vocab_answer", "full_output"}:
                loss = F.cross_entropy(
                    input=selected_logits.float(),
                    target=selected_labels
                    )
            
            elif loss_scope == "choice_answer":
                loss = choice_ce_loss(
                    logits=selected_logits,
                    target=target
                )
            else:
                raise ValueError(f"Unsupported loss_scope for cross_entropy: {loss_scope}")
        elif loss_type == "kl":
            if clean_probs is None:
                raise ValueError("KL loss requires clean_probs.")
            
            log_probs = F.log_softmax(selected_logits.float(), dim=-1)
            loss = F.kl_div(
                input=log_probs,
                target=clean_probs,
                reduction="batchmean" # See https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.kl_div.html
                )
        else:
            raise ValueError(
                f"Unsupported loss type: {loss_type}"
            )
            
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Untargeted loss is not finite: {loss.item()}"
            )
        if not loss.requires_grad:
            raise RuntimeError("The loss has no gradient")
        
        
        if loss_scope == "choice_answer":
            choice_token_ids = torch.tensor(
                [ANSWER_TOKEN_IDS[x] for x in VALID_ANSWERS],
                device=selected_logits.device,
            )

            choice_probs = torch.softmax(
                selected_logits[0, choice_token_ids].float(),
                dim=0,
            )
        else:
            choice_probs = None
        
        return loss, choice_probs
    
def select_labelled_tokens(logits, labels):
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    labelled_mask = shift_labels != -100
    selected_logits = shift_logits[labelled_mask]
    selected_labels = shift_labels[labelled_mask]
    
    return selected_logits, selected_labels

def choice_ce_loss(logits, target, targeted=TARGETED):
    
    # Validation
    if logits.shape[0] != 1:
        raise RuntimeError(f"choice_ce_loss expects exactly one answer position but got {logits.shape[0]} positions.")

    # Select only the answer choices logits
    choice_token_ids = torch.tensor(
        [
            ANSWER_TOKEN_IDS["A"],
            ANSWER_TOKEN_IDS["B"],
            ANSWER_TOKEN_IDS["C"],
            ANSWER_TOKEN_IDS["D"],
        ],
        device=logits.device,
        dtype=torch.long,
    )
    
    # Reduce the full vocabulary logits to only the answer choice logits
    choice_logits = logits[0, choice_token_ids].float()
    
    target_index = torch.tensor(
        [VALID_ANSWERS.index(target)],
        device=choice_logits.device,
        dtype=torch.long,
    )

    ce_loss = F.cross_entropy(
        choice_logits.unsqueeze(0),
        target_index,
    )

    # The bias-field parameters are updated with gradient ascent.
    if targeted:
        total_loss = -ce_loss
    else:
        total_loss = ce_loss
        
    return total_loss

def attack_bf(image, problem, target, reference_output=None, num_steps=NUM_STEPS, learning_rate=LEARNING_RATE, eval_step=EVAL_STEP, early_stopping=EARLY_STOPPING, patience=EARLY_STOPPING_PATIENCE):

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.ndim != 4:
        raise ValueError("image must have shape [C, H, W] or [B, C, H, W].")
    
    if eval_step < 1:
        raise ValueError("eval_step must be at least 1.")

    if early_stopping and patience < 0:
        raise ValueError(
            "patience must be at least 0 when early stopping is enabled."
        )
    
    if VERBOSE:
        print(f"Question: {problem}")
        print(f"Solution: {target}")
    
    # Detach image from computation graph
    image = image.detach().to(device=device, dtype=computation_dtype)
    
    # Build attack inputs and labels
    inputs, labels = build_attack_input(
        tensor_image=image,
        problem=problem,
        target=target,
        reference_output=reference_output
        )

    # Initialize control points and bias field
    control_points, bias_field = generate_bias_field(image_tensor=image, control_points=None)
    
    # Adam optimizer
    optimizer = torch.optim.Adam([control_points], lr=learning_rate)

    history = []
    best_candidate = None
    patience_counter = None

    for step in range(num_steps):

        # # Restart computation graph
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        # Generate bias field
        control_points, bias_field = generate_bias_field(image_tensor=image, control_points=control_points)

        # adversarial_image = (image * bias_field).clamp(0, 1)
        adversarial_image = clamp_ste(image * bias_field, 0, 1)

        # Calculate the correct-answer loss.
        loss, choice_probs = compute_loss(
            image=adversarial_image,
            inputs=inputs,
            labels=labels,
            target=target,
        )
        loss_value = loss.detach().item()

        # Backpropagation
        adam_loss = -loss
        adam_loss.backward()
        
        # Attack history
        history_entry = {
            "step": step + 1,
            "loss": loss_value,
            "predicted_answer": None,
            "attack_success": None,
            "evaluated": False,
        }
        
        if (step + 1) % eval_step == 0 or step == num_steps - 1:
            intermediate_output = run_model(question=problem, image=adversarial_image.detach())
            intermediate_answer = extract_answer(intermediate_output, tag="answer")
            
            valid_answer = intermediate_answer in VALID_ANSWERS
            
            if TARGETED:
                attack_success = valid_answer and intermediate_answer == target
            else:
                attack_success = valid_answer and intermediate_answer != target
            
            history_entry["predicted_answer"] = intermediate_answer
            history_entry["attack_success"] = attack_success
            history_entry["evaluated"] = True
                        
            # Criteria 1 : No best candidate yet
            # Criteria 2 : Success when no success yet
            # Criteria 3 : No success yet → Higher loss is better
            # Criteria 4 : Success when success already → Lower loss is better
            is_better = (
                best_candidate is None
                or (attack_success and not best_candidate["attack_success"])
                or (attack_success and best_candidate["attack_success"] and loss_value < best_candidate["loss"])
                or (not attack_success and not best_candidate["attack_success"] and loss_value > best_candidate["loss"])
                )
            successful_candidate_found = (
                best_candidate is not None
                and best_candidate["attack_success"]
                )
            
            if is_better:
                best_candidate = {
                    "step": step + 1,
                    "loss": loss_value,
                    "attack_success": attack_success,
                    "control_points": control_points.detach().clone(),
                    "image": adversarial_image.detach().clone(),
                    "bias_field": bias_field.detach().clone(),
                    "output": intermediate_output,
                    "answer": intermediate_answer,
                }
                if attack_success:
                    patience_counter = patience
                    
                if VERBOSE:
                    print(f"New best candidate found at step {step + 1}")
            elif successful_candidate_found:
                patience_counter -= 1
            
            
            if VERBOSE and (step + 1) % PRINT_STEP == 0:        
                with torch.no_grad():
                    control_max = control_points.max().item()
                    control_min = control_points.min().item()
                    field_min = bias_field.min().item()
                    field_max = bias_field.max().item()
                print(
                    f"Step {step + 1:02d} | "
                    f"Loss: {loss_value:.6f} | "
                    f"Control points range: [{control_min:.6f}, {control_max:.6f}] | "
                    f"Bias field range: [{field_min:.6f}, {field_max:.6f}]"
                )
                print(repr(intermediate_output))
                print("Answer probabilities:")
                for letter, probability in zip(VALID_ANSWERS, choice_probs):
                    print(f"  {letter}: {probability.item() * 100:.2f}%")


        if choice_probs is not None:
            history_entry["answer_probabilities"] = {
                letter: probability.detach().cpu().item()
                for letter, probability in zip(VALID_ANSWERS, choice_probs)
            }
        history.append(history_entry)

        if early_stopping and patience_counter is not None and patience_counter <= 0:
            if VERBOSE:
                print(f"Early stopping triggered at step {step + 1} after not improving for {patience} steps.")
            break
        
        optimizer.step()
        
    if VERBOSE:
        print("\nAttack results:")
        print(f"Best step: {best_candidate['step']} | Best loss: {best_candidate['loss']:.6f}")
        print("Best candidate :")
        [print(f"  {key}: {value}") for key, value in best_candidate.items() if key not in {"image", "bias_field", "control_points", "output"}]

    # Image Loss
    image_loss = eval_image(image, best_candidate["image"])
    best_candidate["image_loss"] = image_loss
    if VERBOSE:
        print("Image evaluation metrics after attack:")
        print(f"  MSE: {image_loss['mse']:.6f}")
        print(f"  PSNR: {image_loss['psnr']:.2f} dB")
        print(f"  SSIM: {image_loss['ssim']:.4f}")
    
    return best_candidate, history

if __name__ == "__main__":
    torch.cuda.synchronize()
    total_start = perf_counter()
    set_seed(SEED)
    ANSWER_TOKEN_IDS = extract_answer_token_ids(tokenizer)
    
    for sample_index, sample in enumerate(tqdm(mri_samples, desc="MRI VQA tasks", disable=not sys.stdout.isatty())):
        
        torch.cuda.synchronize()
        sample_start_time = perf_counter()
    
        question_id = str(sample["id"])
        image_path = SAMPLE_ROOT / sample.get("image")[0]

        image = load_image_tensor(image_path)
        problem = sample.get("problem")
        solution = sample.get("solution")

        if VERBOSE:
            print(f"\nProcessing sample {sample_index + 1}/{len(mri_samples)}: ID={question_id}")

        clean_output = run_model(problem, image_path)
        clean_answer = extract_answer(clean_output, tag="answer")
        
        adversarial_output, attack_history = attack_bf(
            image=image,
            problem=problem,
            target=solution,
            reference_output=clean_output,
            num_steps=NUM_STEPS,
            learning_rate=LEARNING_RATE
            )
        
        # Save image
        attacked_image_path, bias_field_path = get_output_paths(question_id)
        tensor_to_pil(adversarial_output["image"]).save(attacked_image_path)
        torch.save(adversarial_output["bias_field"], bias_field_path)
        adversarial_output["attacked_image_path"] = attacked_image_path
        adversarial_output["bias_field_path"] = bias_field_path

        if VERBOSE > 1:
            print("\n========== CLEAN MODEL OUTPUT ==========")
            print(repr(clean_output))
            print("========================================")
            print("======= ADVERSARIAL MODEL OUTPUT =======")
            print(repr(adversarial_output["output"]))
            print("========================================")
            print(f"Attack Success: {adversarial_output['attack_success']}")


        attack_result = {
            "question_id": question_id,
            "correct_answer": solution,
            "adversarial_answer": adversarial_output["answer"],
            "attack_success": adversarial_output["attack_success"],
            
            "control_point_spacing": CONTROL_POINT_SPACING[0],
            "epsilon": EPSILON,
            "downscale": DOWNSCALE,
            "interpolation_order": INTERPOLATION_ORDER,
            "space": SPACE,
            
            "learning_rate": LEARNING_RATE,
            "steps": len(attack_history),
            "eval_step": EVAL_STEP,
            
            "best_step": adversarial_output["step"],
            "best_loss": adversarial_output["loss"],
            "image_loss": adversarial_output["image_loss"],
            
            "original_image_path": relative_to_project(image_path),
            "attacked_image_path": relative_to_project(adversarial_output["attacked_image_path"]),
            "bias_field_path": relative_to_project(adversarial_output["bias_field_path"]),
        }
        
        # Save results
        with RESULT_FILE_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(attack_result) + "\n")
        
        # Save history
        with HISTORY_FILE_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)

            for entry in attack_history:
                probabilities = entry.get("answer_probabilities", {})

                writer.writerow({
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
                })
        torch.cuda.synchronize()
        sample_time = perf_counter() - sample_start_time
        print(f"Saved results for ID:{question_id} | {sample_index + 1}/{len(mri_samples)} | Time: {sample_time:.2f}s ({sample_time / 60:.2f}m)")
    
    torch.cuda.synchronize()
    total_time = perf_counter() - total_start
    print(f"\nFinished running script.")
    print(f"Total runtime: {total_time:.2f}s ({total_time / 60:.2f}m, {total_time / 3600:.2f}h)")
