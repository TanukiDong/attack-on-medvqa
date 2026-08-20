# from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_pil_image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

MODALITY = {
    "mri": "MRI",
    "ct": "CT",
    "us": "Ultrasound",
}

dtype = torch.bfloat16

def load_image_tensor(image_path, device="cuda"):
    """Load image from path and convert to tensor."""
    image = read_image(str(image_path)).float().div(255.0)

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] > 3:
        image = image[:3]

    return image.clamp(0, 1).to(device)


def tensor_to_pil(image_tensor):
    """Convert tensor image to PIL image."""
    if image_tensor.ndim == 4:
        if image_tensor.shape[0] != 1:
            raise ValueError("Only batch size one is supported.")
        image_tensor = image_tensor.squeeze(0)

    if image_tensor.ndim != 3:
        raise ValueError(
            "Expected image shape [C, H, W] or [1, C, H, W], "
            f"but got {tuple(image_tensor.shape)}."
        )

    return to_pil_image(image_tensor.detach().cpu().float().clamp(0, 1))


def get_image_info(image_tensor, image_processor):
    """Get height, width, and grid size of the resized image."""
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size

    # Resize image
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

    # Create a tensor for the grid size
    image_grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]],
        device=image_tensor.device,
        dtype=torch.long,
    )

    return resized_h, resized_w, image_grid_thw

def process_image(image, image_processor):
    """Process image for the model"""
    if image.ndim == 4:
        image = image.squeeze(0)
    
    # Resize image
    resized_h, resized_w, grid = get_image_info(image, image_processor)
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