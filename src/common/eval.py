import torch.nn.functional as F
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def eval_image(image1, image2):
    """Evaluate two images and return MSE, PSNR, and SSIM metrics."""
    if image1.ndim == 3:
        image1 = image1.unsqueeze(0)
    if image2.ndim == 3:
        image2 = image2.unsqueeze(0)

    if image1.shape != image2.shape:
        raise ValueError(f"Shape mismatch: image1={image1.shape}, image2={image2.shape}")

    image1 = image1.detach().float()
    image2 = image2.detach().float()

    mse = F.mse_loss(image2, image1)
    psnr = peak_signal_noise_ratio(image2, image1, data_range=1.0)
    ssim = structural_similarity_index_measure(image2, image1, data_range=1.0)

    return {
        "mse": mse.item(),
        "psnr": psnr.item(),
        "ssim": ssim.item(),
    }
