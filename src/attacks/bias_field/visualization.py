import matplotlib.pyplot as plt
import torch

def plot_bias_field_attack(
    image,
    bias_field,
    biased_image,
    epsilon=0.3,
    step=None,
    loss=None,
    predicted_answer=None,
    attack_success=None,
    debug_dir=None,
):
    # Remove batch dimension
    if image.ndim == 4:
        image = image.squeeze(0)
    if bias_field.ndim == 4:
        bias_field = bias_field.squeeze(0)
    if biased_image.ndim == 4:
        biased_image = biased_image.squeeze(0)

    image = image[0]
    bias_field = bias_field[0]
    biased_image = biased_image[0]

    # Absolute difference
    difference = torch.abs(biased_image - image)

    # Save image
    torch.save(bias_field, debug_dir["bias_field_directory"] / f"bias_field_{step:04d}.pt")
    plt.imsave(debug_dir["bias_field_directory"] / f"bias_field_{step:04d}.png", bias_field, cmap="jet", vmin=1 - epsilon, vmax=1 + epsilon)
    plt.imsave(debug_dir["attacked_image_directory"] / f"attacked_image_{step:04d}.png", biased_image, cmap="gray", vmin=0, vmax=1)
    plt.imsave(debug_dir["difference_directory"] / f"difference_{step:04d}.png", difference, cmap="binary", vmin=0, vmax=1)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Original image
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Bias field
    bias_plot = axes[1].imshow(bias_field, cmap="jet", vmin=1 - epsilon, vmax=1 + epsilon)
    axes[1].set_title(f"Bias Field \n min={bias_field.min().item():.4f}, max={bias_field.max().item():.4f}")
    axes[1].axis("off")
    fig.colorbar(bias_plot, ax=axes[1])

    # Attacked image
    axes[2].imshow(biased_image, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Image x Bias Field")
    axes[2].axis("off")

    # Difference
    diff_plot = axes[3].imshow(difference, cmap="binary", vmin=0, vmax=1)
    axes[3].set_title(f"Absolute Difference \n mean={difference.mean().item():.4f}, max={difference.max().item():.4f}")
    axes[3].axis("off")
    fig.colorbar(diff_plot, ax=axes[3])


    # Title
    title = []

    if step is not None:
        title.append(f"Step {step}")

    if loss is not None:
        title.append(f"Loss: {loss:.6f}")

    if predicted_answer is not None:
        title.append(f"Answer: {predicted_answer}")

    if attack_success is not None:
        title.append(f"Success: {attack_success}")

    if title:
        fig.suptitle(" | ".join(title), fontsize=13)

    plt.tight_layout()
    plt.show()
    
def plot_progressive_bias_field_attack(
    image,
    bias_field_directory,
    epsilon=0.3,
    columns=5,
):

    # Load bias fields
    bias_field_paths = sorted(
        bias_field_directory.glob("bias_field_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    
    for start in range(0, len(bias_field_paths), columns):

        paths = bias_field_paths[start:start + columns]
        n = len(paths)

        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)

        for col, bias_field_path in enumerate(paths):

            # Step number
            step = int(bias_field_path.stem.split("_")[-1])

            # Load bias field
            bias_field = torch.load(bias_field_path, map_location="cpu").float()

            # Reconstruct biased image
            biased_image = torch.clamp(image * bias_field, min=0, max=1)

            # Bias field
            axes[0, col].imshow(bias_field, cmap="jet", vmin=1 - epsilon, vmax=1 + epsilon)
            axes[0, col].set_title(f"Step {step}")
            axes[0, col].axis("off")

            # Biased image
            axes[1, col].imshow(biased_image.permute(1, 2, 0))
            axes[1, col].set_title(f"Biased image")
            axes[1, col].axis("off")

        plt.tight_layout()
        plt.show()