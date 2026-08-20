import math

import torch
import torch.nn.functional as F


def bspline(spacing, order, device="cuda", computation_dtype=torch.float32):
    """B-spline kernel for bias field interpolation."""
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


def generate_bias_field(image_tensor, bias_config, control_points=None, computation_dtype=torch.float32, verbose=1):
    """Generate a bias field from control points using B-spline interpolation."""
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have shape [batch_size, channels, height, width] or [channels, height, width].")

    epsilon = bias_config["epsilon"]
    control_point_spacing = tuple(bias_config["control_point_spacing"])
    downscale = bias_config["downscale"]
    interpolation_order = bias_config["interpolation_order"]
    random_start = bias_config["random_start"]

    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be within [0, 1).")
    if downscale < 1:
        raise ValueError("downscale must be at least 1.")

    batch_size, channels, height, width = image_tensor.shape
    device = image_tensor.device

    # Downscale image
    low_resolution_height = max(1, math.ceil(height / downscale))
    low_resolution_width = max(1, math.ceil(width / downscale))
    
    # Control point spacing
    spacing_height = max(1, control_point_spacing[0] // downscale)
    spacing_width = max(1, control_point_spacing[1] // downscale)
    spacing = (spacing_height, spacing_width)

    # Control points initialization
    if control_points is None:
        control_points_height = math.ceil(low_resolution_height / spacing_height) + 2
        control_points_width = math.ceil(low_resolution_width / spacing_width) + 2
        control_points_shape = (
            batch_size,
            1,
            control_points_height,
            control_points_width,
        )
        
        if random_start:
            # Random initialization
            lower_bound = math.log1p(-epsilon)
            upper_bound = math.log1p(epsilon)
            control_points = torch.nn.Parameter(
                torch.zeros(control_points_shape, device=device, dtype=computation_dtype).uniform_(lower_bound, upper_bound),
                requires_grad=True)
        else:
            # Identity initialization
            control_points = torch.nn.Parameter(
                torch.zeros(control_points_shape, device=device, dtype=computation_dtype),
                requires_grad=True)

        if verbose > 1:
            print(
                "Control points initialized: "
                f"{control_points_height * control_points_width}."
            )
    else:
        if control_points.ndim != 4:
            raise ValueError("control_points must have shape [B, 1, Hcp, Wcp].")
        if control_points.shape[0] != batch_size:
            raise ValueError("The control-point batch size must match the image batch size.")

    # Bspline interpolation
    bspline_kernel = bspline(
        spacing=spacing,
        order=interpolation_order,
        device=device,
        computation_dtype=computation_dtype,
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
        crop_start_h:crop_start_h + low_resolution_height,
        crop_start_w:crop_start_w + low_resolution_width,
    ]
    
    # Upscale to original image size
    upscaled_field = F.interpolate(
        low_resolution_field,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    # Bound using tanh
    unit_field = torch.tanh(upscaled_field)

    pos_scale = math.log1p(epsilon)
    neg_scale = -math.log1p(-epsilon)
    log_field = torch.where(
        unit_field >= 0,
        unit_field * pos_scale,
        unit_field * neg_scale,
    )
    bias_field = torch.exp(log_field)


    if channels > 1:
        bias_field = bias_field.expand(-1, channels, -1, -1)

    return control_points, bias_field
