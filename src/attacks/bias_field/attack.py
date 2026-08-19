import torch

from attacks.bias_field.bias_field import clamp_ste, generate_bias_field
from attacks.bias_field.loss import compute_loss, get_choice_probs
from attacks.bias_field.visualization import plot_bias_field_attack
from common.eval import eval_image
from common.model import (
    VALID_ANSWERS,
    build_attack_input,
    extract_answer,
    get_selected_logits,
    run_model,
)

def attack_bf(
    image,
    problem,
    target,
    reference_output,
    model,
    processor,
    generation_config,
    answer_token_ids,
    attack_config,
    bias_config,
    verbose=1,
    device="cuda",
    debug=False,
    debug_dir=None,
):
    num_steps = attack_config["num_steps"]
    learning_rate = attack_config["learning_rate"]
    eval_step = attack_config["eval_step"]
    print_step = attack_config["print_step"]
    early_stopping = attack_config["early_stopping"]
    patience = attack_config["early_stopping_patience"]
    loss_type = attack_config["loss_type"]
    loss_scope = attack_config["loss_scope"]

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("image must have shape [C, H, W] or [B, C, H, W].")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if eval_step < 1:
        raise ValueError("eval_step must be at least 1.")
    if print_step < 1:
        raise ValueError("print_step must be at least 1.")
    if early_stopping and patience < 0:
        raise ValueError("early_stopping_patience must be at least 0 when early stopping is enabled.")

    if verbose:
        print(f"Question: {problem}")
        print(f"Solution: {target}")

    # Detach image from computation graph
    image = image.detach().to(device=device, dtype=torch.float32)

    # Build attack inputs and labels
    inputs, labels = build_attack_input(
        tensor_image=image,
        problem=problem,
        target=target,
        processor=processor,
        answer_token_ids=answer_token_ids,
        reference_output=reference_output,
        loss_scope=loss_scope,
        verbose=verbose,
    )

    # Initialize control points and bias field
    control_points, bias_field = generate_bias_field(
        image_tensor=image,
        bias_config=bias_config,
        control_points=None,
        verbose=verbose,
    )
    
    # Debug visualization
    if debug:
        initial_bias_field = clamp_ste(image * bias_field, 0, 1)
        plot_bias_field_attack(
            image=image.detach().cpu().float(),
            bias_field=bias_field.detach().cpu().float(),
            biased_image=initial_bias_field.detach().cpu().float(),
            epsilon=bias_config["epsilon"],
            step=0,
            loss=None,
            predicted_answer=None,
            attack_success=None,
            debug_dir=debug_dir,
        )

    # Adam optimizer
    optimizer = torch.optim.Adam([control_points], lr=learning_rate)

    # Clean probs for KL loss
    clean_probs = None
    if loss_type == "kl":
        with torch.no_grad():
            clean_logits, _ = get_selected_logits(
                image=image,
                inputs=inputs,
                labels=labels,
                model=model,
                processor=processor,
                device=device,
            )

            clean_probs = get_choice_probs(
                clean_logits,
                answer_token_ids,
            ).detach()
            
    history = []
    best_candidate = None
    patience_counter = None
    
    # Attack loop
    for step in range(num_steps):
        
        # Restart computation graph
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        # Generate bias field
        control_points, bias_field = generate_bias_field(
            image_tensor=image,
            bias_config=bias_config,
            control_points=control_points,
            verbose=verbose,
        )

        # Clamp image into [0, 1]
        adversarial_image = clamp_ste(image * bias_field, 0, 1)

        evaluate_step = (step + 1) % eval_step == 0 or step == num_steps - 1
        
        if evaluate_step:
            # Run model
            intermediate_output = run_model(
                question=problem,
                image=adversarial_image.detach(),
                model=model,
                processor=processor,
                generation_config=generation_config,
                device=device,
            )
            
            intermediate_answer = extract_answer(intermediate_output, tag="answer")
            
            if loss_scope == "conditioned_answer":
                try:
                    inputs, labels = build_attack_input(
                        tensor_image=adversarial_image.detach(),
                        problem=problem,
                        target=target,
                        processor=processor,
                        answer_token_ids=answer_token_ids,
                        reference_output=intermediate_output,
                        loss_scope=loss_scope,
                        device=device,
                        verbose=verbose,
                    )
                except RuntimeError as error:
                    print(f"Error building attack input at step {step + 1}: {error}")
                    print("Keeping previous reasoning")

            valid_answer = intermediate_answer in VALID_ANSWERS
            attack_success = valid_answer and intermediate_answer != target 

        # Select logits and labels based on the loss scope
        selected_logits, selected_labels = get_selected_logits(
            image=adversarial_image,
            inputs=inputs,
            labels=labels,
            model=model,
            processor=processor,
            device=device,
        )

        # Compute loss
        loss, choice_probs = compute_loss(
            selected_logits=selected_logits,
            selected_labels=selected_labels,
            target=target,
            answer_token_ids=answer_token_ids,
            attack_config=attack_config,
            clean_probs=clean_probs,
        )
        loss_value = loss.detach().item()

        # Gradient Ascent
        adam_loss = -loss
        
        # Backpropagation
        adam_loss.backward()

        # Attack history
        history_entry = {
            "step": step + 1,
            "loss": loss_value,
            "predicted_answer": intermediate_answer if evaluate_step else None,
            "attack_success": attack_success if evaluate_step else None,
            "evaluated": evaluate_step,
        }
        if choice_probs is not None:
            history_entry["choice_probs"] = {
                letter: probability.detach().cpu().item()
                for letter, probability in zip(VALID_ANSWERS, choice_probs)
            }
            
        if evaluate_step:
            
            # Debug visualization
            if debug:
                plot_bias_field_attack(
                    image=image.detach().cpu().float(),
                    bias_field=bias_field.detach().cpu().float(),
                    biased_image=adversarial_image.detach().cpu().float(),
                    epsilon=bias_config["epsilon"],
                    step=step + 1,
                    loss=loss_value,
                    predicted_answer=intermediate_answer,
                    attack_success=attack_success,
                    debug_dir=debug_dir,
                )
                
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

                if verbose > 1:
                    print(f"New best candidate found at step {step + 1}")

            elif successful_candidate_found:
                patience_counter -= 1

            # Print progress
            if verbose and (step + 1) % print_step == 0:
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

                if choice_probs is not None:
                    if loss_scope == "conditioned_answer":
                        print("Conditioned choice probabilities:")
                    else:
                        print("Choice probabilities:")
                    for letter, probability in zip(VALID_ANSWERS, choice_probs):
                        print(f"  {letter}: {probability.item() * 100:.2f}%")


        history.append(history_entry)

        # Early stopping
        if early_stopping and patience_counter is not None and patience_counter <= 0:
            if verbose:
                print(f"Early stopping triggered at step {step + 1} after not improving for {patience} steps.")
            break
        
        # Update control points
        optimizer.step()

    if verbose:
        print("\nAttack results:")
        print(f"Best step: {best_candidate['step']} | Best loss: {best_candidate['loss']:.6f}")
        print("Best candidate :")
        [print(f"  {key}: {value}") for key, value in best_candidate.items() if key not in {"image", "bias_field", "control_points", "output"}]

    # Image loss
    image_loss = eval_image(image, best_candidate["image"])
    best_candidate["image_loss"] = image_loss

    if verbose:
        print("Image evaluation metrics after attack:")
        print(f"  MSE: {image_loss['mse']:.6f}")
        print(f"  PSNR: {image_loss['psnr']:.2f} dB")
        print(f"  SSIM: {image_loss['ssim']:.4f}")

    return best_candidate, history
