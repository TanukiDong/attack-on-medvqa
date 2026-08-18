import torch
import torch.nn.functional as F

from common.model import VALID_ANSWERS


def get_choice_logits(selected_logits, answer_token_ids):
    """Get the logits corresponding to the answer choices (A, B, C, D)."""
    if selected_logits.shape[0] != 1:
        raise RuntimeError(f"Expected exactly one answer position, but got {selected_logits.shape[0]} positions.")
    choice_token_ids = torch.tensor(
        [answer_token_ids[x] for x in VALID_ANSWERS],
        device=selected_logits.device,)
    return selected_logits[0, choice_token_ids].float()


def get_choice_probs(selected_logits, answer_token_ids):
    """Get the probabilities corresponding to the answer choices (A, B, C, D)."""
    choice_logits = get_choice_logits(selected_logits, answer_token_ids)
    return torch.softmax(choice_logits, dim=0)


def choice_ce_loss(logits, target, answer_token_ids):
    """Compute cross-entropy loss for the answer choices (A, B, C, D)."""
    if logits.shape[0] != 1:
        raise RuntimeError(f"choice_ce_loss expects exactly one answer position but got {logits.shape[0]} positions.")

    # Select only the answer choices logits
    choice_logits = get_choice_logits(logits, answer_token_ids)
    
    # Convert letter to index
    target_index = torch.tensor(
        [VALID_ANSWERS.index(target)],
        device=choice_logits.device,
        dtype=torch.long,
    )

    ce_loss = F.cross_entropy(choice_logits.unsqueeze(0), target_index)

    return ce_loss


def compute_loss(selected_logits, selected_labels, target, answer_token_ids, attack_config, clean_probs=None):
    """Compute the loss based on the selected logits and labels."""
    loss_type = attack_config["loss_type"]
    loss_scope = attack_config["loss_scope"]

    if loss_type == "cross_entropy":
        if loss_scope == "vocab_answer":
            loss = F.cross_entropy(
                input=selected_logits.float(),
                target=selected_labels,
            )
        elif loss_scope in {"choice_answer", "conditioned_answer"}:
            loss = choice_ce_loss(
                logits=selected_logits,
                target=target,
                answer_token_ids=answer_token_ids,
            )
        elif loss_scope == "full_output":
            loss = F.cross_entropy(
                input=selected_logits.float(),
                target=selected_labels,
            )
        else:
            raise ValueError(
                f"Unsupported loss_scope for cross_entropy: {loss_scope}"
            )

    elif loss_type == "kl":
        if clean_probs is None:
            raise ValueError("KL loss requires clean_probs.")

        if loss_scope == "choice_answer":
            adv_choice_logits = get_choice_logits(
                selected_logits,
                answer_token_ids,
            )
            adv_log_probs = F.log_softmax(adv_choice_logits, dim=0)

            loss = F.kl_div(
                input=adv_log_probs.unsqueeze(0),
                target=clean_probs.unsqueeze(0),
                reduction="batchmean", # See https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.kl_div.html
            )
        else:
            raise ValueError(f"Unsupported loss_scope for KL: {loss_scope}")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    if not torch.isfinite(loss):
        raise RuntimeError(f"Attack loss is not finite: {loss.item()}")
    if not loss.requires_grad:
        raise RuntimeError("The loss has no gradient.")

    if loss_scope in {"choice_answer", "conditioned_answer"}:
        choice_probs = get_choice_probs(selected_logits, answer_token_ids)
    else:
        choice_probs = None

    return loss, choice_probs
