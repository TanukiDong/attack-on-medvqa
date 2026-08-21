import torch
import torch.nn.functional as F

from common.model import VALID_ANSWERS


def get_choice_logits(selected_logits, answer_token_ids):
    """Get the logits corresponding to the answer choices (A, B, C, D)."""
    if selected_logits.shape[0] != 1:
        raise RuntimeError(f"Expected exactly one answer position, but got {selected_logits.shape[0]} positions.")
    
    # Extract probabilities belonging to A-D
    choice_token_ids = get_choice_token_ids(answer_token_ids, device=selected_logits.device)
    return selected_logits[0, choice_token_ids].float()

def get_vocab_choice_probs(selected_logits, answer_token_ids):
    """Get next-token probabilities of A-D relative to the entire model vocabulary."""
    if selected_logits.shape[0] != 1:
        raise RuntimeError(
            f"Expected exactly one answer position, but got {selected_logits.shape[0]} positions.")

    # Softmax over the entire vocabulary
    vocab_probs = F.softmax(selected_logits[0].float(), dim=0)

    # Extract probabilities belonging to A-D
    choice_token_ids = get_choice_token_ids(answer_token_ids, device=selected_logits.device)

    return vocab_probs[choice_token_ids]

def get_choice_token_ids(answer_token_ids, device):
    """Get the token IDs corresponding to the answer choices (A, B, C, D)."""
    return torch.tensor(
        [answer_token_ids[x] for x in VALID_ANSWERS],
        device=device,
        dtype=torch.long)

def get_clean_probs(clean_logits, loss_scope, answer_token_ids):
    """Get the clean reference probabilities for KL divergence."""
    if loss_scope == "answer":
        clean_logits = get_choice_logits(
            clean_logits,
            answer_token_ids,
        )
    elif loss_scope != "full_output":
        raise ValueError(f"Unsupported loss_scope for KL loss: {loss_scope}")

    return F.softmax(clean_logits.float(), dim=-1).detach()

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

def choice_entropy_loss(logits, answer_token_ids):
    """Compute entropy over the answer choices (A, B, C, D)."""
    choice_logits = get_choice_logits(logits, answer_token_ids)

    choice_probs = F.softmax(choice_logits, dim=0)
    choice_log_probs = F.log_softmax(choice_logits, dim=0)

    # -Σ(p * log(p)) 
    entropy = -(choice_probs * choice_log_probs).sum()

    return entropy

def compute_loss(selected_logits, selected_labels, target, answer_token_ids, attack_config, clean_probs=None):
    """Compute the loss based on the selected logits and labels."""
    loss_type = attack_config["loss_type"]
    loss_scope = attack_config["loss_scope"]

    if loss_type == "cross_entropy":
        if loss_scope == "answer":
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
                f"Unsupported loss_scope for cross_entropy loss: {loss_scope}"
            )

    elif loss_type == "entropy":
        if loss_scope == "answer":
            loss = choice_entropy_loss(
                logits=selected_logits,
                answer_token_ids=answer_token_ids,
            )
        else:
            raise ValueError(f"Unsupported loss_scope for entropy loss: {loss_scope}")

    elif loss_type == "kl":
        if clean_probs is None:
            raise ValueError("KL loss requires clean_probs.")

        if loss_scope == "answer":
            adv_choice_logits = get_choice_logits(
                selected_logits,
                answer_token_ids,
            )
            adv_log_probs = F.log_softmax(adv_choice_logits, dim=-1)

            loss = F.kl_div(
                input=adv_log_probs.unsqueeze(0),
                target=clean_probs.unsqueeze(0),
                reduction="batchmean", # See https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.kl_div.html
            )
            
        elif loss_scope == "full_output":
            adv_log_probs = F.log_softmax(selected_logits.float(), dim=-1)

            loss = F.kl_div(
                input=adv_log_probs,
                target=clean_probs,
                reduction="batchmean",
            )
            
        else:
            raise ValueError(f"Unsupported loss_scope for KL loss: {loss_scope}")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    if not torch.isfinite(loss):
        raise RuntimeError(f"Attack loss is not finite: {loss.item()}")
    if not loss.requires_grad:
        raise RuntimeError("The loss has no gradient.")

    if loss_scope == "answer":
        choice_probs = get_vocab_choice_probs(selected_logits, answer_token_ids)
    else:
        choice_probs = None

    return loss, choice_probs
