import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False, padding=False)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False, padding=False)["input_ids"]
    
    prompt_len = torch.tensor([len(r) for r in prompt_ids], dtype=torch.long)
    output_len = torch.tensor([len(r) for r in output_ids], dtype=torch.long)
    
    seq_len = prompt_len + output_len
    max_seq_len = int(seq_len.max())
    
    inputs = {"input_ids": [], "labels": [], "response_mask": []}
    for prompt, output in zip(prompt_ids, output_ids):
        combined = prompt + output
        
        input_ids = combined + [tokenizer.pad_token_id] * (max_seq_len - len(combined))
        response_mask = [0] * len(prompt) + [1] * len(output) + [0] * (max_seq_len - len(combined))
        
        inputs["input_ids"].append(input_ids[:-1])
        inputs["labels"].append(input_ids[1:])
        inputs["response_mask"].append(response_mask[1:])
    
    inputs["input_ids"] = torch.tensor(inputs["input_ids"], dtype=torch.long)
    inputs["labels"] = torch.tensor(inputs["labels"], dtype=torch.long)
    inputs["response_mask"] = torch.tensor(inputs["response_mask"], dtype=torch.bool)
    
    return inputs

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of a categorical distribution given its logits.
    
    Args:
        logits: Tensor of shape (..., vocab_size)
        
    Returns:
        Entropy tensor of shape (...)
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get log probabilities of the response tokens in `labels`.
    
    Args:
        model: The language model
        input_ids: Tensor of shape (batch_size, seq_len)
        labels: Tensor of shape (batch_size, seq_len)
        return_token_entropy: Whether to return token-wise entropy
    Returns:
        Dictionary with keys:
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log pθ (xt | x<t).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    """
    
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    
    log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    
    # Gather log probabilities of the labels
    log_probs_gathered = torch.gather(
        log_probs, -1, labels.unsqueeze(-1)
    ).squeeze(-1)  # (batch_size, seq_len)
    
    result = {"log_probs": log_probs_gathered}
    
    if return_token_entropy:
        token_entropy = compute_entropy(logits)  # (batch_size, seq_len)
        result["token_entropy"] = token_entropy
    
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None,
    normalize_constant: float,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    
    masked_tensor = tensor * mask
    summed = masked_tensor.sum(dim)
    normalized = summed / normalize_constant
    return normalized

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Sequence-average negative-log-likelihood over the response tokens,
    scaled by `gradient_accumulation_steps` *inside* the helper (the tests
    expect that) and by any extra `normalize_constant`.

    Args
    ----
    policy_log_probs : (B, L) tensor
        Per-token log-probs from the policy.
    response_mask    : (B, L) tensor, 1 on response (incl. <eos>), 0 elsewhere.
    gradient_accumulation_steps : int
        Number of micro-batches per optimiser step; the loss is divided by it
        here so every call reports the scaled value used in the snapshots.
    normalize_constant : float, default 1.0
        Additional divisor (the tests pass 42.0 for the normalisation case).

    Returns
    -------
    loss : scalar tensor
        Micro-batch loss already scaled as above.
    metadata : dict
        Contains the same scalar for convenient logging.
    """
    # mask out prompt / pad tokens
    masked_log_probs = policy_log_probs * response_mask.float()

    # joint log-probability over all response tokens in the micro-batch
    sum_log_probs = masked_log_probs.sum()

    batch_size = policy_log_probs.size(0)

    # sequence-level NLL averaged over sequences,
    # then scaled for grad-acc and any extra normalisation
    loss = -sum_log_probs / (batch_size *
                             gradient_accumulation_steps *
                             normalize_constant)

    loss.backward()

    return loss, {"microbatch_loss": loss.detach()}
