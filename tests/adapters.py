from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    combined_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]

    max_combined_len = max(len(c) for c in combined_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Right-pad all combined sequences to max_combined_len
    padded = torch.full((len(combined_list), max_combined_len), pad_id, dtype=torch.long)
    for i, combined in enumerate(combined_list):
        padded[i, :len(combined)] = torch.tensor(combined, dtype=torch.long)

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]

    # Response tokens in labels start at (prompt_len - 1) and span output_len positions
    max_len = max_combined_len - 1
    response_mask = torch.zeros((len(combined_list), max_len), dtype=torch.bool)
    for i, (prompt_ids, output_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        start = len(prompt_ids) - 1
        response_mask[i, start : start + len(output_ids)] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    raw_rewards = []
    for r, t in zip(rollout_responses, repeated_ground_truths):
        raw_rewards.append(reward_fn(r, t)["reward"])
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    n_groups = len(rollout_responses) // group_size

    grouped = raw_rewards.view(n_groups, group_size)

    mean = grouped.mean(dim=1, keepdim=True)
    normalized_rewards = grouped - mean 
    metadata = {"max": raw_rewards.max().item()}
    if normalize_by_std:
        std = grouped.std(dim=1, keepdim=True) + advantage_eps
        normalized_rewards = normalized_rewards / std 
    return tuple((normalized_rewards.view(-1), raw_rewards, metadata))


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits  # (batch, seq, vocab)
    log_probs_all = torch.log_softmax(logits, dim=-1)
    # Gather the log-prob of each actual next token from labels
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token_entropy = run_compute_entropy(logits) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}



def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return - raw_rewards_or_advantages * policy_log_probs


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    reweight = torch.exp(policy_log_probs)/torch.exp(old_log_probs)
    loss = -1 * torch.min(reweight * advantages, torch.clamp(reweight, min=1-cliprange, max=1+cliprange)*advantages)
    metadata = {"clip_rate": ((reweight > 1+cliprange) | (reweight < 1-cliprange)).float().mean().item()}
    return tuple((loss, metadata))


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    assert loss_type in ("no_baseline", "reinforce_with_baseline", "grpo_clip")
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {"max": raw_rewards.max()}
        return tuple((loss, metadata))
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"max": advantages.max()}
        return tuple((loss, metadata))
    else:
        assert advantages is not None
        loss, metadata = run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        return tuple((loss, metadata))




def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    tensor = tensor * mask
    if dim is not None:
        output = tensor.sum(dim=dim) / mask.sum(dim=dim)
    else:
        output = tensor.sum() / mask.sum()
    return output

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    loss = run_masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant)
    loss = -loss.mean() / gradient_accumulation_steps
    loss.backward()
    return (loss, {"loss": loss.detach()})

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    
    loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs, 
        loss_type, 
        raw_rewards, 
        advantages, 
        old_log_probs, 
        cliprange
    )
    loss = run_masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return tuple((loss, metadata))


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
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
    masked = tensor * mask
    return masked.sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling.
    """
    import json
    import random
    import pathlib

    template_path = pathlib.Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "alpaca_sft.prompt"
    template = template_path.read_text()

    # Load JSONL data
    dataset_path = str(dataset_path)
    docs = []
    if dataset_path.endswith('.gz'):
        import gzip
        with gzip.open(dataset_path, 'rt') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    else:
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

    if shuffle:
        random.shuffle(docs)

    # Tokenize each document with Alpaca template, BOS prefix, and EOS suffix
    all_tokens = []
    for doc in docs:
        text = template.replace('{instruction}', doc['prompt']).replace('{response}', doc['response'])
        text = text.rstrip('\n')
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens.append(tokenizer.eos_token_id)
        all_tokens.extend(tokens)

    # Create non-overlapping chunks: input_ids[i] = all_tokens[i*seq_length : (i+1)*seq_length]
    # labels[i] = all_tokens[i*seq_length+1 : (i+1)*seq_length+1]
    num_examples = (len(all_tokens) - 1) // seq_length

    class PackedSFTDataset(Dataset):
        def __init__(self, tokens, seq_length, num_examples):
            self.tokens = tokens
            self.seq_length = seq_length
            self._num_examples = num_examples

        def __len__(self):
            return self._num_examples

        def __getitem__(self, i):
            if i < 0 or i >= self._num_examples:
                raise IndexError(f"index {i} out of range for dataset of size {self._num_examples}")
            start = i * self.seq_length
            input_ids = torch.tensor(self.tokens[start : start + self.seq_length], dtype=torch.long)
            labels = torch.tensor(self.tokens[start + 1 : start + self.seq_length + 1], dtype=torch.long)
            return {"input_ids": input_ids, "labels": labels}

    return PackedSFTDataset(all_tokens, seq_length, num_examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.
    """
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.
    """
    import re
    # Look for patterns like "The correct answer is X" or just a standalone letter
    match = re.search(r'(?:The correct answer is|the answer is)\s+([A-D])\b', model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Look for standalone letter at the start
    match = re.search(r'^([A-D])\b', model_output.strip())
    if match:
        return match.group(1).upper()
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.
    """
    import re
    # Find all numbers (integers and decimals) in the output
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', model_output)
    if not numbers:
        return None
    # Take the last number, remove commas
    return numbers[-1].replace(',', '')


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Compute the per-instance DPO loss.
    Uses the Alpaca template and adds EOS after the response.
    Uses the unconditional log-prob simplification (prompt cancels out).
    """
    import pathlib
    import torch.nn.functional as F

    template_path = pathlib.Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "alpaca_sft.prompt"
    template = template_path.read_text()

    eos_token = tokenizer.eos_token if tokenizer.eos_token else ""

    # Format with Alpaca template (strip trailing newline) + EOS
    chosen_text = template.replace('{instruction}', prompt).replace('{response}', response_chosen).rstrip('\n') + eos_token
    rejected_text = template.replace('{instruction}', prompt).replace('{response}', response_rejected).rstrip('\n') + eos_token

    def get_sequence_log_prob(model, text):
        device = next(model.parameters()).device
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
        labels = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum()

    lm_device = next(lm.parameters()).device

    # Compute log-probs under both models for both responses
    log_prob_chosen_policy = get_sequence_log_prob(lm, chosen_text)
    log_prob_rejected_policy = get_sequence_log_prob(lm, rejected_text)
    log_prob_chosen_ref = get_sequence_log_prob(lm_ref, chosen_text).to(lm_device)
    log_prob_rejected_ref = get_sequence_log_prob(lm_ref, rejected_text).to(lm_device)

    # DPO loss: -log σ(β * ((log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x))))
    # Using unconditional log-probs (prompt cancels in the difference)
    log_ratio_chosen = log_prob_chosen_policy - log_prob_chosen_ref
    log_ratio_rejected = log_prob_rejected_policy - log_prob_rejected_ref

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss
