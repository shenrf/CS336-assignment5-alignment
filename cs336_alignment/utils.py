"""Shared utilities for SFT, Expert Iteration, and GRPO trainers."""

import json
import logging
import random
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Start a vLLM inference engine on the given device."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """Copy policy weights into a running vLLM instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    """Load data from JSONL (one JSON object per line) or JSON array file."""
    data = []
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.loads(f.read())
        else:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples from {path}")
    return data


def format_sft_examples(
    data: list[dict],
    prompt_template: str,
) -> tuple[list[str], list[str]]:
    """
    Format data into (prompt_string, output_string) pairs.

    Supports multiple data formats:
      - GSM8K: keys "question" and "answer", template uses {question}.
      - Instruction tuning: keys "prompt" and "response",
        template uses {instruction} (response is the output).
      - Math reasoning: keys "problem" and "reasoning_trace",
        template uses {question}.
    """
    prompts = []
    outputs = []
    for ex in data:
        if "question" in ex and "answer" in ex:
            prompts.append(prompt_template.format(question=ex["question"]))
            outputs.append(ex["answer"])
        elif "prompt" in ex:
            prompts.append(prompt_template.format(instruction=ex["prompt"], response=""))
            outputs.append(ex["response"])
        else:
            prompts.append(prompt_template.format(question=ex["problem"]))
            outputs.append(ex["reasoning_trace"])
    return prompts, outputs


def get_batches(
    prompts: list[str],
    outputs: list[str],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
):
    """Yield (prompt_batch, output_batch) tuples for one epoch."""
    indices = list(range(len(prompts)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield (
            [prompts[i] for i in batch_idx],
            [outputs[i] for i in batch_idx],
        )


# ---------------------------------------------------------------------------
# Core compute helpers
# ---------------------------------------------------------------------------

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 2048,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt+response pairs and build response mask."""
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    combined_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    if max_seq_len is not None:
        combined_list = [c[:max_seq_len] for c in combined_list]
    max_combined_len = max(len(c) for c in combined_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded = torch.full((len(combined_list), max_combined_len), pad_id, dtype=torch.long)
    for i, combined in enumerate(combined_list):
        padded[i, : len(combined)] = torch.tensor(combined, dtype=torch.long)

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]

    max_len = max_combined_len - 1
    response_mask = torch.zeros((len(combined_list), max_len), dtype=torch.bool)
    for i, (prompt_ids, output_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        start = len(prompt_ids) - 1
        response_mask[i, start : start + len(output_ids)] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Forward pass returning per-token log-probs and optionally entropy."""
    logits = model(input_ids).logits
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token_entropy = None
    if return_token_entropy:
        probs = torch.exp(log_probs_all)
        token_entropy = -(probs * log_probs_all).sum(dim=-1)
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over masked elements along a dimension and divide by a constant."""
    masked = tensor * mask
    return masked.sum(dim=dim) / normalize_constant


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean over masked elements along a dimension."""
    tensor = tensor * mask
    if dim is not None:
        return tensor.sum(dim=dim) / mask.sum(dim=dim)
    return tensor.sum() / mask.sum()


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards and group-normalize them (GRPO advantage computation)."""
    raw_rewards = []
    for r, t in zip(rollout_responses, repeated_ground_truths):
        raw_rewards.append(reward_fn(r, t)["reward"])
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    n_groups = len(rollout_responses) // group_size

    grouped = raw_rewards.view(n_groups, group_size)
    mean_r = grouped.mean(dim=1, keepdim=True)
    normalized_rewards = grouped - mean_r
    metadata = {"max": raw_rewards.max().item()}
    if normalize_by_std:
        std = grouped.std(dim=1, keepdim=True) + advantage_eps
        normalized_rewards = normalized_rewards / std
    return normalized_rewards.view(-1), raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Basic policy gradient loss: -reward * log_prob."""
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped policy gradient loss (PPO/GRPO-style)."""
    reweight = torch.exp(policy_log_probs) / torch.exp(old_log_probs)
    loss = -torch.min(
        reweight * advantages,
        torch.clamp(reweight, min=1 - cliprange, max=1 + cliprange) * advantages,
    )
    metadata = {
        "clip_rate": ((reweight > 1 + cliprange) | (reweight < 1 - cliprange)).float().mean().item()
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dispatcher for policy gradient loss types."""
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss and call backward for one microbatch."""
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SFT loss and call backward for one microbatch."""
    loss = masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant)
    loss = -loss.mean() / gradient_accumulation_steps
    loss.backward()
    return (loss, {"loss": loss.detach()})
