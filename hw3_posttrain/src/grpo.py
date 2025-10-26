"""
GRPO (Group Relative Policy Optimization) Implementation

This module implements GRPO for training language models on GSM8K with verified rewards.
Includes all the core functions for computing advantages, policy gradient losses, and training.

Problems 3-8 implementation from the homework.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from data.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn


# ==================== Problem 3: Group Normalization ====================

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute group-normalized rewards (advantages) for GRPO.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses
        rollout_responses: list[str] Rollouts from the policy
        repeated_ground_truths: list[str] Ground truths for the examples
        group_size: int Number of responses per question (group)
        advantage_eps: float Small constant to avoid division by zero
        normalize_by_std: bool If True, divide by per-group standard deviation

    Returns:
        advantages: torch.Tensor shape (rollout_batch_size,) Group-normalized rewards
        raw_rewards: torch.Tensor shape (rollout_batch_size,) Unnormalized rewards
        metadata: dict[str, float] Statistics to log
    """
    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size == len(repeated_ground_truths)
    assert rollout_batch_size % group_size == 0

    # Compute raw rewards for all rollout responses
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, ground_truth)
        raw_rewards_list.append(scores.get("reward", 0.0))
        format_rewards_list.append(scores.get("format_reward", 0.0))
        answer_rewards_list.append(scores.get("answer_reward", 0.0))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # Normalize within groups
    n_groups = rollout_batch_size // group_size
    advantages = torch.zeros_like(raw_rewards)

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = raw_rewards[start_idx:end_idx]

        # Compute group mean
        group_mean = group_rewards.mean()

        if normalize_by_std:
            # Compute group std
            group_std = group_rewards.std()
            # Normalize: (r - mean) / (std + eps)
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / (group_std + advantage_eps)
        else:
            # Just subtract mean (Dr. GRPO approach)
            advantages[start_idx:end_idx] = group_rewards - group_mean

    # Compute metadata
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "mean_format_reward": sum(format_rewards_list) / len(format_rewards_list),
        "mean_answer_reward": sum(answer_rewards_list) / len(answer_rewards_list),
    }

    return advantages, raw_rewards, metadata


# ==================== Problem 4: Naive Policy Gradient Loss ====================

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the per-token policy-gradient loss using raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: shape (batch_size, 1) or (batch_size,)
        policy_log_probs: shape (batch_size, sequence_length)

    Returns:
        loss: shape (batch_size, sequence_length) per-token loss
    """
    # Ensure advantages have correct shape for broadcasting
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)

    # Broadcast advantages across sequence length and compute loss
    # Loss = -A * log_prob
    loss = -raw_rewards_or_advantages * policy_log_probs

    return loss


# ==================== Problem 5: GRPO-Clip Loss ====================

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the per-token GRPO-Clip loss.

    Args:
        advantages: shape (batch_size, 1) or (batch_size,)
        policy_log_probs: shape (batch_size, sequence_length)
        old_log_probs: shape (batch_size, sequence_length)
        cliprange: float Clip parameter epsilon

    Returns:
        loss: shape (batch_size, sequence_length) per-token clipped loss
        metadata: dict with statistics (e.g., clip fraction)
    """
    # Ensure advantages have correct shape for broadcasting
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    # Compute probability ratios: pi_theta / pi_old
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Compute the two terms in the min
    # Term 1: ratio * A
    term1 = ratio * advantages

    # Term 2: clip(ratio, 1-eps, 1+eps) * A
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    term2 = clipped_ratio * advantages

    # Take the minimum (worst case) and negate for loss
    loss = -torch.min(term1, term2)

    # Compute metadata: fraction of tokens that were clipped
    clipped = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))
    clip_fraction = clipped.float().mean()

    metadata = {
        "clip_fraction": clip_fraction,
        "mean_ratio": ratio.mean(),
        "mean_log_ratio": log_ratio.mean(),
    }

    return loss, metadata


# ==================== Problem 6: Policy Gradient Loss Wrapper ====================

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs: shape (batch_size, sequence_length)
        loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip"
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1)
        advantages: Required for "reinforce_with_baseline" and "grpo_clip"
        old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length)
        cliprange: Required for "grpo_clip"; scalar epsilon

    Returns:
        loss: shape (batch_size, sequence_length) per-token loss
        metadata: dict of statistics
    """
    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"
        loss, clip_metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        metadata.update(clip_metadata)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata


# ==================== Problem 7: Masked Mean ====================

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a dimension, considering only masked elements.

    Args:
        tensor: The data to be averaged
        mask: Same shape as tensor; positions with 1 are included
        dim: Dimension to average over; if None, compute mean over all masked elements

    Returns:
        The masked mean
    """
    masked_tensor = tensor * mask

    if dim is None:
        # Mean over all masked elements
        total = masked_tensor.sum()
        count = mask.sum().clamp(min=1)
        return total / count
    else:
        # Mean over specified dimension
        total = masked_tensor.sum(dim=dim)
        count = mask.sum(dim=dim).clamp(min=1)
        return total / count


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    constant_normalizer: float = 1.0,
) -> torch.Tensor:
    """
    Sum over masked elements and divide by a constant normalizer.

    Args:
        tensor: The data to normalize
        mask: Same shape as tensor; positions with 1 are included
        dim: Dimension to sum over; if None, sum over all masked elements
        constant_normalizer: Constant to divide by

    Returns:
        The normalized tensor
    """
    masked_tensor = tensor * mask

    if dim is None:
        # Sum over all masked elements
        total = masked_tensor.sum()
        return total / max(constant_normalizer, 1e-8)
    else:
        # Sum over specified dimension
        total = masked_tensor.sum(dim=dim)
        return total / max(constant_normalizer, 1e-8)


# ==================== Problem 8: GRPO Microbatch Train Step ====================

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
    use_length_normalization: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs: (batch_size, sequence_length)
        response_mask: (batch_size, sequence_length) 1 for response tokens, 0 otherwise
        gradient_accumulation_steps: Number of microbatches per optimizer step
        loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip"
        raw_rewards: Needed when loss_type == "no_baseline"
        advantages: Needed when loss_type != "no_baseline"
        old_log_probs: Required for GRPO-Clip
        cliprange: Clip parameter for GRPO-Clip
        use_length_normalization: If True, use masked_mean; else use masked_normalize

    Returns:
        loss: Scalar tensor (microbatch loss adjusted for gradient accumulation)
        metadata: Dict with statistics
    """
    # Compute per-token loss
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Aggregate to scalar per example, then to scalar over batch
    if use_length_normalization:
        # Normalize by sequence length (masked_mean)
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    else:
        # Sum over sequence, divide by constant (max sequence length)
        max_seq_len = response_mask.sum(dim=1).max().item()
        per_example_loss = masked_normalize(
            per_token_loss, response_mask, dim=1, constant_normalizer=max_seq_len
        )

    # Average over batch
    loss = per_example_loss.mean()

    # Adjust for gradient accumulation
    loss = loss / gradient_accumulation_steps

    # Backward pass
    loss.backward()

    # Add loss to metadata
    metadata["loss"] = loss.detach()

    return loss.detach(), metadata


# ==================== Helper Functions ====================

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer,
    max_seq_len: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompts and outputs separately, then concatenate.
    Returns input_ids, labels, and response_mask.
    """
    # Tokenize separately to get prompt length for masking
    tokenized_prompts = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )
    tokenized_outputs = tokenizer(
        output_strs,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )

    # Concatenate prompt and output
    input_ids = []
    for i in range(len(prompt_strs)):
        ids = torch.cat([
            tokenized_prompts.input_ids[i],
            tokenized_outputs.input_ids[i]
        ], dim=0)
        input_ids.append(ids)

    # Pad to same length
    max_len = min(max([x.size(0) for x in input_ids]), max_seq_len)
    padded = torch.full(
        (len(input_ids), max_len),
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        dtype=torch.long
    )
    for i, ids in enumerate(input_ids):
        length = min(ids.size(0), max_len)
        padded[i, :length] = ids[:length]

    # Create response mask: 1 on response positions, else 0
    response_mask = torch.zeros_like(padded)
    for i in range(len(prompt_strs)):
        prompt_len = tokenized_prompts.input_ids[i].size(0)
        total_len = min(
            tokenized_prompts.input_ids[i].size(0) + tokenized_outputs.input_ids[i].size(0),
            max_len
        )
        response_mask[i, prompt_len:total_len] = 1

    # Shift for causal language modeling: predict next token
    input_ids = padded[:, :-1].contiguous()
    labels = padded[:, 1:].contiguous()
    response_mask = response_mask[:, 1:].contiguous()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


@torch.no_grad()
def get_response_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute log probabilities of labels given inputs.

    Args:
        model: The language model
        input_ids: shape (batch_size, sequence_length)
        labels: shape (batch_size, sequence_length)
        return_token_entropy: If True, also return per-token entropy

    Returns:
        dict with "log_probs" and optionally "token_entropy"
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch, seq, vocab)
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log-probs for the actual labels
    label_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": label_log_probs}

    if return_token_entropy:
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        result["token_entropy"] = entropy

    return result


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vllm_generate(
    llm: LLM,
    prompts: List[str],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: List[str],
    min_tokens: int = 4,
    logprobs: Optional[int] = None,
) -> List[List[str]]:
    """
    Generate N completions per prompt using vLLM.

    Returns:
        List of lists, where each inner list contains N completions for one prompt
    """
    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        min_tokens=min_tokens,
        logprobs=logprobs,
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, params)

    all_texts = []
    for out in outputs:
        texts = [c.text for c in out.outputs]
        all_texts.append(texts)

    return all_texts


# ==================== Data Loading ====================

class GSM8KDataset:
    """Dataset loader for GSM8K."""

    def __init__(self, jsonl_path: Path, prompt_template: str):
        self.examples: List[Tuple[str, str]] = []  # (prompt, gold_answer)
        with jsonl_path.open("r") as f:
            for line in f:
                obj = json.loads(line)
                question = obj["question"]
                gold = obj["answer"].split("####")[-1].strip()
                prompt = prompt_template.replace("{question}", question)
                self.examples.append((prompt, gold))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.examples[idx]


def load_prompt_template(path: Path) -> str:
    """Load prompt template from file."""
    return path.read_text().strip()


if __name__ == "__main__":
    # Simple test
    print("GRPO utilities module loaded successfully!")
    print("Functions available:")
    print("  - compute_group_normalized_rewards (Problem 3)")
    print("  - compute_naive_policy_gradient_loss (Problem 4)")
    print("  - compute_grpo_clip_loss (Problem 5)")
    print("  - compute_policy_gradient_loss (Problem 6)")
    print("  - masked_mean, masked_normalize (Problem 7)")
    print("  - grpo_microbatch_train_step (Problem 8)")
