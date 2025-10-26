"""
GRPO Training Script (Problems 9-16)

This script implements the complete GRPO training loop with support for:
- Different loss types (no_baseline, reinforce_with_baseline, grpo_clip)
- On-policy and off-policy training
- Length normalization ablations
- Standard deviation normalization ablations
- Different prompts (r1_zero, question_only)

Usage:
    python grpo_train.py --loss_type reinforce_with_baseline --n_grpo_steps 200
"""

import json
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional
from datetime import datetime

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Import GRPO utilities from grpo.py
from grpo import (
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    masked_mean,
    masked_normalize,
    grpo_microbatch_train_step,
    tokenize_prompt_and_output,
    get_response_log_probs,
    set_seed,
    vllm_generate,
    GSM8KDataset,
    load_prompt_template,
)
from data.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model and data
    model_name: str = "Qwen/Qwen2.5-0.6B"
    train_path: Path = Path("./data/gsm8k/train.jsonl")
    val_path: Path = Path("./data/gsm8k/test.jsonl")
    prompt_path: Path = Path("./data/r1_zero.prompt")
    results_dir: Path = Path("results/grpo")

    # GRPO hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024

    # Training configuration
    epochs_per_rollout_batch: int = 1  # On-policy by default
    train_batch_size: int = 256  # On-policy by default
    gradient_accumulation_steps: int = 128  # microbatch size = 2
    grad_clip: float = 1.0
    max_seq_len: int = 2048

    # Loss configuration
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    use_length_normalization: bool = True
    cliprange: float = 0.2  # For GRPO-Clip

    # Prompt configuration
    prompt_type: Literal["r1_zero", "question_only"] = "r1_zero"

    # Logging and evaluation
    eval_every_steps: int = 5
    eval_batch_size: int = 1024
    save_every_steps: int = 50
    log_every_steps: int = 1

    # System
    seed: int = 42
    gpu_memory_utilization: float = 0.85

    def __post_init__(self):
        # Validation
        assert self.train_batch_size % self.gradient_accumulation_steps == 0
        assert self.rollout_batch_size % self.group_size == 0
        assert self.train_batch_size >= self.group_size

        # Computed values
        self.micro_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size

        # Set prompt and reward function based on prompt_type
        if self.prompt_type == "question_only":
            self.prompt_path = Path("./data/question_only.prompt")
            self.reward_fn = question_only_reward_fn
        else:
            self.prompt_path = Path("./data/r1_zero.prompt")
            self.reward_fn = r1_zero_reward_fn


class GRPOTrainer:
    """GRPO Trainer implementing the full training loop."""

    def __init__(self, config: GRPOConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config.loss_type}_std{config.use_std_normalization}_len{config.use_length_normalization}"
        self.run_dir = config.results_dir / f"{exp_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with (self.run_dir / "config.json").open('w') as f:
            json.dump(asdict(config), f, indent=2, default=str)

        # Load prompt template and datasets
        self.prompt_template = load_prompt_template(config.prompt_path)
        self.train_ds = GSM8KDataset(config.train_path, self.prompt_template)
        self.val_ds = GSM8KDataset(config.val_path, self.prompt_template)

        print(f"Loaded {len(self.train_ds)} training examples")
        print(f"Loaded {len(self.val_ds)} validation examples")

        # Initialize vLLM for generation
        print("Initializing vLLM for generation...")
        self.llm = LLM(
            model=config.model_name,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )

        # Initialize torch model for training
        print("Initializing torch model for training...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        # Training state
        self.global_step = 0
        self.training_logs = []

    def sample_train_prompts(self, n_prompts: int) -> List[tuple]:
        """Sample random prompts from training set."""
        indices = random.sample(range(len(self.train_ds)), k=min(n_prompts, len(self.train_ds)))
        return [self.train_ds[i] for i in indices]

    def generate_rollouts(self, prompts: List[str], n_samples: int) -> List[str]:
        """Generate rollouts using vLLM."""
        # Determine stop tokens based on prompt type
        if self.cfg.prompt_type == "question_only":
            stop_tokens = ["\n\n", self.tokenizer.eos_token]
        else:
            stop_tokens = ["</answer>"]

        all_completions = vllm_generate(
            self.llm,
            prompts,
            n=n_samples,
            temperature=self.cfg.sampling_temperature,
            top_p=1.0,
            max_tokens=self.cfg.sampling_max_tokens,
            stop=stop_tokens,
            min_tokens=self.cfg.sampling_min_tokens,
        )

        # Flatten: n_prompts * group_size completions
        rollouts = []
        for completions_per_prompt in all_completions:
            rollouts.extend(completions_per_prompt)

        return rollouts

    @torch.no_grad()
    def evaluate(self, n_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate on validation set."""
        if n_samples is None:
            n_samples = min(self.cfg.eval_batch_size, len(self.val_ds))

        # Sample validation examples
        val_indices = random.sample(range(len(self.val_ds)), k=n_samples)
        val_prompts = [self.val_ds[i][0] for i in val_indices]
        val_golds = [self.val_ds[i][1] for i in val_indices]

        # Generate single completion per prompt
        completions = self.generate_rollouts(val_prompts, n_samples=1)

        # Compute rewards
        total_reward = 0.0
        total_format = 0.0
        total_answer = 0.0

        for completion, gold in zip(completions, val_golds):
            scores = self.cfg.reward_fn(completion, gold)
            total_reward += scores.get("reward", 0.0)
            total_format += scores.get("format_reward", 0.0)
            total_answer += scores.get("answer_reward", 0.0)

        return {
            "reward": total_reward / n_samples,
            "format_reward": total_format / n_samples,
            "answer_reward": total_answer / n_samples,
        }

    def train_step(self) -> Dict[str, float]:
        """Execute one GRPO training step."""
        step_metrics = {}

        # 1. Sample prompts and generate rollouts
        sampled_examples = self.sample_train_prompts(self.cfg.n_prompts_per_rollout_batch)
        prompts = [ex[0] for ex in sampled_examples]
        golds = [ex[1] for ex in sampled_examples]

        print(f"  Generating {self.cfg.rollout_batch_size} rollouts...")
        rollouts = self.generate_rollouts(prompts, n_samples=self.cfg.group_size)

        # 2. Compute rewards and advantages
        repeated_golds = [gold for gold in golds for _ in range(self.cfg.group_size)]

        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=self.cfg.reward_fn,
            rollout_responses=rollouts,
            repeated_ground_truths=repeated_golds,
            group_size=self.cfg.group_size,
            advantage_eps=self.cfg.advantage_eps,
            normalize_by_std=self.cfg.use_std_normalization,
        )

        step_metrics.update({f"train/{k}": v for k, v in reward_metadata.items()})

        # 3. Tokenize rollouts
        repeated_prompts = [prompt for prompt in prompts for _ in range(self.cfg.group_size)]
        tokenized = tokenize_prompt_and_output(
            repeated_prompts,
            rollouts,
            self.tokenizer,
            max_seq_len=self.cfg.max_seq_len,
        )

        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        response_mask = tokenized["response_mask"]

        # 4. Get old log probs if doing off-policy or GRPO-Clip
        old_log_probs = None
        if self.cfg.loss_type == "grpo_clip" or self.cfg.epochs_per_rollout_batch > 1:
            print(f"  Computing old log probs...")
            self.model.eval()
            old_log_probs_list = []
            for i in range(0, len(input_ids), 32):  # Process in smaller batches
                batch_input = input_ids[i:i+32].to(self.device)
                batch_labels = labels[i:i+32].to(self.device)
                with torch.inference_mode():
                    result = get_response_log_probs(
                        self.model,
                        batch_input,
                        batch_labels,
                        return_token_entropy=False,
                    )
                old_log_probs_list.append(result["log_probs"].cpu())
            old_log_probs = torch.cat(old_log_probs_list, dim=0)
            self.model.train()

        # 5. Training loop: epochs and microbatches
        epoch_losses = []
        epoch_grad_norms = []
        epoch_entropies = []
        epoch_clip_fractions = []

        for epoch in range(self.cfg.epochs_per_rollout_batch):
            # Shuffle indices for this epoch
            indices = torch.randperm(len(input_ids))

            for mb_idx in range(0, len(indices), self.cfg.micro_train_batch_size):
                # Get microbatch indices
                mb_indices = indices[mb_idx:mb_idx + self.cfg.micro_train_batch_size]

                # Prepare microbatch
                mb_input = input_ids[mb_indices].to(self.device)
                mb_labels = labels[mb_indices].to(self.device)
                mb_mask = response_mask[mb_indices].to(self.device)
                mb_advantages = advantages[mb_indices].unsqueeze(1).to(self.device)
                mb_raw_rewards = raw_rewards[mb_indices].unsqueeze(1).to(self.device)

                mb_old_log_probs = None
                if old_log_probs is not None:
                    mb_old_log_probs = old_log_probs[mb_indices].to(self.device)

                # Forward pass to get policy log probs
                result = get_response_log_probs(
                    self.model,
                    mb_input,
                    mb_labels,
                    return_token_entropy=True,
                )
                policy_log_probs = result["log_probs"]
                token_entropy = result["token_entropy"]

                # Compute loss and backward
                loss, loss_metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
                    loss_type=self.cfg.loss_type,
                    raw_rewards=mb_raw_rewards if self.cfg.loss_type == "no_baseline" else None,
                    advantages=mb_advantages if self.cfg.loss_type != "no_baseline" else None,
                    old_log_probs=mb_old_log_probs,
                    cliprange=self.cfg.cliprange if self.cfg.loss_type == "grpo_clip" else None,
                    use_length_normalization=self.cfg.use_length_normalization,
                )

                epoch_losses.append(loss.item())
                avg_entropy = masked_mean(token_entropy, mb_mask, dim=None).item()
                epoch_entropies.append(avg_entropy)

                if "clip_fraction" in loss_metadata:
                    epoch_clip_fractions.append(loss_metadata["clip_fraction"].item())

                # Optimizer step every gradient_accumulation_steps microbatches
                if (mb_idx // self.cfg.micro_train_batch_size + 1) % self.cfg.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip
                    )
                    epoch_grad_norms.append(grad_norm.item())

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        # Aggregate epoch metrics
        step_metrics["train/loss"] = sum(epoch_losses) / len(epoch_losses)
        step_metrics["train/grad_norm"] = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        step_metrics["train/entropy"] = sum(epoch_entropies) / len(epoch_entropies)

        if epoch_clip_fractions:
            step_metrics["train/clip_fraction"] = sum(epoch_clip_fractions) / len(epoch_clip_fractions)

        return step_metrics

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print(f"Starting GRPO training: {self.cfg.loss_type}")
        print(f"Results directory: {self.run_dir}")
        print("=" * 80 + "\n")

        set_seed(self.cfg.seed)

        # Initial evaluation
        print("Initial evaluation...")
        val_metrics = self.evaluate()
        print(f"Initial validation: {val_metrics}")

        # Training loop
        for step in range(1, self.cfg.n_grpo_steps + 1):
            print(f"\n--- GRPO Step {step}/{self.cfg.n_grpo_steps} ---")

            # Training step
            train_metrics = self.train_step()

            # Log metrics
            if step % self.cfg.log_every_steps == 0:
                log_entry = {"step": step, **train_metrics}

                # Evaluation
                if step % self.cfg.eval_every_steps == 0:
                    print(f"  Running evaluation...")
                    val_metrics = self.evaluate()
                    log_entry.update({f"val/{k}": v for k, v in val_metrics.items()})
                    print(f"  Validation: {val_metrics}")

                # Save log
                self.training_logs.append(log_entry)
                with (self.run_dir / "training_log.jsonl").open('a') as f:
                    f.write(json.dumps(log_entry) + "\n")

                # Print summary
                print(f"  Train loss: {train_metrics.get('train/loss', 0.0):.4f}")
                print(f"  Train reward: {train_metrics.get('train/mean_reward', 0.0):.4f}")
                if 'val/reward' in log_entry:
                    print(f"  Val reward: {log_entry['val/reward']:.4f}")

            # Save checkpoint
            if step % self.cfg.save_every_steps == 0:
                checkpoint_dir = self.run_dir / f"checkpoint_step{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                print(f"  Saved checkpoint to {checkpoint_dir}")

        print("\n" + "=" * 80)
        print("Training complete!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training")

    # Model and data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.6B")
    parser.add_argument("--results_dir", type=Path, default=Path("results/grpo"))

    # Training hyperparameters
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)

    # Loss configuration
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline",
                       choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--use_std_normalization", type=bool, default=True)
    parser.add_argument("--use_length_normalization", type=bool, default=True)
    parser.add_argument("--cliprange", type=float, default=0.2)

    # Prompt type
    parser.add_argument("--prompt_type", type=str, default="r1_zero",
                       choices=["r1_zero", "question_only"])

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every_steps", type=int, default=5)

    args = parser.parse_args()

    # Create config
    config = GRPOConfig(
        model_name=args.model_name,
        results_dir=args.results_dir,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        loss_type=args.loss_type,
        use_std_normalization=args.use_std_normalization,
        use_length_normalization=args.use_length_normalization,
        cliprange=args.cliprange,
        prompt_type=args.prompt_type,
        seed=args.seed,
        eval_every_steps=args.eval_every_steps,
    )

    # Create trainer and train
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
