"""
GRPO Experiment Runner and Analysis Script

This script helps run the experiments for Problems 9-16 and analyze results.

Usage:
    python run_experiments.py --experiment problem9  # Learning rate sweep
    python run_experiments.py --experiment problem10  # Baseline ablation
    python run_experiments.py --analyze              # Analyze and plot results
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def run_command(cmd: List[str]):
    """Run a command and stream output."""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode


def problem9_learning_rate_sweep():
    """Problem 9: Learning rate sweep."""
    print("\n" + "=" * 80)
    print("PROBLEM 9: Learning Rate Sweep")
    print("=" * 80 + "\n")

    learning_rates = [5e-6, 1e-5, 2e-5, 5e-5]

    for lr in learning_rates:
        print(f"\nTraining with learning_rate={lr}")
        cmd = [
            "python", "src/grpo_train.py",
            "--learning_rate", str(lr),
            "--n_grpo_steps", "200",
            "--loss_type", "reinforce_with_baseline",
            "--use_std_normalization", "True",
            "--eval_every_steps", "5"
        ]
        run_command(cmd)


def problem10_baseline_ablation():
    """Problem 10: Effect of baselining."""
    print("\n" + "=" * 80)
    print("PROBLEM 10: Baseline Ablation")
    print("=" * 80 + "\n")

    loss_types = ["no_baseline", "reinforce_with_baseline"]

    for loss_type in loss_types:
        print(f"\nTraining with loss_type={loss_type}")
        cmd = [
            "python", "src/grpo_train.py",
            "--loss_type", loss_type,
            "--n_grpo_steps", "200",
            "--learning_rate", "1e-5",  # Use best LR from problem 9
            "--eval_every_steps", "5"
        ]
        run_command(cmd)


def problem12_length_normalization():
    """Problem 12: Effect of length normalization."""
    print("\n" + "=" * 80)
    print("PROBLEM 12: Length Normalization Ablation")
    print("=" * 80 + "\n")

    for use_length_norm in [True, False]:
        print(f"\nTraining with use_length_normalization={use_length_norm}")
        cmd = [
            "python", "src/grpo_train.py",
            "--use_length_normalization", str(use_length_norm),
            "--n_grpo_steps", "200",
            "--learning_rate", "1e-5",
            "--eval_every_steps", "5"
        ]
        run_command(cmd)


def problem13_std_normalization():
    """Problem 13: Effect of std normalization."""
    print("\n" + "=" * 80)
    print("PROBLEM 13: Std Normalization Ablation")
    print("=" * 80 + "\n")

    for use_std in [True, False]:
        print(f"\nTraining with use_std_normalization={use_std}")
        cmd = [
            "python", "src/grpo_train.py",
            "--use_std_normalization", str(use_std),
            "--n_grpo_steps", "200",
            "--learning_rate", "1e-5",
            "--eval_every_steps", "5"
        ]
        run_command(cmd)


def problem15_off_policy():
    """Problem 15: Off-policy hyperparameter sweep."""
    print("\n" + "=" * 80)
    print("PROBLEM 15: Off-Policy Training Sweep")
    print("=" * 80 + "\n")

    # Broad sweep: different epochs and batch sizes
    configs = [
        {"epochs": 1, "train_batch": 256},   # On-policy baseline
        {"epochs": 2, "train_batch": 512},   # Mild off-policy
        {"epochs": 4, "train_batch": 1024},  # More off-policy
        {"epochs": 8, "train_batch": 1024},  # Heavy off-policy
    ]

    for cfg in configs:
        epochs = cfg["epochs"]
        train_batch = cfg["train_batch"]
        grad_accum = train_batch // 2  # Keep microbatch size = 2

        print(f"\nTraining with epochs={epochs}, train_batch_size={train_batch}")
        cmd = [
            "python", "src/grpo_train.py",
            "--loss_type", "grpo_clip",
            "--epochs_per_rollout_batch", str(epochs),
            "--train_batch_size", str(train_batch),
            "--gradient_accumulation_steps", str(grad_accum),
            "--n_grpo_steps", "200",
            "--learning_rate", "1e-5",
            "--eval_every_steps", "5"
        ]
        run_command(cmd)


def problem16_prompt_ablation():
    """Problem 16: Prompt ablation."""
    print("\n" + "=" * 80)
    print("PROBLEM 16: Prompt Ablation")
    print("=" * 80 + "\n")

    for prompt_type in ["r1_zero", "question_only"]:
        print(f"\nTraining with prompt_type={prompt_type}")
        cmd = [
            "python", "src/grpo_train.py",
            "--prompt_type", prompt_type,
            "--n_grpo_steps", "200",
            "--learning_rate", "1e-5",
            "--eval_every_steps", "5"
        ]
        run_command(cmd)


def load_training_log(log_path: Path) -> List[Dict]:
    """Load training log from jsonl file."""
    logs = []
    if log_path.exists():
        with log_path.open('r') as f:
            for line in f:
                logs.append(json.loads(line))
    return logs


def plot_training_curves(results_dir: Path, output_path: Path):
    """Plot training curves from all runs."""
    print(f"\nPlotting training curves from {results_dir}")

    # Find all run directories
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not run_dirs:
        print("No run directories found!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for run_dir in run_dirs:
        log_file = run_dir / "training_log.jsonl"
        if not log_file.exists():
            continue

        logs = load_training_log(log_file)
        if not logs:
            continue

        # Extract metrics
        steps = [log['step'] for log in logs]
        train_rewards = [log.get('train/mean_reward', 0) for log in logs]
        val_rewards = [log.get('val/reward', None) for log in logs]
        losses = [log.get('train/loss', 0) for log in logs]
        entropies = [log.get('train/entropy', 0) for log in logs]

        # Filter out None values for validation
        val_steps = [s for s, v in zip(steps, val_rewards) if v is not None]
        val_rewards_filtered = [v for v in val_rewards if v is not None]

        run_name = run_dir.name

        # Plot
        axes[0].plot(steps, train_rewards, label=run_name, alpha=0.7)
        axes[1].plot(val_steps, val_rewards_filtered, label=run_name, alpha=0.7, marker='o')
        axes[2].plot(steps, losses, label=run_name, alpha=0.7)
        axes[3].plot(steps, entropies, label=run_name, alpha=0.7)

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Train Reward')
    axes[0].set_title('Training Reward')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Validation Reward')
    axes[1].set_title('Validation Reward')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Entropy')
    axes[3].set_title('Token Entropy')
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def analyze_results():
    """Analyze and plot all results."""
    print("\n" + "=" * 80)
    print("ANALYZING RESULTS")
    print("=" * 80 + "\n")

    results_dir = Path("results/grpo")
    analysis_dir = Path("results/analysis")
    analysis_dir.mkdir(exist_ok=True, parents=True)

    # Plot training curves
    plot_training_curves(results_dir, analysis_dir / "grpo_training_curves.png")

    print("\nAnalysis complete!")


def main():
    parser = argparse.ArgumentParser(description="GRPO Experiment Runner")
    parser.add_argument("--experiment", type=str, choices=[
        "problem9", "problem10", "problem12", "problem13", "problem15", "problem16"
    ], help="Which experiment to run")
    parser.add_argument("--analyze", action="store_true", help="Analyze and plot results")
    parser.add_argument("--all", action="store_true", help="Run all experiments (WARNING: takes a long time)")

    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    elif args.all:
        problem9_learning_rate_sweep()
        problem10_baseline_ablation()
        problem12_length_normalization()
        problem13_std_normalization()
        problem15_off_policy()
        problem16_prompt_ablation()
        analyze_results()
    elif args.experiment:
        if args.experiment == "problem9":
            problem9_learning_rate_sweep()
        elif args.experiment == "problem10":
            problem10_baseline_ablation()
        elif args.experiment == "problem12":
            problem12_length_normalization()
        elif args.experiment == "problem13":
            problem13_std_normalization()
        elif args.experiment == "problem15":
            problem15_off_policy()
        elif args.experiment == "problem16":
            problem16_prompt_ablation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
