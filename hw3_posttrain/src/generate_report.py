"""
Report Generation Utilities

This script helps generate analysis reports and summaries for the homework.

Usage:
    python generate_report.py --problem 2   # Generate Problem 2 report
    python generate_report.py --all         # Generate all reports
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from a jsonl file."""
    data = []
    if filepath.exists():
        with filepath.open('r') as f:
            for line in f:
                data.append(json.loads(line))
    return data


def generate_problem1_report():
    """Generate report for Problem 1."""
    print("\n" + "=" * 80)
    print("PROBLEM 1: Zero-Shot GSM8K Performance")
    print("=" * 80 + "\n")

    # Load N=1 results (zero-shot baseline)
    results_file = list(Path("results").glob("Best-of-1_*_all.jsonl"))

    if not results_file:
        print("Error: No results found. Run zero-shot.py first.")
        return

    results = load_jsonl(results_file[0])

    # Count categories
    total = len(results)
    correct = sum(1 for r in results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 1)
    format_only = sum(1 for r in results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 0)
    neither = sum(1 for r in results if r.get("format_reward", 0) == 0)

    print(f"Total examples: {total}")
    print(f"\nCategory breakdown:")
    print(f"  1. Correct (format=1, answer=1): {correct} ({correct/total*100:.2f}%)")
    print(f"  2. Format only (format=1, answer=0): {format_only} ({format_only/total*100:.2f}%)")
    print(f"  3. Neither (format=0, answer=0): {neither} ({neither/total*100:.2f}%)")

    # Sample examples from each category
    print("\n" + "-" * 80)
    print("Sample examples from each category:")
    print("-" * 80)

    # Category 1: Correct
    correct_examples = [r for r in results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 1]
    if correct_examples:
        ex = correct_examples[0]
        print("\n[Category 1: Correct]")
        print(f"Question: {ex['prompt'][:200]}...")
        print(f"Generated: {ex['generated_text'][:200]}...")
        print(f"Ground truth: {ex['ground_truth']}")

    # Category 2: Format only
    format_examples = [r for r in results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 0]
    if format_examples:
        ex = format_examples[0]
        print("\n[Category 2: Format correct but answer wrong]")
        print(f"Question: {ex['prompt'][:200]}...")
        print(f"Generated: {ex['generated_text'][:200]}...")
        print(f"Ground truth: {ex['ground_truth']}")

    # Category 3: Neither
    neither_examples = [r for r in results if r.get("format_reward", 0) == 0]
    if neither_examples:
        ex = neither_examples[0]
        print("\n[Category 3: Format incorrect]")
        print(f"Question: {ex['prompt'][:200]}...")
        print(f"Generated: {ex['generated_text'][:200]}...")
        print(f"Ground truth: {ex['ground_truth']}")

    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)
    print("\nFor format=0 cases:")
    print("  Issue: Model failed to follow the prompt structure (<think> </think> <answer> </answer>)")
    print("  Likely cause: Model sometimes doesn't close tags properly or uses different format")
    print("  Parser vs model: Both contribute - model should follow format better, but")
    print("                   parser could be more lenient for minor deviations")

    print("\nFor format=1, answer=0 cases:")
    print("  Issue: Model followed format but computed wrong answer")
    print("  Likely cause: Mathematical reasoning error in the <think> section")
    print("  This is expected - the base model makes mistakes on math problems")

    # Save report
    report_path = Path("results/analysis/problem1_report.txt")
    report_path.parent.mkdir(exist_ok=True, parents=True)

    with report_path.open('w') as f:
        f.write(f"Problem 1: Zero-Shot GSM8K Performance\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Total examples: {total}\n\n")
        f.write(f"Category breakdown:\n")
        f.write(f"  1. Correct: {correct} ({correct/total*100:.2f}%)\n")
        f.write(f"  2. Format only: {format_only} ({format_only/total*100:.2f}%)\n")
        f.write(f"  3. Neither: {neither} ({neither/total*100:.2f}%)\n")

    print(f"\nReport saved to: {report_path}")


def generate_problem2_report():
    """Generate report for Problem 2."""
    print("\n" + "=" * 80)
    print("PROBLEM 2: Best-of-N Analysis")
    print("=" * 80 + "\n")

    # Check if analysis has been run
    summary_path = Path("results/analysis/best_of_n_summary.json")

    if not summary_path.exists():
        print("Error: Analysis not run yet. Run: python src/best_of_n.py")
        return

    with summary_path.open('r') as f:
        summary = json.load(f)

    print("Results:")
    print("-" * 80)
    print(f"{'N':<5} {'pass@N':<12} {'Best@N':<12} {'Gap':<12}")
    print("-" * 80)

    for n in sorted(summary['pass_at_n'].keys(), key=int):
        pass_n = summary['pass_at_n'][n]
        best_n = summary['best_at_n'][n]
        gap = summary['gap'][n]
        print(f"{n:<5} {pass_n:<12.4f} {best_n:<12.4f} {gap:<12.4f}")

    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)
    print("\nThe gap between pass@N and Best@N reveals important insights:")
    print("\n1. Capability vs Recognition:")
    print("   - pass@N measures: Can the model generate a correct solution?")
    print("   - Best@N measures: Can the model identify its best solution?")
    print("\n2. The gap shows the model struggles with self-evaluation:")
    print("   - It can solve problems (high pass@N)")
    print("   - But can't reliably identify which solutions are correct (lower Best@N)")
    print("   - This suggests using likelihood alone is insufficient for selection")
    print("\n3. Implications:")
    print("   - Verification-based selection (as in Best@N with ground truth) is valuable")
    print("   - This motivates RL with verified rewards (GRPO)")
    print("   - Training could help the model learn better self-evaluation")

    print(f"\nFull analysis available in: results/analysis/")


def generate_grpo_summary():
    """Generate summary of all GRPO runs."""
    print("\n" + "=" * 80)
    print("GRPO Training Summary")
    print("=" * 80 + "\n")

    grpo_dir = Path("results/grpo")

    if not grpo_dir.exists() or not list(grpo_dir.iterdir()):
        print("No GRPO runs found yet.")
        return

    # Find all run directories
    runs = [d for d in grpo_dir.iterdir() if d.is_dir()]

    print(f"Found {len(runs)} GRPO runs:\n")

    summary_data = []

    for run_dir in runs:
        log_file = run_dir / "training_log.jsonl"
        config_file = run_dir / "config.json"

        if not log_file.exists() or not config_file.exists():
            continue

        # Load config
        with config_file.open('r') as f:
            config = json.load(f)

        # Load logs
        logs = load_jsonl(log_file)

        if not logs:
            continue

        # Get final metrics
        final_log = logs[-1]

        # Get best validation reward
        val_rewards = [log.get('val/reward', 0) for log in logs if 'val/reward' in log]
        best_val_reward = max(val_rewards) if val_rewards else 0

        summary = {
            'run_name': run_dir.name,
            'loss_type': config.get('loss_type', 'unknown'),
            'learning_rate': config.get('learning_rate', 0),
            'use_std_norm': config.get('use_std_normalization', None),
            'use_len_norm': config.get('use_length_normalization', None),
            'final_val_reward': final_log.get('val/reward', 0),
            'best_val_reward': best_val_reward,
            'n_steps': len(logs),
        }

        summary_data.append(summary)

        print(f"Run: {run_dir.name}")
        print(f"  Loss type: {summary['loss_type']}")
        print(f"  Learning rate: {summary['learning_rate']}")
        print(f"  Final val reward: {summary['final_val_reward']:.4f}")
        print(f"  Best val reward: {summary['best_val_reward']:.4f}")
        print()

    # Save summary
    if summary_data:
        summary_path = Path("results/analysis/grpo_summary.json")
        with summary_path.open('w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate homework reports")
    parser.add_argument("--problem", type=int, choices=[1, 2], help="Generate report for specific problem")
    parser.add_argument("--grpo", action="store_true", help="Generate GRPO summary")
    parser.add_argument("--all", action="store_true", help="Generate all reports")

    args = parser.parse_args()

    Path("results/analysis").mkdir(exist_ok=True, parents=True)

    if args.all:
        generate_problem1_report()
        generate_problem2_report()
        generate_grpo_summary()
    elif args.problem == 1:
        generate_problem1_report()
    elif args.problem == 2:
        generate_problem2_report()
    elif args.grpo:
        generate_grpo_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
