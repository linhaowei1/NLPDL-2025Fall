"""
Best-of-N Analysis Script for Problem 2

This script analyzes the results from zero-shot.py to compute:
1. pass@N: At least one correct answer in N samples
2. Best@N: Best sample selected by normalized log-likelihood

Usage:
    python best_of_n.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis")


def load_results(filepath: Path) -> List[Dict]:
    """Load results from a jsonl file."""
    results = []
    with filepath.open('r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_pass_at_n(all_results: List[Dict], group_size: int) -> float:
    """
    Compute pass@N: percentage of prompts where at least one of N samples is correct.

    Args:
        all_results: All generated results
        group_size: Number of samples per prompt (N)

    Returns:
        pass@N accuracy
    """
    # Group results by prompt
    prompt_groups = defaultdict(list)
    for result in all_results:
        prompt = result['prompt']
        prompt_groups[prompt].append(result)

    # For each prompt, check if at least one sample is correct
    num_prompts = len(prompt_groups)
    num_passed = 0

    for prompt, samples in prompt_groups.items():
        # Take only the first group_size samples (in case we have more)
        samples = samples[:group_size]

        # Check if at least one is correct (reward == 1.0)
        if any(sample.get('reward', 0.0) == 1.0 for sample in samples):
            num_passed += 1

    return num_passed / num_prompts if num_prompts > 0 else 0.0


def compute_best_at_n(all_results: List[Dict], group_size: int) -> float:
    """
    Compute Best@N: Select best sample by normalized log-likelihood and check if correct.

    Uses the normalized log-likelihood (average log-prob per token) to select
    the best sample from N candidates, then checks if it's correct.
    """
    # Group results by prompt
    prompt_groups = defaultdict(list)
    for result in all_results:
        prompt = result['prompt']
        prompt_groups[prompt].append(result)

    num_prompts = len(prompt_groups)
    num_correct = 0

    for prompt, samples in prompt_groups.items():
        # Take only the first group_size samples
        samples = samples[:group_size]

        # Select by normalized log-likelihood (highest is best)
        best_sample = max(samples, key=lambda x: x.get('normalized_logprob', float('-inf')))

        if best_sample.get('reward', 0.0) == 1.0:
            num_correct += 1

    return num_correct / num_prompts if num_prompts > 0 else 0.0


def analyze_best_of_n():
    """Main analysis function for Problem 2."""

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Configuration for N values to analyze
    n_values = [1, 4, 8, 16]

    # Store results
    pass_at_n_results = {}
    best_at_n_results = {}

    print("=" * 80)
    print("BEST-OF-N ANALYSIS")
    print("=" * 80)

    for n in n_values:
        # Find the corresponding result file
        all_files = list(RESULTS_DIR.glob(f"Best-of-{n}_*_all.jsonl"))

        if not all_files:
            print(f"\nWarning: No results file found for N={n}")
            continue

        result_file = all_files[0]
        print(f"\nAnalyzing N={n} from {result_file.name}")

        # Load all results
        all_results = load_results(result_file)

        # Compute pass@N
        pass_n = compute_pass_at_n(all_results, n)
        pass_at_n_results[n] = pass_n

        # Compute Best@N
        best_n = compute_best_at_n(all_results, n)
        best_at_n_results[n] = best_n

        print(f"  pass@{n}: {pass_n:.4f} ({pass_n*100:.2f}%)")
        print(f"  Best@{n}: {best_n:.4f} ({best_n*100:.2f}%)")

        # Additional analysis: category breakdown
        correct = sum(1 for r in all_results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 1)
        format_only = sum(1 for r in all_results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 0)
        neither = sum(1 for r in all_results if r.get("format_reward", 0) == 0)
        total = len(all_results)

        print(f"  Category breakdown (all {total} samples):")
        print(f"    - Correct (format=1, answer=1): {correct} ({correct/total*100:.2f}%)")
        print(f"    - Format only (format=1, answer=0): {format_only} ({format_only/total*100:.2f}%)")
        print(f"    - Neither (format=0): {neither} ({neither/total*100:.2f}%)")

    # Create comparison plot
    print("\n" + "=" * 80)
    print("Creating comparison plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by N
    sorted_n = sorted(pass_at_n_results.keys())
    pass_values = [pass_at_n_results[n] for n in sorted_n]
    best_values = [best_at_n_results[n] for n in sorted_n]

    # Plot both metrics
    ax.plot(sorted_n, pass_values, marker='o', linewidth=2, markersize=8, label='pass@N', color='blue')
    ax.plot(sorted_n, best_values, marker='s', linewidth=2, markersize=8, label='Best@N', color='red')

    ax.set_xlabel('N (number of samples)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('pass@N vs Best@N Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted_n)

    # Add percentage labels on the points
    for n, pass_val, best_val in zip(sorted_n, pass_values, best_values):
        ax.annotate(f'{pass_val*100:.1f}%', (n, pass_val),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='blue')
        ax.annotate(f'{best_val*100:.1f}%', (n, best_val),
                   textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='red')

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "pass_at_n_vs_best_at_n.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Save numerical results
    results_summary = {
        'pass_at_n': pass_at_n_results,
        'best_at_n': best_at_n_results,
        'gap': {n: pass_at_n_results[n] - best_at_n_results[n] for n in sorted_n}
    }

    summary_path = OUTPUT_DIR / "best_of_n_summary.json"
    with summary_path.open('w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Analysis and interpretation
    print("\n" + "=" * 80)
    print("ANALYSIS & INTERPRETATION")
    print("=" * 80)
    print("\nGap between pass@N and Best@N:")
    for n in sorted_n:
        gap = pass_at_n_results[n] - best_at_n_results[n]
        print(f"  N={n}: {gap:.4f} ({gap*100:.2f} percentage points)")

    print("\nInterpretation:")
    print("- pass@N measures the model's CAPABILITY: Can it generate a correct answer?")
    print("- Best@N measures the model's RECOGNITION: Can it identify its best answer?")
    print("- The gap shows how much performance is lost due to imperfect self-evaluation.")
    print("- A large gap suggests the model can solve problems but struggles to recognize")
    print("  which of its solutions are correct using likelihood alone.")

    print("\n" + "=" * 80)
    print("PROBLEM 2 DELIVERABLES COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    analyze_best_of_n()
