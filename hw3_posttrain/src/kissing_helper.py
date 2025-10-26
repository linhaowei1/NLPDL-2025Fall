"""
Kissing Number Problem Helper (Problem 17)

This script helps with the agentic approach to finding a good lower bound for
the kissing number in 11 dimensions.

It provides:
1. A framework for iterative refinement with an LLM
2. Verification after each attempt
3. Logging of the conversation
4. Best-of-N sampling support

Usage:
    python kissing_helper.py --mode interactive  # Interactive refinement
    python kissing_helper.py --mode batch --n 10  # Generate N candidates
    python kissing_helper.py --verify             # Just verify current program.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_evaluator(program_path: Path) -> Dict:
    """Run the evaluator on a program file."""
    result = subprocess.run(
        [sys.executable, "data/kissing_number/evaluator.py", str(program_path)],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")

    # Parse the output to extract metrics
    # The evaluator prints metrics at the end
    lines = result.stdout.strip().split('\n')
    metrics = {}

    # Look for verification info in output
    for line in lines:
        if "Verified:" in line:
            # Extract num_points from the line
            if ">=" in line:
                num_points = line.split(">=")[-1].strip()
                metrics["num_points"] = int(num_points)
                metrics["validity"] = 1.0
        elif "Verification failed:" in line:
            metrics["validity"] = 0.0
            metrics["error"] = line.split(":", 1)[-1].strip()

    return metrics


def verify_current_program():
    """Verify the current program.py."""
    program_path = Path("data/kissing_number/program.py")

    print("\n" + "=" * 80)
    print("VERIFYING CURRENT PROGRAM")
    print("=" * 80 + "\n")

    metrics = run_evaluator(program_path)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(json.dumps(metrics, indent=2))

    if metrics.get("validity") == 1.0:
        print(f"\n✓ VALID! Score: {metrics.get('num_points', 0)} points")
    else:
        print(f"\n✗ INVALID: {metrics.get('error', 'Unknown error')}")

    return metrics


def save_conversation_log(log_path: Path, conversation: List[Dict]):
    """Save conversation log to JSON."""
    with log_path.open('w') as f:
        json.dump(conversation, f, indent=2)
    print(f"\nConversation log saved to: {log_path}")


def generate_initial_prompt() -> str:
    """Generate the initial prompt for the LLM."""
    return """I need your help to find a large set of integer vectors C in 11 dimensions such that:
1. 0 is not in C
2. The minimum pairwise distance between vectors >= maximum norm of any vector
   Mathematically: min_{x ≠ y in C} ||x - y|| >= max_{x in C} ||x||

If this condition holds, it proves a lower bound for the kissing number: τ_11 >= |C|.

Your task: Write a Python function that constructs such a set. The larger |C|, the better.

Key constraints:
- All vectors must be integers (in Z^11)
- The set must satisfy the distance condition above

Current known bounds: 595 <= τ_11 <= 870

Please provide Python code that defines a function construct_centers(d=11) that returns
a numpy array of shape (n, 11) where n is as large as possible while satisfying the constraints.

Think carefully about geometric constructions. Some approaches to consider:
1. Lattice constructions (E8, Leech lattice projections)
2. Coordinate permutations with specific patterns
3. Sum of orthogonal components
4. Known kissing configurations in lower dimensions extended to 11D

Start with a concrete construction that you're confident will satisfy the constraints."""


def interactive_mode():
    """Interactive mode for working with an LLM."""
    print("\n" + "=" * 80)
    print("INTERACTIVE KISSING NUMBER HELPER")
    print("=" * 80 + "\n")

    print("This is a helper for Problem 17: Kissing Number Challenge")
    print("\nWorkflow:")
    print("1. Use the initial prompt below with an LLM (e.g., Qwen3-32B)")
    print("2. Implement the suggested construction in data/kissing_number/program.py")
    print("3. Run verification: python src/kissing_helper.py --verify")
    print("4. Based on the result, refine with the LLM")
    print("5. Repeat until you get a good score!")

    print("\n" + "=" * 80)
    print("INITIAL PROMPT FOR LLM:")
    print("=" * 80)
    print(generate_initial_prompt())

    print("\n" + "=" * 80)
    print("TIPS:")
    print("=" * 80)
    print("- Start simple: The baseline (22 points) uses ±2e_i")
    print("- Test incrementally: Verify after each change")
    print("- Use exact feedback: Copy error messages back to the LLM")
    print("- Think geometrically: What configurations maximize packing?")
    print("- Reference: The E8 lattice achieves τ_8 = 240")

    print("\n" + "=" * 80)


def create_report_template():
    """Create a report template for the kissing number solution."""
    report = """# Kissing Number Challenge Report (Problem 17)

## Final Result

- **Final Score**: [Your |C| value]
- **Dimension**: 11
- **Validity**: [Pass/Fail]

## Methodology

[Describe your approach - examples:]
- **Approach Used**: [e.g., "Iterative refinement with Qwen3-32B", "Best-of-N with verification", etc.]
- **Key Strategy**: [e.g., "Started with standard lattice basis, then added carefully chosen combinations"]

## Process

### Initial Attempt

**Baseline Construction**: [Describe what you started with]
- Score: 22 (±2e_i basis vectors)
- This gave us a starting point to improve upon

### Iteration 1

**Prompt**: [Show the prompt you used]

**Model Response**: [Summarize what the model suggested]

**Result**: [Did it work? What was the score?]

### Iteration 2

[Continue for each significant iteration...]

### Breakthrough Moment

[Describe what led to your best result - what insight or strategy worked?]

## Key Insights

1. [What did you learn about the problem?]
2. [What worked well?]
3. [What didn't work?]

## Reflection

[2-3 paragraphs reflecting on:]
- What was most effective in your strategy?
- How did the LLM help (or not help)?
- What would you try differently with more time?
- What did you learn about using LLMs for mathematical discovery?

## Conversation Log

[Note: Full conversation log is in conversation_log.json]

Key prompts that led to improvements:
1. [Quote important prompts]
2. [...]

"""

    report_path = Path("data/kissing_number/report_template.md")
    report_path.write_text(report)
    print(f"Report template created at: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Kissing Number Problem Helper")
    parser.add_argument("--mode", type=str, choices=["interactive", "verify"],
                       default="interactive", help="Mode of operation")

    args = parser.parse_args()

    if args.mode == "verify":
        verify_current_program()
    elif args.mode == "interactive":
        interactive_mode()
        create_report_template()

        # Also verify current state
        print("\n" + "=" * 80)
        print("Current program status:")
        verify_current_program()


if __name__ == "__main__":
    main()
