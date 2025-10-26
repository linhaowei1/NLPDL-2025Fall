"""
Script to evaluate a vLLM model on a dataset using a custom reward function.

This script loads a dataset, formats prompts, and runs evaluations for various
sampling configurations, including Best-of-N. It generates N responses per prompt,
selects the best one based on a reward function, and saves both the aggregated
best results and the detailed results for all generated samples.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from vllm import LLM, SamplingParams
from data.drgrpo_grader import r1_zero_reward_fn

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.6B"
DATASET_PATH = Path("./data/gsm8k/test.jsonl")
PROMPT_TEMPLATE_PATH = Path("./data/r1_zero.prompt")
RESULTS_DIR = Path("results")

# --- Define different sampling configurations to run ---
EVALUATION_CONFIGS = [
    # Baseline: Standard greedy decoding (n=1)
    {'n': 1, 'temperature': 1.0, 'top_p': 1.0, 'name': 'Best-of-1'},
    {'n': 2, 'temperature': 1.0, 'top_p': 1.0, 'name': 'Best-of-2'},
    {'n': 4, 'temperature': 1.0, 'top_p': 1.0, 'name': 'Best-of-4'},
    {'n': 8, 'temperature': 1.0, 'top_p': 1.0, 'name': 'Best-of-8'},
    {'n': 16, 'temperature': 1.0, 'top_p': 1.0, 'name': 'Best-of-16'}
]


class R1ZeroEvaluator:
    """
    Handles the evaluation of a language model using the R1-Zero prompt format.
    
    This class encapsulates data loading, prompt formatting, model generation,
    and reward calculation, with support for Best-of-N evaluation.
    """

    def __init__(self, model: LLM, dataset_path: Path, prompt_template_path: Path):
        """
        Initializes the evaluator with the model and data paths.

        Args:
            model: An initialized vLLM model instance.
            dataset_path: Path to the .jsonl dataset file.
            prompt_template_path: Path to the prompt template file.
        """
        self.llm = model
        self._prompt_template = self._load_prompt_template(prompt_template_path)
        self.prompt_to_answer_map = self._load_and_format_dataset(dataset_path)

    def _load_prompt_template(self, path: Path) -> str:
        """Loads the prompt template from a file."""
        try:
            return path.read_text()
        except FileNotFoundError:
            print(f"Error: Prompt template file not found at {path}")
            raise

    def _load_and_format_dataset(self, path: Path) -> Dict[str, str]:
        """
        Loads the dataset and formats it into a prompt-to-answer dictionary.
        
        This method is specific to the GSM8K dataset format, where the answer
        is separated by "####".
        """
        formatted_dataset = {}
        try:
            with path.open('r') as f:
                for line in f:
                    data = json.loads(line)
                    question = data['question']
                    # Extract the ground truth answer
                    answer = data['answer'].split("####")[-1].strip()
                    # Format the prompt using the template
                    prompt = self._prompt_template.replace("{question}", question)
                    formatted_dataset[prompt] = answer
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {path}")
            raise
        return formatted_dataset

    def _reward_fn(self, prompt: str, generated_text: str) -> Dict:
        """
        Calculates the reward for a generated response against the ground truth.
        """
        ground_truth_answer = self.prompt_to_answer_map[prompt]
        reward_scores = r1_zero_reward_fn(generated_text, ground_truth_answer)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth_answer,
            **reward_scores
        }

    def run(self, sampling_params: SamplingParams) -> Tuple[List[Dict], List[Dict]]:
        """
        Executes the full evaluation pipeline with Best-of-N sampling.

        Args:
            sampling_params: The sampling parameters for the vLLM generator.
                             'n' > 1 enables Best-of-N evaluation.

        Returns:
            A tuple containing:
            - best_results: A list of the highest-reward result for each prompt.
            - all_results: A list of all N results for each prompt.
        """
        prompts = list(self.prompt_to_answer_map.keys())
        print(f"Starting evaluation on {len(prompts)} prompts (n={sampling_params.n})...")

        outputs = self.llm.generate(prompts, sampling_params)
        
        best_results = []
        all_results = []
        
        for output in outputs:
            prompt = output.prompt

            # Evaluate all N generated responses for this prompt
            prompt_candidates = []
            for completion in output.outputs:
                # Append the required stop token to conform to the reward function's expected format
                generated_text = completion.text + "</answer>"
                result = self._reward_fn(prompt, generated_text)

                # Compute normalized log-likelihood for Best@N selection
                if completion.logprobs:
                    # Sum log probabilities over all tokens
                    total_logprob = sum(token_logprobs.logprob for token_logprobs in completion.logprobs)
                    # Normalize by number of tokens
                    num_tokens = len(completion.logprobs)
                    normalized_logprob = total_logprob / num_tokens if num_tokens > 0 else float('-inf')
                else:
                    normalized_logprob = float('-inf')

                result['normalized_logprob'] = normalized_logprob
                prompt_candidates.append(result)

            if not prompt_candidates:
                continue
            
            # Select the best candidate based on the main 'reward' score
            best_candidate = max(prompt_candidates, key=lambda x: x.get('reward', -float('inf')))
            
            best_results.append(best_candidate)
            all_results.extend(prompt_candidates)
            
        print("Evaluation complete.")
        return best_results, all_results

def evaluate_vllm(model: LLM, dataset_path: Path, prompt_template_path: Path, sampling_params: SamplingParams) -> Tuple[List[Dict], List[Dict]]:
    """
    Thin helper to run evaluation and return (best_results, all_results).
    Matches the README's wording for reuse in EI/RL.
    """
    evaluator = R1ZeroEvaluator(
        model=model,
        dataset_path=dataset_path,
        prompt_template_path=prompt_template_path
    )
    return evaluator.run(sampling_params)

def aggregate_results(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregates a list of result dictionaries to calculate the mean scores.

    Args:
        results: A list of evaluation result dictionaries.

    Returns:
        A dictionary containing the mean of each score type.
    """
    if not results:
        return {}
    
    num_results = len(results)
    total_scores = defaultdict(float)
    score_keys = ["format_reward", "answer_reward", "reward"]

    for result in results:
        for key in score_keys:
            total_scores[key] += result.get(key, 0.0)

    return {key: val / num_results for key, val in total_scores.items()}

def save_results(results: List[Dict], output_path: Path) -> None:
    """Saves the detailed evaluation results to a .jsonl file."""
    print(f"Saving {len(results)} results to {output_path}...")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open('w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print("Results saved.")

def generate_output_path(config: Dict, score: Dict, suffix: str) -> Path:
    """
    Generates a descriptive filepath for the results file.
    Example: results/Best-of-4_reward=0.8500_best.jsonl
    """
    reward_val = score.get('reward', 0.0)
    filename = f"{config['name']}_reward={reward_val:.4f}_{suffix}.jsonl"
    return RESULTS_DIR / filename

def main():
    """Main function to set up and run the evaluation for all configs."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Initialization ---
    print("Initializing model...")
    llm = LLM(model=MODEL_NAME)
    evaluator = R1ZeroEvaluator(
        model=llm,
        dataset_path=DATASET_PATH,
        prompt_template_path=PROMPT_TEMPLATE_PATH
    )

    # --- Run Evaluation for each configuration ---
    for config in EVALUATION_CONFIGS:
        print("\n" + "="*60)
        print(f"RUNNING CONFIGURATION: {config['name']}")
        print(f"Parameters: {config}")
        print("="*60)
        
        sampling_params = SamplingParams(
            n=config['n'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=1024,
            stop=["</answer>"],
            logprobs=1  # Enable log probabilities for Best@N selection
        )

        # Use helper to align with README spec
        best_results, all_results = evaluate_vllm(
            model=llm,
            dataset_path=DATASET_PATH,
            prompt_template_path=PROMPT_TEMPLATE_PATH,
            sampling_params=sampling_params,
        )

        # --- Process and Save Results ---
        if best_results:
            aggregate_score = aggregate_results(best_results)
            print(f"\nAggregated Score for {config['name']}: {aggregate_score}")

            # Print category counts over all generated samples
            correct = sum(1 for r in all_results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 1)
            format_only = sum(1 for r in all_results if r.get("format_reward", 0) == 1 and r.get("answer_reward", 0) == 0)
            neither = sum(1 for r in all_results if r.get("format_reward", 0) == 0 and r.get("answer_reward", 0) == 0)
            total = len(all_results)
            print(f"Category counts (over all {total} samples): correct={correct}, format_only={format_only}, neither={neither}")

            # Generate descriptive filenames and save both best and all results
            best_results_path = generate_output_path(config, aggregate_score, "best")
            all_results_path = generate_output_path(config, aggregate_score, "all")
            
            save_results(best_results, best_results_path)
            save_results(all_results, all_results_path)
        else:
            print(f"No results were generated for config: {config['name']}")

if __name__ == '__main__':
    main()
