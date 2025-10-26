"""
Evaluator for kissing number lower bound via integer-center construction (d=11).

The user program should output a set of integer vectors (points) in R^d
representing candidate centers C, such that 0 not in C and

  min_{x!=y in C} ||x - y|| >= max_{x in C} ||x||.

If valid, by the lemma in the prompt, unit spheres centered at {2x/||x||}
form a kissing configuration in dimension d, and the score is |C|.

Accepted interfaces for the user program (preferred first):
- run_code() -> (points, d) or (points,) where points is (n,d) array-like of ints
  Optionally returns d; otherwise inferred from points shape.
- get_centers() -> points
- global variable named `points` (array-like)

Metrics returned:
- num_points: number of centers |C|
- dimension: ambient dimension d
- validity: 1.0 if lemma condition satisfied; else 0.0
- combined_score: equals |C| if valid else 0.0
"""

import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")



def _compute_squared_norm_int(vec: np.ndarray) -> int:
    v = np.asarray(vec, dtype=np.int64)
    return int(np.dot(v, v))


def _verify_lemma_condition(points: np.ndarray) -> tuple[bool, dict]:
    """
    Verifies the lemma condition using exact integer arithmetic after rounding.

    Returns (is_valid, info_dict)
    """
    if points.ndim != 2:
        return False, {"error": f"Expected 2D array, got shape {points.shape}"}

    # Round to integers to ensure exact computation
    P = np.around(points).astype(np.int64)

    # 0 not in C
    squared_norms = np.sum(P.astype(np.int64) * P.astype(np.int64), axis=1)
    min_sq = int(np.min(squared_norms)) if len(squared_norms) > 0 else 0
    if min_sq <= 0:
        return False, {"error": "Set contains 0 vector or is empty."}

    # min pairwise distance >= max norm
    max_sq = int(np.max(squared_norms))
    n = P.shape[0]
    if n <= 1:
        return False, {"error": "Need at least 2 points."}

    min_pair_sq = None
    for i in range(n):
        for j in range(i + 1, n):
            diff = P[i] - P[j]
            d2 = int(np.dot(diff, diff))
            if min_pair_sq is None or d2 < min_pair_sq:
                min_pair_sq = d2

    if min_pair_sq is None:
        return False, {"error": "Failed to compute pairwise distances."}

    is_valid = min_pair_sq >= max_sq
    info = {
        "min_squared_distance": int(min_pair_sq),
        "max_squared_norm": int(max_sq),
    }
    if not is_valid:
        info["error"] = (
            f"Minimum squared distance {min_pair_sq} < maximum squared norm {max_sq}"
        )
    return is_valid, info

def run_with_timeout(program_path, timeout_seconds=600):
    """
    Run the user program in a separate process with timeout, save results via pickle.

    Returns:
        points (np.ndarray)
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Note the escaped braces: {{ and }} for literals in the inner script
        script = f"""
import sys, os, pickle, traceback, numpy as np, importlib.util

sys.path.insert(0, os.path.dirname('{program_path}'))

def _load(path: str):
    spec = importlib.util.spec_from_file_location("user_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

try:
    print(f"Running in subprocess, Python: {{sys.version}}")
    print(f"Program path: {program_path}")
    mod = _load('{program_path}')

    pts = None
    if hasattr(mod, "run_code") and callable(getattr(mod, "run_code")):
        print("Calling run_code()...")
        out = mod.run_code()
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            pts = out[0]
        else:
            pts = out
    if pts is None and hasattr(mod, "get_centers") and callable(getattr(mod, "get_centers")):
        print("Calling get_centers()...")
        pts = mod.get_centers()
    if pts is None and hasattr(mod, "points"):
        print("Using global 'points'...")
        pts = getattr(mod, "points")
    if pts is None:
        raise RuntimeError("Expected run_code(), get_centers(), or global 'points'.")

    pts = np.asarray(pts, dtype=float)

    results = {{"points": pts}}  # <--- FIX 1
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")

except Exception as e:
    print(f"Error in subprocess: {{e}}")  # <--- FIX 2
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{"error": str(e)}}, f)  # <--- FIX 3
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["points"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)



def evaluate(program_path):
    """
    Evaluate the kissing configuration by verifying the lemma condition.

    Returns:
        Dictionary of metrics.
    """
    try:
        start_time = time.time()
        points = run_with_timeout(program_path, timeout_seconds=600)
        end_time = time.time()
        eval_time = end_time - start_time

        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=float)

        if points.ndim != 2 or points.shape[1] <= 0:
            print(f"Invalid points shape: {points.shape}")
            return {
                "num_points": 0,
                "dimension": 0,
                "validity": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
            }

        n, d = points.shape

        is_valid, info = _verify_lemma_condition(points)
        if not is_valid:
            print(f"Verification failed: {info.get('error', 'unknown error')}")
            return {
                "num_points": 0,
                "dimension": int(d),
                "validity": 0.0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
            }

        print(f"Verified: dimension={d}, kissing lower bound >= {n}")
        return {
            "num_points": int(n),
            "dimension": int(d),
            "validity": 1.0,
            "eval_time": float(eval_time),
            "combined_score": float(n) / 593,
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "num_points": 0,
            "dimension": 0,
            "validity": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }


