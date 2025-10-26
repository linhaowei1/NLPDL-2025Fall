
import numpy as np


def construct_centers(d: int = 11) -> np.ndarray:
    
    if d < 1:
        raise ValueError("Dimension must be at least 1")

    centers = []
    # Add ±2 e_i for i in 0..d-1
    for i in range(d):
        v = np.zeros(d, dtype=int)
        v[i] = 2
        centers.append(v.copy())
        centers.append((-v).copy())

    return np.array(centers, dtype=int)


def check_lemma(points: np.ndarray) -> bool:
    P = np.asarray(points, dtype=int)
    if P.ndim != 2:
        return False
    # 0 not in C
    norms_sq = np.sum(P * P, axis=1)
    if len(norms_sq) == 0 or np.min(norms_sq) <= 0:
        return False
    max_sq = int(np.max(norms_sq))
    n = P.shape[0]
    if n <= 1: # Need at least 2 points for pairwise comparison
        return False
        
    min_pair_sq = None
    for i in range(n):
        for j in range(i + 1, n):
            diff = P[i] - P[j]
            d2 = int(np.dot(diff, diff))
            if min_pair_sq is None or d2 < min_pair_sq:
                min_pair_sq = d2
    return min_pair_sq is not None and min_pair_sq >= max_sq


def run_code():
    """Return (points,) where points is an (n, d) integer array.

    The evaluator will verify the lemma condition and score by |C|.
    """
    points = construct_centers(d=11)
    assert check_lemma(points), "Starter set failed lemma check unexpectedly"
    return (points,)

if __name__ == "__main__":
    pts_tuple = run_code()
    pts = pts_tuple[0]
    print(f"Constructed d=11 centers with |C|={pts.shape[0]}")