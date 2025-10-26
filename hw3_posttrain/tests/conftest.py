import hashlib
import os
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
import torch
from torch import Tensor


def pytest_addoption(parser):
    parser.addoption("--snapshot-exact", action="store_true", help="Use exact matching standards for snapshot matching")

_A = TypeVar("_A", np.ndarray, Tensor)


def _canonicalize_array(arr: _A) -> np.ndarray:
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


class NumpySnapshot:
    """Snapshot testing utility for NumPy arrays using .npz format."""

    def __init__(
        self,
        snapshot_dir: str = "tests/_snapshots",
    ):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _get_snapshot_path(self, test_name: str) -> Path:
        """Get the path to the snapshot file."""
        return self.snapshot_dir / f"{test_name}.npz"

    def assert_match(
        self,
        actual: _A | dict[str, _A],
        test_name: str,
        force_update: bool = False,
        rtol: float = 1e-4,
        atol: float = 1e-2,
    ):
        """
        Assert that the actual array(s) matches the snapshot.

        Args:
            actual: Single NumPy array or dictionary of named arrays
            test_name: The name of the test (used for the snapshot file)
            update: If True, update the snapshot instead of comparing
        """
        snapshot_path = self._get_snapshot_path(test_name)

        # Convert single array to dictionary for consistent handling
        arrays_dict = actual if isinstance(actual, dict) else {"array": actual}
        arrays_dict = {k: _canonicalize_array(v) for k, v in arrays_dict.items()}


        # Load the snapshot
        expected_arrays = dict(np.load(snapshot_path))

        # Verify all expected arrays are present
        missing_keys = set(arrays_dict.keys()) - set(expected_arrays.keys())
        if missing_keys:
            raise AssertionError(f"Keys {missing_keys} not found in snapshot for {test_name}")

        # Verify all actual arrays are expected
        extra_keys = set(expected_arrays.keys()) - set(arrays_dict.keys())
        if extra_keys:
            raise AssertionError(f"Snapshot contains extra keys {extra_keys} for {test_name}")

        # Compare all arrays
        for key in arrays_dict:
            np.testing.assert_allclose(
                _canonicalize_array(arrays_dict[key]),
                expected_arrays[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"Array '{key}' does not match snapshot for {test_name}",
            )


# Fixture that can be used in all tests
@pytest.fixture
def numpy_snapshot(request):
    """
    Fixture providing numpy snapshot testing functionality.

    Usage:
        def test_my_function(numpy_snapshot):
            result = my_function()
            numpy_snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    match_exact = request.config.getoption("--snapshot-exact", default=False)

    # Create the snapshot handler with default settings
    snapshot = NumpySnapshot()

    # Patch the assert_match method to include the update flag by default
    original_assert_match = snapshot.assert_match

    def patched_assert_match(actual, test_name=None, force_update=force_update, rtol=1e-4, atol=1e-2):
        # If test_name is not provided, use the test function name
        if test_name is None:
            test_name = request.node.name
        if match_exact:
            rtol = atol = 0
        return original_assert_match(actual, test_name=test_name, force_update=force_update, rtol=rtol, atol=atol)

    snapshot.assert_match = patched_assert_match

    return snapshot


# ==========================================
# Fixtures for GRPO tests
# ==========================================

@pytest.fixture
def reward_fn():
    """Dummy reward function for testing."""
    def dummy_reward_fn(response, ground_truth):
        # Use SHA-256 which is deterministic
        response_hash = int(hashlib.sha256(response.encode()).hexdigest(), 16)
        reward = (response_hash % 10) / 10.0
        return {
            "reward": reward,
            "format_reward": reward,
            "answer_reward": reward,
        }

    return dummy_reward_fn


@pytest.fixture
def num_rollout_responses():
    """Number of rollout responses for testing."""
    return 8


@pytest.fixture
def group_size(num_rollout_responses):
    """Group size for GRPO (number of responses per prompt)."""
    return int(num_rollout_responses / 2)


@pytest.fixture
def rollout_responses(num_rollout_responses):
    """Mock rollout responses for testing."""
    return [f"hmm I think ths answer is {i}" for i in range(num_rollout_responses)]


@pytest.fixture
def repeated_ground_truths(num_rollout_responses):
    """Mock ground truths repeated for each group."""
    return ["42"] * num_rollout_responses


@pytest.fixture
def advantage_eps():
    """Epsilon value for advantage computation."""
    return 1e-6


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 2


@pytest.fixture
def seq_length():
    """Sequence length for testing."""
    return 10


@pytest.fixture
def raw_rewards_or_advantages(batch_size):
    """Mock raw rewards or advantages."""
    torch.manual_seed(42)
    return torch.rand(size=(batch_size, 1))


@pytest.fixture
def policy_log_probs(batch_size, seq_length):
    """Mock policy log probabilities."""
    torch.manual_seed(42)
    return torch.randn(size=(batch_size, seq_length))


@pytest.fixture
def old_log_probs(policy_log_probs):
    """Mock old policy log probabilities (for GRPO-Clip)."""
    torch.manual_seed(42)
    return policy_log_probs + torch.randn_like(policy_log_probs)


@pytest.fixture
def advantages(raw_rewards_or_advantages):
    """Mock advantages (mean-centered)."""
    return raw_rewards_or_advantages - torch.mean(raw_rewards_or_advantages, dim=0)


@pytest.fixture
def raw_rewards(raw_rewards_or_advantages):
    """Mock raw rewards."""
    return raw_rewards_or_advantages


@pytest.fixture
def tensor(batch_size, seq_length):
    """Generic tensor for masked operations testing."""
    torch.manual_seed(42)
    return torch.randn(size=(batch_size, seq_length))


@pytest.fixture
def mask(tensor):
    """Mock mask for testing masked operations."""
    torch.manual_seed(42)
    return torch.rand_like(tensor) > 0.5


@pytest.fixture
def response_mask(policy_log_probs):
    """Mock response mask for testing."""
    torch.manual_seed(42)
    return torch.rand_like(policy_log_probs) > 0.5


@pytest.fixture
def gradient_accumulation_steps():
    """Number of gradient accumulation steps."""
    return 2


@pytest.fixture
def cliprange():
    """Clip range for GRPO-Clip."""
    return 0.1
