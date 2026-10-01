#!/usr/bin/env python3

"""
Tests for RLState._get_state_index stable state hashing.

The index must be reproducible across interpreter runs (independent
PYTHONHASHSEED) and must distinguish sign-swapped vectors, both of which
the previous positional/hash(str()) scheme violated.
"""

import subprocess
import sys

import numpy as np

from geo_infer_agent.models.rl import RLState


def _make_state(state_size: int = 512) -> RLState:
    return RLState(state_size=state_size, action_size=4)


PROBE_SCRIPT = """
import numpy as np
from geo_infer_agent.models.rl import RLState

state = RLState(state_size=512, action_size=4)
arr = np.array([0.5, -1.25, 2.0], dtype=np.float64)
print(state._get_state_index(arr))
print(state._get_state_index("sensor-a"))
"""


class TestRLStateIndex:
    """State-to-index hashing must be stable and collision-free by sign."""

    def test_same_array_maps_to_same_index(self) -> None:
        state = _make_state()
        arr = np.array([0.5, -1.25, 2.0])
        assert state._get_state_index(arr) == state._get_state_index(arr.copy())

    def test_integer_states_pass_through(self) -> None:
        state = _make_state()
        assert state._get_state_index(7) == 7

    def test_sign_swapped_vectors_get_different_indices(self) -> None:
        state = _make_state(state_size=1024)
        # Previously sum(i * val) made these two collide.
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 1.0])
        assert state._get_state_index(a) != state._get_state_index(b)

    def test_index_within_table_bounds(self) -> None:
        state = _make_state(state_size=64)
        for i in range(20):
            idx = state._get_state_index(np.full(4, i / 7.0))
            assert 0 <= idx < 64

    def test_reproducible_across_interpreter_seeds(self) -> None:
        # hash(str(...)) depended on PYTHONHASHSEED; the stable hash must not.
        env_base = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp"}
        results = []
        for seed in ("0", "1", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", PROBE_SCRIPT],
                capture_output=True,
                text=True,
                env={**env_base, "PYTHONHASHSEED": seed},
                timeout=120,
            )
            assert proc.returncode == 0, proc.stderr
            results.append(proc.stdout.strip())
        assert len(set(results)) == 1, results
