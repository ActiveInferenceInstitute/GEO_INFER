"""Analytical permutation scores and caller-owned process boundaries."""

import json
import os
import sys

import numpy as np

from geo_infer_test.process import run_process


def test_permutation_scores_preserve_linear_reference_without_child_pools(tmp_path):
    # Observe children while the caller is alive: resource trackers may exit
    # only after their parent, making an exit-only check platform-dependent.
    code = """
import json
import numpy as np
import psutil
from sklearn.linear_model import LinearRegression
from geo_infer_ai.core.explainability import ModelExplainer
X = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,0.],[0.,2.]])
y = 3 * X[:,0] + X[:,1]
model = LinearRegression().fit(X,y)
before = {child.pid for child in psutil.Process().children(recursive=True)}
values = ModelExplainer(model).calculate_feature_importance(X,y)
after = {child.pid for child in psutil.Process().children(recursive=True)}
assert after <= before, 'permutation importance retained child processes'
print(json.dumps(values), flush=True)
"""
    result = run_process(
        [sys.executable, "-W", "error", "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "LOKY_MAX_CPU_COUNT": "2",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        timeout=60,
        check=True,
    )
    actual = json.loads(result.stdout)
    # Independent R-squared loss from the known coefficients, rather than
    # invoking the estimator or sklearn's importance/scoring implementation.
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
    )
    coefficients = np.array([3.0, 1.0])
    y = X @ coefficients
    denominator = np.sum((y - np.mean(y)) ** 2)
    seed = np.random.RandomState(42).randint(np.iinfo(np.int32).max + 1)
    expected = []
    for feature in range(2):
        rng = np.random.RandomState(seed)
        shuffled = X.copy()
        indices = np.arange(len(X))
        losses = []
        for _ in range(10):
            rng.shuffle(indices)
            shuffled[:, feature] = shuffled[indices, feature]
            losses.append(np.sum((y - shuffled @ coefficients) ** 2) / denominator)
        expected.append(np.mean(losses))
    np.testing.assert_allclose(
        [actual["feature_0"], actual["feature_1"]], expected, rtol=0, atol=1e-14
    )
