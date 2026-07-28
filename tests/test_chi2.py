"""Correctness tests for the four chi-squared implementations in project_benchmark.py.

Covers: cross-implementation agreement on random data, a hand-computed
2x2 table checked against scipy, missing-value handling, the two NaN edge
cases (category disappearing after pairwise deletion, only one category
present), and - when napypi is installed - direct numerical agreement with
the real NApy package.
"""

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from project_benchmark import (
    NAN_VALUE_CAT,
    chi2_dask,
    chi2_dask_numba,
    chi2_reference,
    chi2_vectorized,
    simulate_categorical_matrix,
)

try:
    import napypi as napy
except Exception:
    napy = None

ALL_IMPLS = {
    "python_ref": chi2_reference,
    "vectorized": chi2_vectorized,
    "dask": lambda data: chi2_dask(data, workload=4),
    "dask_numba": lambda data: chi2_dask_numba(data, workload=4),
}


@pytest.fixture(params=[(8, 60, 4, 0.1), (15, 40, 3, 0.2), (5, 25, 5, 0.0)])
def sim_data(request):
    """Simulated categorical matrix, parametrized over several (features, samples,
    categories, missing_ratio) combinations - including one with no missing
    values at all - so every test using this fixture runs three times.
    """
    n_features, n_samples, n_categories, missing_ratio = request.param
    rng = np.random.default_rng(123)
    return simulate_categorical_matrix(
        n_features, n_samples, n_categories=n_categories, missing_ratio=missing_ratio, rng=rng
    )


def test_all_implementations_agree(sim_data):
    """All four implementations must produce identical chi2/p-values on the same data."""
    results = {name: fn(sim_data) for name, fn in ALL_IMPLS.items()}
    ref = results["python_ref"]
    for name, res in results.items():
        np.testing.assert_allclose(
            res["chi2"], ref["chi2"], atol=1e-8, equal_nan=True, err_msg=f"{name} chi2 mismatch"
        )
        np.testing.assert_allclose(
            res["p_unadjusted"], ref["p_unadjusted"], atol=1e-8, equal_nan=True, err_msg=f"{name} p-value mismatch"
        )


def test_known_2x2_table_matches_scipy():
    """Independent ground truth: compare against scipy's chi2_contingency directly.

    Hand-built so the resulting 2x2 contingency table is exactly:
    [[2, 1], [1, 2]]
    """
    x = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
    data = np.stack([x, y])

    result = chi2_reference(data)
    table = np.array([[2, 1], [1, 2]])
    expected_chi2, expected_p, _, _ = chi2_contingency(table, correction=False)

    assert result["chi2"][0, 1] == pytest.approx(expected_chi2, abs=1e-10)
    assert result["p_unadjusted"][0, 1] == pytest.approx(expected_p, abs=1e-10)


def test_missing_values_are_pairwise_removed():
    """Missing entries must be excluded per-pair, not cause a crash or corrupt the result.

    Without missing values, x/y are perfectly correlated (chi2 large).
    Injecting NAN_VALUE_CAT into a few entries must not crash and must
    only use the remaining valid pairs.
    """
    x = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 3, dtype=np.int64)
    y = x.copy()
    y[0] = NAN_VALUE_CAT
    y[5] = NAN_VALUE_CAT
    data = np.stack([x, y])

    result = chi2_reference(data)
    assert np.isfinite(result["chi2"][0, 1])
    assert result["p_unadjusted"][0, 1] < 0.05


def test_category_disappearing_after_deletion_gives_nan():
    """A category present before missing-value removal but absent after must give NaN.

    Feature x has categories {0, 1, 2}, but every sample where x == 2
    has a missing y-value -> category 2 disappears after pairwise deletion.
    """
    x = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y = np.array([0, 1, 0, 1, NAN_VALUE_CAT, NAN_VALUE_CAT], dtype=np.int64)
    data = np.stack([x, y])

    for name, fn in ALL_IMPLS.items():
        result = fn(data)
        assert np.isnan(result["chi2"][0, 1]), f"{name} should return nan"
        assert np.isnan(result["p_unadjusted"][0, 1]), f"{name} should return nan"


def test_single_category_gives_nan():
    """A feature with only one distinct category makes the test undefined (NaN)."""
    x = np.zeros(20, dtype=np.int64)
    y = np.array([0, 1] * 10, dtype=np.int64)
    data = np.stack([x, y])

    for name, fn in ALL_IMPLS.items():
        result = fn(data)
        assert np.isnan(result["chi2"][0, 1]), f"{name} should return nan"


@pytest.mark.skipif(napy is None, reason="napypi not installed")
def test_matches_napy_ground_truth(sim_data):
    """Direct comparison against the real napypi package's chi_squared output.

    The diagonal is excluded because napy computes a real (large) self-vs-
    self statistic there, whereas this project's implementations leave the
    diagonal as NaN by convention (self-comparisons are not a meaningful
    pairwise test).
    """
    napy_result = napy.chi_squared(
        sim_data.astype(np.float64),
        nan_value=float(NAN_VALUE_CAT),
        axis=0,
        threads=1,
        check_data=False,
        return_types=["chi2", "p_unadjusted"],
    )
    n = sim_data.shape[0]
    off_diag = ~np.eye(n, dtype=bool)

    for name, fn in ALL_IMPLS.items():
        result = fn(sim_data)
        np.testing.assert_allclose(
            result["chi2"][off_diag],
            napy_result["chi2"][off_diag],
            atol=1e-6,
            equal_nan=True,
            err_msg=f"{name} chi2 does not match napy",
        )
