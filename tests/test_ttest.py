"""Correctness tests for the four t-test implementations in project_benchmark.py.

Covers: cross-implementation agreement on random data for both Student's
and Welch's t-test, a hand-computed comparison against scipy's ttest_ind,
missing-value handling, the NaN edge cases (too few samples, zero pooled
variance), and - when napypi is installed - direct numerical agreement with
the real NApy package.
"""

import numpy as np
import pytest
from scipy.stats import ttest_ind

from project_benchmark import (
    NAN_VALUE_CAT,
    NAN_VALUE_CONT,
    simulate_binary_continuous_matrices,
    ttest_dask,
    ttest_dask_numba,
    ttest_reference,
    ttest_vectorized,
)

try:
    import napypi as napy
except Exception:
    napy = None

# equal_var=True (Student's t-test) is the only mode all four configurations
# support -- dask_numba does not implement Welch's t-test (equal_var=False).
ALL_IMPLS_STUDENT = {
    "python_ref": lambda b, c: ttest_reference(b, c, equal_var=True),
    "vectorized": lambda b, c: ttest_vectorized(b, c, equal_var=True),
    "dask": lambda b, c: ttest_dask(b, c, equal_var=True, workload=4),
    "dask_numba": lambda b, c: ttest_dask_numba(b, c, workload=4),
}

# Welch's t-test is only implemented in the reference/vectorized/dask configs.
WELCH_IMPLS = {
    "python_ref": lambda b, c: ttest_reference(b, c, equal_var=False),
    "vectorized": lambda b, c: ttest_vectorized(b, c, equal_var=False),
    "dask": lambda b, c: ttest_dask(b, c, equal_var=False, workload=4),
}


@pytest.fixture(params=[(6, 6, 50, 0.1), (10, 8, 30, 0.2), (4, 4, 20, 0.0)])
def sim_data(request):
    """Simulated (binary, continuous) matrix pair, parametrized over several
    (n_bin, n_cont, n_samples, missing_ratio) combinations - including one
    with no missing values at all - so every test using this fixture runs
    three times.
    """
    n_bin, n_cont, n_samples, missing_ratio = request.param
    rng = np.random.default_rng(7)
    return simulate_binary_continuous_matrices(n_bin, n_cont, n_samples, missing_ratio=missing_ratio, rng=rng)


def test_all_implementations_agree_student(sim_data):
    """All four implementations must agree on Student's t-test (equal_var=True)."""
    bin_data, cont_data = sim_data
    results = {name: fn(bin_data, cont_data) for name, fn in ALL_IMPLS_STUDENT.items()}
    ref = results["python_ref"]
    for name, res in results.items():
        np.testing.assert_allclose(res["t"], ref["t"], atol=1e-8, equal_nan=True, err_msg=f"{name} t mismatch")
        np.testing.assert_allclose(
            res["p_unadjusted"], ref["p_unadjusted"], atol=1e-8, equal_nan=True, err_msg=f"{name} p mismatch"
        )


def test_welch_implementations_agree(sim_data):
    """The three Welch-capable implementations must agree with each other (equal_var=False).

    dask_numba is excluded here since it only implements Student's t-test
    (see the ALL_IMPLS_STUDENT / WELCH_IMPLS split above).
    """
    bin_data, cont_data = sim_data
    results = {name: fn(bin_data, cont_data) for name, fn in WELCH_IMPLS.items()}
    ref = results["python_ref"]
    for name, res in results.items():
        np.testing.assert_allclose(res["t"], ref["t"], atol=1e-8, equal_nan=True, err_msg=f"{name} Welch t mismatch")


def test_known_groups_match_scipy():
    """Independent ground truth: compare both Student's and Welch's t-test against scipy's ttest_ind."""
    rng = np.random.default_rng(0)
    group0 = rng.normal(0.0, 1.0, size=15)
    group1 = rng.normal(1.5, 1.0, size=18)

    x_bin = np.array([0] * 15 + [1] * 18, dtype=np.int64)
    y_cont = np.concatenate([group0, group1])
    data = np.stack([x_bin])
    cont = np.stack([y_cont])

    result = ttest_reference(data, cont, equal_var=True)
    expected = ttest_ind(group0, group1, equal_var=True)

    assert result["t"][0, 0] == pytest.approx(expected.statistic, abs=1e-10)
    assert result["p_unadjusted"][0, 0] == pytest.approx(expected.pvalue, abs=1e-10)

    result_welch = ttest_reference(data, cont, equal_var=False)
    expected_welch = ttest_ind(group0, group1, equal_var=False)
    assert result_welch["t"][0, 0] == pytest.approx(expected_welch.statistic, abs=1e-10)
    assert result_welch["p_unadjusted"][0, 0] == pytest.approx(expected_welch.pvalue, abs=1e-10)


def test_missing_values_are_pairwise_removed():
    """Missing entries must be excluded per-sample, not cause a crash or corrupt the result."""
    rng = np.random.default_rng(1)
    x_bin = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    y_cont = np.concatenate([rng.normal(0, 1, 10), rng.normal(3, 1, 10)])
    y_cont[0] = NAN_VALUE_CONT
    y_cont[15] = NAN_VALUE_CONT
    data = np.stack([x_bin])
    cont = np.stack([y_cont])

    result = ttest_reference(data, cont)
    assert np.isfinite(result["t"][0, 0])
    assert result["p_unadjusted"][0, 0] < 0.01


def test_too_few_samples_gives_nan():
    """Fewer than 2 valid samples in either group makes the t-test undefined (NaN)."""
    x_bin = np.array([0, 1, 0, 1], dtype=np.int64)
    y_cont = np.array([1.0, 2.0, NAN_VALUE_CONT, NAN_VALUE_CONT])
    data = np.stack([x_bin])
    cont = np.stack([y_cont])

    for name, fn in ALL_IMPLS_STUDENT.items():
        result = fn(data, cont)
        assert np.isnan(result["t"][0, 0]), f"{name} should return nan for too few samples"


def test_zero_pooled_variance_gives_nan():
    """A pooled variance of zero (both groups constant) makes the t-statistic undefined (NaN)."""
    x_bin = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_cont = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    data = np.stack([x_bin])
    cont = np.stack([y_cont])

    for name, fn in ALL_IMPLS_STUDENT.items():
        result = fn(data, cont)
        assert np.isnan(result["t"][0, 0]), f"{name} should return nan for zero pooled variance"


@pytest.mark.skipif(napy is None, reason="napypi not installed")
def test_matches_napy_ground_truth(sim_data):
    """Direct comparison against the real napypi package's ttest output (Student's t-test)."""
    bin_data, cont_data = sim_data
    bin_for_napy = bin_data.astype(np.float64)
    bin_for_napy[bin_for_napy == NAN_VALUE_CAT] = NAN_VALUE_CONT

    napy_result = napy.ttest(
        bin_for_napy,
        cont_data,
        nan_value=NAN_VALUE_CONT,
        axis=0,
        threads=1,
        check_data=False,
        return_types=["t", "p_unadjusted"],
        equal_var=True,
    )

    for name, fn in ALL_IMPLS_STUDENT.items():
        result = fn(bin_data, cont_data)
        np.testing.assert_allclose(
            result["t"], napy_result["t"], atol=1e-6, equal_nan=True, err_msg=f"{name} t does not match napy"
        )
