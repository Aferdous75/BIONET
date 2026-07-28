
"""
Project benchmark for chi-squared test and t-test:
- Python reference implementation
- vectorized NumPy version
- pure Dask-parallelized version
- Dask + Numba parallelized version
- comparison against NApy

Usage examples
--------------
python project_benchmark.py --test chi2 --mode smoke
python project_benchmark.py --test ttest --mode smoke
python project_benchmark.py --test all --mode benchmark
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask import compute, delayed
from distributed import Client, LocalCluster
from numba import njit
from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as t_dist

try:
    import napypi as napy
except Exception:
    napy = None


# ----------------------------
# Configuration
# ----------------------------

NAN_VALUE_CAT = -1
NAN_VALUE_CONT = -999.0
RNG_SEED = 42


@dataclass
class BenchmarkResult:
    """One timed measurement for a single (test, implementation, problem size) combination.

    Attributes:
        test_name: "chi2" or "ttest".
        implementation: one of "python_ref", "vectorized", "dask", "dask_numba", "napy".
        features: number of features in the simulated data for this run.
        samples: number of samples (columns) in the simulated data for this run.
        missing_ratio: fraction of entries replaced by the missing-value sentinel.
        workers: number of Dask worker processes used for this run.
        workload: number of feature pairs dispatched per Dask task.
        runtime_sec: measured wall-clock runtime in seconds (see time_call).
        sweep: which benchmark sweep this row belongs to - "features", "samples",
            or "workload" - set by main() after the row is produced.
    """

    test_name: str
    implementation: str
    features: int
    samples: int
    missing_ratio: float
    workers: int
    workload: int
    runtime_sec: float
    sweep: str = ""


# ----------------------------
# Data simulation
# ----------------------------

def simulate_categorical_matrix(
    n_features: int,
    n_samples: int,
    n_categories: int = 4,
    missing_ratio: float = 0.1,
    nan_value: int = NAN_VALUE_CAT,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a feature x sample matrix of label-encoded categorical data.

    Each entry is an independently drawn category label in [0, n_categories),
    with a random subset of entries replaced by `nan_value` to represent
    missing data (matching NApy's sentinel-value convention).

    Parameters:
        n_features: number of rows (features) to generate.
        n_samples: number of columns (samples) to generate.
        n_categories: number of distinct category labels per feature (0..n_categories-1).
        missing_ratio: fraction of entries to replace with `nan_value`, in [0, 1].
        nan_value: sentinel used to mark a missing entry.
        rng: optional numpy Generator for reproducible simulation; a fresh
            unseeded Generator is created if omitted.

    Returns:
        An (n_features, n_samples) int64 array of category labels, with
        `nan_value` standing in for missing entries.
    """
    rng = np.random.default_rng() if rng is None else rng
    data = rng.integers(0, n_categories, size=(n_features, n_samples), dtype=np.int64)
    if missing_ratio > 0:
        # Independently mask a `missing_ratio` fraction of entries; using a
        # boolean mask (rather than sampling missing indices directly) keeps
        # this O(n_features * n_samples) and vectorized.
        miss = rng.random(size=data.shape) < missing_ratio
        data = data.copy()
        data[miss] = nan_value
    return data


def simulate_binary_continuous_matrices(
    n_bin_features: int,
    n_cont_features: int,
    n_samples: int,
    missing_ratio: float = 0.1,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate paired binary and continuous matrices for the t-test benchmark.

    Generates one (n_bin_features, n_samples) matrix of binary group labels
    (0/1) and one (n_cont_features, n_samples) matrix of standard-normal
    continuous values, each with independently injected missing entries.

    Parameters:
        n_bin_features: number of binary (grouping) features to generate.
        n_cont_features: number of continuous features to generate.
        n_samples: number of samples (columns) shared by both matrices.
        missing_ratio: fraction of entries to replace with the relevant
            sentinel, in [0, 1].
        nan_bin: sentinel used to mark a missing entry in `bin_data`.
        nan_cont: sentinel used to mark a missing entry in `cont_data`.
        rng: optional numpy Generator for reproducible simulation; a fresh
            unseeded Generator is created if omitted.

    Returns:
        A tuple (bin_data, cont_data): the binary-group matrix (int64) and
        the continuous-value matrix (float64), with missing entries marked
        by nan_bin / nan_cont respectively.
    """
    rng = np.random.default_rng() if rng is None else rng
    bin_data = rng.integers(0, 2, size=(n_bin_features, n_samples), dtype=np.int64)
    cont_data = rng.normal(loc=0.0, scale=1.0, size=(n_cont_features, n_samples)).astype(np.float64)

    if missing_ratio > 0:
        miss_bin = rng.random(size=bin_data.shape) < missing_ratio
        miss_cont = rng.random(size=cont_data.shape) < missing_ratio
        bin_data = bin_data.copy()
        cont_data = cont_data.copy()
        bin_data[miss_bin] = nan_bin
        cont_data[miss_cont] = nan_cont
    return bin_data, cont_data


# ----------------------------
# Shared helpers
# ----------------------------

def upper_triangle_pairs(n_features: int) -> list[tuple[int, int]]:
    """List all unordered feature-index pairs (i, j) with i < j.

    Used for the chi-squared test, where the result matrix is symmetric
    (feature vs. itself is not tested) so only the upper triangle of the
    n_features x n_features grid needs to be computed.

    Parameters:
        n_features: number of features (matrix will be n_features x n_features).

    Returns:
        A list of (i, j) index tuples with 0 <= i < j < n_features.
    """
    return list(combinations(range(n_features), 2))


def rectangular_pairs(n_left: int, n_right: int) -> list[tuple[int, int]]:
    """List all (i, j) index pairs for a rectangular grid of two feature sets.

    Used for the t-test, where every binary feature is tested against every
    continuous feature, so (unlike chi-squared) the full n_left x n_right
    grid is needed rather than just its upper triangle.

    Parameters:
        n_left: number of features on the "left" axis (e.g. binary features).
        n_right: number of features on the "right" axis (e.g. continuous features).

    Returns:
        A list of all (i, j) index tuples with 0 <= i < n_left, 0 <= j < n_right.
    """
    return [(i, j) for i in range(n_left) for j in range(n_right)]


def chunked(seq: list[tuple[int, int]], chunk_size: int) -> list[list[tuple[int, int]]]:
    """Split a list of feature pairs into fixed-size chunks.

    This is the "workload" knob from Task 3: each chunk becomes one Dask
    task, so `chunk_size` controls how many pairwise tests are computed per
    task. Small chunks mean more tasks and more Dask scheduling/IPC overhead
    per unit of work; large chunks mean fewer, coarser tasks with less
    overhead but also less opportunity for the scheduler to load-balance
    across workers.

    Parameters:
        seq: the full list of feature-index pairs to split.
        chunk_size: maximum number of pairs per chunk (the last chunk may be
            smaller if len(seq) is not a multiple of chunk_size).

    Returns:
        A list of chunks (each a list of pairs) covering all of `seq` in order.
    """
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


# ----------------------------
# Chi-squared implementations
# ----------------------------

def chi2_one_pair_python(x: np.ndarray, y: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> tuple[float, float]:
    """Chi-squared test of independence for one pair of categorical features.

    Pure-Python (no NumPy vectorization) reference implementation. Builds
    the contingency table for (x, y) after pairwise missing-value removal,
    then computes Pearson's chi-squared statistic:

        chi2 = sum_ij (observed_ij - expected_ij)^2 / expected_ij

    where expected_ij = row_sum_i * col_sum_j / total is the count expected
    under the null hypothesis that x and y are independent, and degrees of
    freedom = (n_categories_x - 1) * (n_categories_y - 1). The p-value is
    the upper-tail (survival function) probability of the chi-squared
    distribution at that statistic and dof.

    Parameters:
        x: 1D array of category labels for the first feature (one entry per sample).
        y: 1D array of category labels for the second feature, same length as x.
        nan_value: sentinel marking a missing entry in either x or y.

    Returns:
        (chi2, p_unadjusted): the test statistic and its p-value (upper-tail
        probability under the chi-squared distribution). Both are NaN if the
        test is undefined - e.g. fewer than two categories remain in x or y,
        no sample pairs survive missing-value removal, a category present
        before pairwise deletion vanishes afterward, or an expected cell
        count is zero.
    """
    valid_pairs: list[tuple[int, int]] = []
    categories_x = set()
    categories_y = set()

    # Single pass: record every category seen in x/y (before deletion) and
    # collect the subset of (a, b) pairs where both entries are non-missing
    # (this is the pairwise-deletion set I(g, h) from the NApy paper).
    for a, b in zip(x, y):
        if a != nan_value:
            categories_x.add(int(a))
        if b != nan_value:
            categories_y.add(int(b))
        if a != nan_value and b != nan_value:
            valid_pairs.append((int(a), int(b)))

    if len(valid_pairs) == 0 or len(categories_x) < 2 or len(categories_y) < 2:
        return np.nan, np.nan

    present_x = {a for a, _ in valid_pairs}
    present_y = {b for _, b in valid_pairs}

    # Follow the paper's spirit: if a previously existing category disappears after pairwise deletion, return nan.
    if present_x != categories_x or present_y != categories_y:
        return np.nan, np.nan

    # Build the contingency table: rows = categories of x, columns = categories of y.
    n_cat_x = max(categories_x) + 1
    n_cat_y = max(categories_y) + 1
    contingency = [[0 for _ in range(n_cat_y)] for _ in range(n_cat_x)]

    for a, b in valid_pairs:
        contingency[a][b] += 1

    row_sums = [sum(row) for row in contingency]
    col_sums = [sum(contingency[i][j] for i in range(n_cat_x)) for j in range(n_cat_y)]
    total = sum(row_sums)

    if total == 0:
        return np.nan, np.nan

    # Pearson's chi-squared statistic: sum over all cells of
    # (observed - expected)^2 / expected, where expected assumes independence.
    chi2 = 0.0
    for i in range(n_cat_x):
        for j in range(n_cat_y):
            expected = row_sums[i] * col_sums[j] / total
            if expected == 0:
                return np.nan, np.nan
            chi2 += (contingency[i][j] - expected) ** 2 / expected

    dof = (n_cat_x - 1) * (n_cat_y - 1)
    p = float(chi2_dist.sf(chi2, dof))
    return chi2, p


def chi2_reference(data: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> dict[str, np.ndarray]:
    """Pairwise chi-squared test across all feature pairs (pure-Python reference).

    Sequentially calls chi2_one_pair_python for every pair in the upper
    triangle of the feature x feature grid, mirroring each result into both
    (i, j) and (j, i) since the test is symmetric. The diagonal is left as
    NaN (self-vs-self is not tested).

    Parameters:
        data: (n_features, n_samples) matrix of category labels.
        nan_value: sentinel marking a missing entry.

    Returns:
        Dict with "chi2" and "p_unadjusted" keys, each an (n_features,
        n_features) array (NaN on the diagonal and for undefined pairs).
    """
    n_features = data.shape[0]
    stat = np.full((n_features, n_features), np.nan, dtype=np.float64)
    pvals = np.full((n_features, n_features), np.nan, dtype=np.float64)

    for i, j in upper_triangle_pairs(n_features):
        s, p = chi2_one_pair_python(data[i], data[j], nan_value=nan_value)
        stat[i, j] = stat[j, i] = s
        pvals[i, j] = pvals[j, i] = p
    return {"chi2": stat, "p_unadjusted": pvals}


def chi2_one_pair_vectorized(x: np.ndarray, y: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> tuple[float, float]:
    """Chi-squared test for one feature pair, using NumPy array operations.

    Numerically and logically equivalent to chi2_one_pair_python, but the
    contingency table is built with a bincount trick instead of Python
    loops: each (a, b) category pair is mapped to a single flat index
    `a * n_cat_y + b`, so np.bincount can tally all pairs at once (this is
    the standard "flatten a 2D histogram into a 1D one" trick). The
    row/column sums and expected-frequency matrix are then computed with
    broadcasting instead of nested loops.

    Parameters:
        x: 1D array of category labels for the first feature.
        y: 1D array of category labels for the second feature, same length as x.
        nan_value: sentinel marking a missing entry in either x or y.

    Returns:
        (chi2, p_unadjusted), NaN under the same undefined conditions as
        chi2_one_pair_python.
    """
    non_missing_x = x[x != nan_value]
    non_missing_y = y[y != nan_value]
    categories_x = np.unique(non_missing_x)
    categories_y = np.unique(non_missing_y)

    if categories_x.size < 2 or categories_y.size < 2:
        return np.nan, np.nan

    valid = (x != nan_value) & (y != nan_value)
    if valid.sum() == 0:
        return np.nan, np.nan

    xv = x[valid].astype(np.int64)
    yv = y[valid].astype(np.int64)

    # A category present before pairwise deletion must still be present
    # afterward (see chi2_one_pair_python), otherwise the test is undefined.
    if np.unique(xv).size != categories_x.size or np.unique(yv).size != categories_y.size:
        return np.nan, np.nan

    n_cat_x = int(categories_x.max()) + 1
    n_cat_y = int(categories_y.max()) + 1

    # Flatten the 2D (x-category, y-category) contingency table into 1D
    # indices so a single bincount call can tally all observed pairs.
    flat = np.bincount(xv * n_cat_y + yv, minlength=n_cat_x * n_cat_y)
    contingency = flat.reshape(n_cat_x, n_cat_y).astype(np.float64)

    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    total = contingency.sum()

    if total == 0:
        return np.nan, np.nan

    # Expected counts under independence: row_sums (n_cat_x, 1) times
    # col_sums (1, n_cat_y) broadcasts to the full (n_cat_x, n_cat_y) table.
    expected = row_sums @ col_sums / total
    if np.any(expected == 0):
        return np.nan, np.nan

    chi2 = float(((contingency - expected) ** 2 / expected).sum())
    dof = (n_cat_x - 1) * (n_cat_y - 1)
    p = float(chi2_dist.sf(chi2, dof))
    return chi2, p


def chi2_vectorized(data: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> dict[str, np.ndarray]:
    """Pairwise chi-squared test across all feature pairs (vectorized single-pair kernel).

    Same pair-iteration structure as chi2_reference, but each individual
    pair is computed with chi2_one_pair_vectorized instead of the pure-Python
    version - i.e. the speedup here comes from NumPy vectorizing the *inner*
    per-pair computation, not from parallelizing across pairs.

    Parameters:
        data: (n_features, n_samples) matrix of category labels.
        nan_value: sentinel marking a missing entry.

    Returns:
        Dict with "chi2" and "p_unadjusted" keys, each an (n_features,
        n_features) array (NaN on the diagonal and for undefined pairs).
    """
    n_features = data.shape[0]
    stat = np.full((n_features, n_features), np.nan, dtype=np.float64)
    pvals = np.full((n_features, n_features), np.nan, dtype=np.float64)

    for i, j in upper_triangle_pairs(n_features):
        s, p = chi2_one_pair_vectorized(data[i], data[j], nan_value=nan_value)
        stat[i, j] = stat[j, i] = s
        pvals[i, j] = pvals[j, i] = p
    return {"chi2": stat, "p_unadjusted": pvals}


def _chi2_chunk_python(data: np.ndarray, pair_chunk: list[tuple[int, int]], nan_value: int) -> list[tuple[int, int, float, float]]:
    """Compute the chi-squared test for one chunk of feature pairs (Dask task body).

    This is the function each individual Dask task actually runs: it loops
    over its assigned pairs and calls the vectorized single-pair kernel
    (chi2_one_pair_vectorized) for each. Parallelism comes from Dask running
    many chunks concurrently across worker processes, not from anything
    inside this function.

    Parameters:
        data: (n_features, n_samples) matrix of category labels (shared
            read-only by every task).
        pair_chunk: list of (i, j) feature-index pairs assigned to this task.
        nan_value: sentinel marking a missing entry.

    Returns:
        A list of (i, j, chi2, p_unadjusted) tuples, one per pair in the chunk.
    """
    out = []
    for i, j in pair_chunk:
        s, p = chi2_one_pair_vectorized(data[i], data[j], nan_value=nan_value)
        out.append((i, j, s, p))
    return out


def chi2_dask(
    data: np.ndarray,
    nan_value: int = NAN_VALUE_CAT,
    workload: int = 50,
) -> dict[str, np.ndarray]:
    """Pairwise chi-squared test parallelized across feature-pair chunks with Dask.

    The full list of feature pairs is split into chunks of size `workload`
    (see chunked()), and each chunk is wrapped in a `dask.delayed` call so it
    becomes one task in Dask's task graph. `compute(*tasks)` then submits
    all tasks to the active Dask scheduler/cluster and blocks until every
    chunk has been computed (by default across the worker processes of the
    LocalCluster set up in main()). Results from all chunks are merged back
    into the symmetric (n_features, n_features) output matrices afterward.

    Parameters:
        data: (n_features, n_samples) matrix of category labels.
        nan_value: sentinel marking a missing entry.
        workload: number of feature pairs per Dask task (task granularity -
            see Task 3 in the benchmark: too small and per-task scheduling
            overhead dominates, too large and there is less opportunity to
            balance work across workers).

    Returns:
        Dict with "chi2" and "p_unadjusted" keys, each an (n_features,
        n_features) array (NaN on the diagonal and for undefined pairs).
    """
    n_features = data.shape[0]
    stat = np.full((n_features, n_features), np.nan, dtype=np.float64)
    pvals = np.full((n_features, n_features), np.nan, dtype=np.float64)

    pairs = upper_triangle_pairs(n_features)
    tasks = [delayed(_chi2_chunk_python)(data, chunk, nan_value) for chunk in chunked(pairs, workload)]
    results = compute(*tasks)

    for chunk_result in results:
        for i, j, s, p in chunk_result:
            stat[i, j] = stat[j, i] = s
            pvals[i, j] = pvals[j, i] = p
    return {"chi2": stat, "p_unadjusted": pvals}


@njit(cache=True)
def _chi2_numba_pair(x: np.ndarray, y: np.ndarray, nan_value: int) -> tuple[float, int, int]:
    """Numba-JIT-compiled chi-squared test for one feature pair.

    Computes the same statistic as chi2_one_pair_python /
    chi2_one_pair_vectorized, but written so it can be compiled by Numba's
    `nopython` mode: no Python sets, lists, or dicts are used (Numba does
    not support them in nopython mode), only fixed-size NumPy arrays and
    plain loops. Because of that constraint, the category-presence tracking
    that chi2_one_pair_python does with `set()` is instead done with
    "seen" arrays indexed by category label, and the algorithm runs in two
    explicit passes over the samples instead of one:

      Pass 1: find the largest category label in x and in y (so we know how
              big the contingency table needs to be) and check whether any
              valid (non-missing) pairs exist at all.
      Pass 2: build the contingency table itself and, at the same time,
              track which categories actually survive pairwise deletion
              (`seen_x`/`seen_y`), to detect a category disappearing.

    A category label is assumed to range over 0..max_label (as required by
    NApy's label-encoding convention), so `max_x + 1` / `max_y + 1` gives
    the number of categories directly - this is also why category labels
    must be non-negative integers.

    This function returns dof and a success flag instead of a p-value,
    because scipy's chi2 survival function is not Numba-compatible; the
    caller (_chi2_chunk_numba) computes the p-value afterward in plain Python.

    Parameters:
        x: 1D array of category labels for the first feature.
        y: 1D array of category labels for the second feature, same length as x.
        nan_value: sentinel marking a missing entry in either x or y.

    Returns:
        (chi2, dof, ok): the chi-squared statistic, its degrees of freedom,
        and 1 if the test is well-defined or 0 if it is not (in which case
        chi2 is NaN and dof is 0).
    """
    max_x = -1
    max_y = -1
    total_nonmiss_x = 0
    total_nonmiss_y = 0
    valid_count = 0

    # Pass 1: largest category label seen in x/y (defines table size) and
    # whether any sample has both entries non-missing.
    for k in range(x.shape[0]):
        a = int(x[k])
        b = int(y[k])
        if a != nan_value:
            total_nonmiss_x += 1
            if a > max_x:
                max_x = a
        if b != nan_value:
            total_nonmiss_y += 1
            if b > max_y:
                max_y = b
        if a != nan_value and b != nan_value:
            valid_count += 1

    if max_x < 1 or max_y < 1 or valid_count == 0:
        return np.nan, 0, 0

    n_cat_x = max_x + 1
    n_cat_y = max_y + 1
    contingency = np.zeros((n_cat_x, n_cat_y), dtype=np.int64)
    # seen_x[c] / seen_y[c] act as a fixed-size stand-in for a Python set:
    # they flip to 1 the first time category c is observed in a valid pair.
    seen_x = np.zeros(n_cat_x, dtype=np.int64)
    seen_y = np.zeros(n_cat_y, dtype=np.int64)
    present_x_count = 0
    present_y_count = 0

    # Pass 2: build the contingency table from valid pairs, and count how
    # many distinct categories actually survived pairwise deletion.
    for k in range(x.shape[0]):
        a = int(x[k])
        b = int(y[k])
        if a != nan_value and b != nan_value:
            contingency[a, b] += 1
            if seen_x[a] == 0:
                seen_x[a] = 1
                present_x_count += 1
            if seen_y[b] == 0:
                seen_y[b] = 1
                present_y_count += 1

    # category disappeared after pairwise deletion
    if present_x_count != n_cat_x or present_y_count != n_cat_y:
        return np.nan, 0, 0

    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    total = contingency.sum()

    if total == 0:
        return np.nan, 0, 0

    # Pearson's chi-squared statistic (same formula as the other implementations).
    chi2 = 0.0
    for i in range(n_cat_x):
        for j in range(n_cat_y):
            expected = row_sums[i] * col_sums[j] / total
            if expected == 0:
                return np.nan, 0, 0
            diff = contingency[i, j] - expected
            chi2 += diff * diff / expected

    dof = (n_cat_x - 1) * (n_cat_y - 1)
    return chi2, dof, 1


def _chi2_chunk_numba(data: np.ndarray, pair_chunk: list[tuple[int, int]], nan_value: int) -> list[tuple[int, int, float, float]]:
    """Compute the chi-squared test for one chunk of feature pairs, using the Numba kernel.

    Dask-task body for the Dask + Numba configuration: loops over the
    assigned pairs, calls the JIT-compiled _chi2_numba_pair for the
    statistic and dof, then computes the p-value here in plain Python
    (scipy's chi2_dist.sf is not Numba-compatible, so it can't live inside
    the @njit function itself).

    Parameters:
        data: (n_features, n_samples) matrix of category labels.
        pair_chunk: list of (i, j) feature-index pairs assigned to this task.
        nan_value: sentinel marking a missing entry.

    Returns:
        A list of (i, j, chi2, p_unadjusted) tuples, one per pair in the chunk.
    """
    out: list[tuple[int, int, float, float]] = []
    for i, j in pair_chunk:
        s, dof, ok = _chi2_numba_pair(data[i], data[j], nan_value)
        p = np.nan if ok == 0 else float(chi2_dist.sf(s, dof))
        out.append((i, j, s, p))
    return out


def chi2_dask_numba(
    data: np.ndarray,
    nan_value: int = NAN_VALUE_CAT,
    workload: int = 50,
) -> dict[str, np.ndarray]:
    """Pairwise chi-squared test parallelized with Dask, using the Numba-JIT per-pair kernel.

    Identical task/chunking structure to chi2_dask (see its docstring for
    how `workload` controls chunk size and how dask.delayed/compute drive
    the parallelism); the only difference is that each task runs
    _chi2_chunk_numba (JIT-compiled per-pair math) instead of
    _chi2_chunk_python (NumPy-vectorized per-pair math).

    Parameters:
        data: (n_features, n_samples) matrix of category labels.
        nan_value: sentinel marking a missing entry.
        workload: number of feature pairs per Dask task.

    Returns:
        Dict with "chi2" and "p_unadjusted" keys, each an (n_features,
        n_features) array (NaN on the diagonal and for undefined pairs).
    """
    n_features = data.shape[0]
    stat = np.full((n_features, n_features), np.nan, dtype=np.float64)
    pvals = np.full((n_features, n_features), np.nan, dtype=np.float64)

    pairs = upper_triangle_pairs(n_features)
    tasks = [delayed(_chi2_chunk_numba)(data, chunk, nan_value) for chunk in chunked(pairs, workload)]
    results = compute(*tasks)

    for chunk_result in results:
        for i, j, s, p in chunk_result:
            stat[i, j] = stat[j, i] = s
            pvals[i, j] = pvals[j, i] = p
    return {"chi2": stat, "p_unadjusted": pvals}


def chi2_napy(data: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> dict[str, np.ndarray]:
    """Pairwise chi-squared test via the real NApy package (ground-truth comparison).

    Thin wrapper around napy.chi_squared, used as the performance baseline
    that the four hand-written implementations above are benchmarked
    against (and, in tests/test_chi2.py, checked for numerical agreement with).

    Parameters:
        data: (n_features, n_samples) matrix of category labels.
        nan_value: sentinel marking a missing entry.

    Returns:
        Dict with "chi2" and "p_unadjusted" keys, as returned by napy.chi_squared.

    Raises:
        ImportError: if the napypi package is not installed.
    """
    if napy is None:
        raise ImportError("napypi is not installed.")
    return napy.chi_squared(
        data,
        nan_value=float(nan_value),
        axis=0,
        threads=1,
        check_data=False,
        return_types=["chi2", "p_unadjusted"],
    )


# ----------------------------
# t-test implementations
# ----------------------------

def ttest_one_pair_python(
    x_bin: np.ndarray,
    y_cont: np.ndarray,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
    equal_var: bool = True,
) -> tuple[float, float]:
    """Two-sample t-test for one (binary feature, continuous feature) pair.

    Pure-Python reference implementation. Splits the continuous values into
    two groups by the binary feature (group0 where x_bin == 0, group1 where
    x_bin == 1) after pairwise missing-value removal, then computes either:

    - Student's t-test (equal_var=True): assumes both groups share the same
      population variance, so a single pooled variance estimate is used:
          pooled_var = ((n0-1)*v0 + (n1-1)*v1) / (n0+n1-2)
          se = sqrt(pooled_var * (1/n0 + 1/n1))
          t = (mean0 - mean1) / se,  dof = n0 + n1 - 2

    - Welch's t-test (equal_var=False): does not assume equal variances, so
      each group's variance is used separately:
          se = sqrt(v0/n0 + v1/n1)
          t = (mean0 - mean1) / se
      and the degrees of freedom are estimated with the Welch-Satterthwaite
      equation, dof = se^4 / (v0/n0)^2/(n0-1) + (v1/n1)^2/(n1-1)).

    Either way the p-value is the two-sided probability under Student's
    t-distribution at that statistic and dof.

    Parameters:
        x_bin: 1D array of binary group labels (0 or 1), one per sample.
        y_cont: 1D array of continuous values, same length as x_bin.
        nan_bin: sentinel marking a missing entry in x_bin.
        nan_cont: sentinel marking a missing entry in y_cont.
        equal_var: True for Student's pooled-variance t-test, False for Welch's.

    Returns:
        (t, p_unadjusted), both NaN if undefined - e.g. a group label other
        than 0/1, fewer than 2 samples in either group, fewer than 3 samples
        total, or (Student's test only) a pooled variance of zero.
    """
    group0 = []
    group1 = []

    # Pairwise-delete samples with a missing entry in either matrix, then
    # split the remaining continuous values into the two groups defined by
    # the binary feature.
    for a, b in zip(x_bin, y_cont):
        if a == nan_bin or b == nan_cont:
            continue
        if int(a) == 0:
            group0.append(float(b))
        elif int(a) == 1:
            group1.append(float(b))
        else:
            return np.nan, np.nan

    n0 = len(group0)
    n1 = len(group1)

    if n0 < 2 or n1 < 2 or (n0 + n1) < 3:
        return np.nan, np.nan

    m0 = sum(group0) / n0
    m1 = sum(group1) / n1
    v0 = sum((z - m0) ** 2 for z in group0) / (n0 - 1)
    v1 = sum((z - m1) ** 2 for z in group1) / (n1 - 1)

    if equal_var:
        # Student's t-test: pool both groups' variance into one estimate,
        # weighted by degrees of freedom (n0-1 and n1-1).
        pooled_num = (n0 - 1) * v0 + (n1 - 1) * v1
        pooled_den = n0 + n1 - 2
        if pooled_den <= 0:
            return np.nan, np.nan
        pooled_var = pooled_num / pooled_den
        if pooled_var <= 0:
            return np.nan, np.nan
        se = math.sqrt(pooled_var * (1 / n0 + 1 / n1))
        if se == 0:
            return np.nan, np.nan
        tval = (m0 - m1) / se
        dof = n0 + n1 - 2
    else:
        # Welch's t-test: keep each group's variance separate (no equal-
        # variance assumption) and estimate dof via Welch-Satterthwaite.
        se2 = v0 / n0 + v1 / n1
        if se2 <= 0:
            return np.nan, np.nan
        tval = (m0 - m1) / math.sqrt(se2)
        num = se2 ** 2
        den = ((v0 / n0) ** 2) / (n0 - 1) + ((v1 / n1) ** 2) / (n1 - 1)
        if den == 0:
            return np.nan, np.nan
        dof = num / den

    p = float(2.0 * t_dist.sf(abs(tval), dof))
    return tval, p


def ttest_reference(
    bin_data: np.ndarray,
    cont_data: np.ndarray,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
    equal_var: bool = True,
) -> dict[str, np.ndarray]:
    """Pairwise t-test across every (binary, continuous) feature pair (pure-Python reference).

    Unlike chi-squared, the two inputs are different feature sets (binary
    grouping features vs. continuous measurement features), so every pair
    in the full n_bin x n_cont grid is tested (see rectangular_pairs) rather
    than just an upper triangle.

    Parameters:
        bin_data: (n_bin, n_samples) matrix of binary (0/1) group labels.
        cont_data: (n_cont, n_samples) matrix of continuous values.
        nan_bin: sentinel marking a missing entry in bin_data.
        nan_cont: sentinel marking a missing entry in cont_data.
        equal_var: True for Student's t-test, False for Welch's t-test.

    Returns:
        Dict with "t" and "p_unadjusted" keys, each an (n_bin, n_cont) array.
    """
    n_bin = bin_data.shape[0]
    n_cont = cont_data.shape[0]
    stat = np.full((n_bin, n_cont), np.nan, dtype=np.float64)
    pvals = np.full((n_bin, n_cont), np.nan, dtype=np.float64)

    for i, j in rectangular_pairs(n_bin, n_cont):
        s, p = ttest_one_pair_python(bin_data[i], cont_data[j], nan_bin, nan_cont, equal_var)
        stat[i, j] = s
        pvals[i, j] = p
    return {"t": stat, "p_unadjusted": pvals}


def ttest_one_pair_vectorized(
    x_bin: np.ndarray,
    y_cont: np.ndarray,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
    equal_var: bool = True,
) -> tuple[float, float]:
    """Two-sample t-test for one (binary, continuous) pair, using NumPy array operations.

    Same statistic and edge-case handling as ttest_one_pair_python (see its
    docstring for the pooled-variance vs. Welch-Satterthwaite formulas);
    here the group split and mean/variance computation use boolean masking
    and `.mean()`/`.var(ddof=1)` instead of Python loops and generator
    expressions.

    Parameters:
        x_bin: 1D array of binary group labels (0 or 1), one per sample.
        y_cont: 1D array of continuous values, same length as x_bin.
        nan_bin: sentinel marking a missing entry in x_bin.
        nan_cont: sentinel marking a missing entry in y_cont.
        equal_var: True for Student's pooled-variance t-test, False for Welch's.

    Returns:
        (t, p_unadjusted), NaN under the same conditions as ttest_one_pair_python.
    """
    valid = (x_bin != nan_bin) & (y_cont != nan_cont)
    xb = x_bin[valid]
    yc = y_cont[valid]

    if xb.size == 0:
        return np.nan, np.nan
    if not np.all((xb == 0) | (xb == 1)):
        return np.nan, np.nan

    g0 = yc[xb == 0]
    g1 = yc[xb == 1]

    n0 = g0.size
    n1 = g1.size
    if n0 < 2 or n1 < 2 or (n0 + n1) < 3:
        return np.nan, np.nan

    m0 = g0.mean()
    m1 = g1.mean()
    v0 = g0.var(ddof=1)
    v1 = g1.var(ddof=1)

    if equal_var:
        pooled_den = n0 + n1 - 2
        pooled_num = (n0 - 1) * v0 + (n1 - 1) * v1
        if pooled_den <= 0:
            return np.nan, np.nan
        pooled_var = pooled_num / pooled_den
        if pooled_var <= 0:
            return np.nan, np.nan
        se = np.sqrt(pooled_var * (1 / n0 + 1 / n1))
        if se == 0:
            return np.nan, np.nan
        tval = float((m0 - m1) / se)
        dof = n0 + n1 - 2
    else:
        se2 = v0 / n0 + v1 / n1
        if se2 <= 0:
            return np.nan, np.nan
        tval = float((m0 - m1) / np.sqrt(se2))
        num = se2 ** 2
        den = ((v0 / n0) ** 2) / (n0 - 1) + ((v1 / n1) ** 2) / (n1 - 1)
        if den == 0:
            return np.nan, np.nan
        dof = float(num / den)

    p = float(2.0 * t_dist.sf(abs(tval), dof))
    return tval, p


def ttest_vectorized(
    bin_data: np.ndarray,
    cont_data: np.ndarray,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
    equal_var: bool = True,
) -> dict[str, np.ndarray]:
    """Pairwise t-test across every (binary, continuous) feature pair (vectorized single-pair kernel).

    Same pair-iteration structure as ttest_reference, but each pair uses
    ttest_one_pair_vectorized instead of the pure-Python version.

    Parameters:
        bin_data: (n_bin, n_samples) matrix of binary (0/1) group labels.
        cont_data: (n_cont, n_samples) matrix of continuous values.
        nan_bin: sentinel marking a missing entry in bin_data.
        nan_cont: sentinel marking a missing entry in cont_data.
        equal_var: True for Student's t-test, False for Welch's t-test.

    Returns:
        Dict with "t" and "p_unadjusted" keys, each an (n_bin, n_cont) array.
    """
    n_bin = bin_data.shape[0]
    n_cont = cont_data.shape[0]
    stat = np.full((n_bin, n_cont), np.nan, dtype=np.float64)
    pvals = np.full((n_bin, n_cont), np.nan, dtype=np.float64)

    for i, j in rectangular_pairs(n_bin, n_cont):
        s, p = ttest_one_pair_vectorized(bin_data[i], cont_data[j], nan_bin, nan_cont, equal_var)
        stat[i, j] = s
        pvals[i, j] = p
    return {"t": stat, "p_unadjusted": pvals}


def _ttest_chunk_python(
    bin_data: np.ndarray,
    cont_data: np.ndarray,
    pair_chunk: list[tuple[int, int]],
    nan_bin: int,
    nan_cont: float,
    equal_var: bool,
) -> list[tuple[int, int, float, float]]:
    """Compute the t-test for one chunk of (binary, continuous) pairs (Dask task body).

    Analogous to _chi2_chunk_python: this is the function each Dask task
    actually executes, looping over its assigned pairs and calling the
    vectorized single-pair kernel for each.

    Parameters:
        bin_data: (n_bin, n_samples) matrix of binary group labels (shared
            read-only by every task).
        cont_data: (n_cont, n_samples) matrix of continuous values (shared
            read-only by every task).
        pair_chunk: list of (i, j) feature-index pairs assigned to this task.
        nan_bin: sentinel marking a missing entry in bin_data.
        nan_cont: sentinel marking a missing entry in cont_data.
        equal_var: True for Student's t-test, False for Welch's t-test.

    Returns:
        A list of (i, j, t, p_unadjusted) tuples, one per pair in the chunk.
    """
    out = []
    for i, j in pair_chunk:
        s, p = ttest_one_pair_vectorized(bin_data[i], cont_data[j], nan_bin, nan_cont, equal_var)
        out.append((i, j, s, p))
    return out


def ttest_dask(
    bin_data: np.ndarray,
    cont_data: np.ndarray,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
    equal_var: bool = True,
    workload: int = 50,
) -> dict[str, np.ndarray]:
    """Pairwise t-test parallelized across feature-pair chunks with Dask.

    Same block-splitting/task-graph approach as chi2_dask (see its
    docstring): the full n_bin x n_cont pair list is split into chunks of
    `workload` pairs, each chunk becomes one `dask.delayed` task, and
    `compute(*tasks)` runs them across the active Dask cluster's workers.

    Parameters:
        bin_data: (n_bin, n_samples) matrix of binary (0/1) group labels.
        cont_data: (n_cont, n_samples) matrix of continuous values.
        nan_bin: sentinel marking a missing entry in bin_data.
        nan_cont: sentinel marking a missing entry in cont_data.
        equal_var: True for Student's t-test, False for Welch's t-test.
        workload: number of feature pairs per Dask task.

    Returns:
        Dict with "t" and "p_unadjusted" keys, each an (n_bin, n_cont) array.
    """
    n_bin = bin_data.shape[0]
    n_cont = cont_data.shape[0]
    stat = np.full((n_bin, n_cont), np.nan, dtype=np.float64)
    pvals = np.full((n_bin, n_cont), np.nan, dtype=np.float64)

    pairs = rectangular_pairs(n_bin, n_cont)
    tasks = [
        delayed(_ttest_chunk_python)(bin_data, cont_data, chunk, nan_bin, nan_cont, equal_var)
        for chunk in chunked(pairs, workload)
    ]
    results = compute(*tasks)

    for chunk_result in results:
        for i, j, s, p in chunk_result:
            stat[i, j] = s
            pvals[i, j] = p
    return {"t": stat, "p_unadjusted": pvals}


@njit(cache=True)
def _ttest_numba_pair(
    x_bin: np.ndarray,
    y_cont: np.ndarray,
    nan_bin: int,
    nan_cont: float,
) -> tuple[float, float, int]:
    """Numba-JIT-compiled Student's t-test for one (binary, continuous) pair.

    Computes the same pooled-variance statistic as ttest_one_pair_python
    with equal_var=True (this kernel only implements Student's t-test, not
    Welch's - see ttest_dask_numba). The computation runs in two explicit
    passes over the samples, because the variance formula needs each
    group's mean before it can accumulate squared deviations from that mean:

      Pass 1: accumulate each group's sample count and sum, to get mean0/mean1.
      Pass 2: using those means, accumulate each group's sum of squared
              deviations (ss0/ss1), from which sample variance = ss/(n-1).

    (Note: Welch's t-test is not supported here - equal_var is not a
    parameter, unlike ttest_one_pair_python/vectorized.)

    Parameters:
        x_bin: 1D array of binary group labels (0 or 1), one per sample.
        y_cont: 1D array of continuous values, same length as x_bin.
        nan_bin: sentinel marking a missing entry in x_bin.
        nan_cont: sentinel marking a missing entry in y_cont.

    Returns:
        (t, dof, ok): the t-statistic, its degrees of freedom, and 1 if the
        test is well-defined or 0 if not (in which case t and dof are NaN).
        The p-value is computed by the caller (_ttest_chunk_numba) since
        scipy's t-distribution survival function is not Numba-compatible.
    """
    n0 = 0
    n1 = 0
    sum0 = 0.0
    sum1 = 0.0

    # Pass 1: per-group sample count and sum (pairwise-deleting missing entries).
    for k in range(x_bin.shape[0]):
        a = int(x_bin[k])
        b = y_cont[k]
        if a == nan_bin or b == nan_cont:
            continue
        if a == 0:
            n0 += 1
            sum0 += b
        elif a == 1:
            n1 += 1
            sum1 += b
        else:
            return np.nan, np.nan, 0

    if n0 < 2 or n1 < 2 or (n0 + n1) < 3:
        return np.nan, np.nan, 0

    mean0 = sum0 / n0
    mean1 = sum1 / n1

    # Pass 2: sum of squared deviations from each group's mean, needed for
    # the (unbiased, ddof=1) sample variance below.
    ss0 = 0.0
    ss1 = 0.0
    for k in range(x_bin.shape[0]):
        a = int(x_bin[k])
        b = y_cont[k]
        if a == nan_bin or b == nan_cont:
            continue
        if a == 0:
            d = b - mean0
            ss0 += d * d
        elif a == 1:
            d = b - mean1
            ss1 += d * d

    v0 = ss0 / (n0 - 1)
    v1 = ss1 / (n1 - 1)

    pooled_den = n0 + n1 - 2
    if pooled_den <= 0:
        return np.nan, np.nan, 0

    # Student's pooled variance estimate and standard error (see
    # ttest_one_pair_python for the equal_var=True formula).
    pooled_var = ((n0 - 1) * v0 + (n1 - 1) * v1) / pooled_den
    if pooled_var <= 0:
        return np.nan, np.nan, 0

    se = math.sqrt(pooled_var * (1.0 / n0 + 1.0 / n1))
    if se == 0:
        return np.nan, np.nan, 0

    tval = (mean0 - mean1) / se
    dof = pooled_den
    return tval, dof, 1


def _ttest_chunk_numba(
    bin_data: np.ndarray,
    cont_data: np.ndarray,
    pair_chunk: list[tuple[int, int]],
    nan_bin: int,
    nan_cont: float,
) -> list[tuple[int, int, float, float]]:
    """Compute the t-test for one chunk of pairs, using the Numba kernel (Dask task body).

    Parameters:
        bin_data: (n_bin, n_samples) matrix of binary group labels.
        cont_data: (n_cont, n_samples) matrix of continuous values.
        pair_chunk: list of (i, j) feature-index pairs assigned to this task.
        nan_bin: sentinel marking a missing entry in bin_data.
        nan_cont: sentinel marking a missing entry in cont_data.

    Returns:
        A list of (i, j, t, p_unadjusted) tuples, one per pair in the chunk.
    """
    out = []
    for i, j in pair_chunk:
        tval, dof, ok = _ttest_numba_pair(bin_data[i], cont_data[j], nan_bin, nan_cont)
        p = np.nan if ok == 0 else float(2.0 * t_dist.sf(abs(tval), dof))
        out.append((i, j, tval, p))
    return out


def ttest_dask_numba(
    bin_data: np.ndarray,
    cont_data: np.ndarray,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
    workload: int = 50,
) -> dict[str, np.ndarray]:
    """Pairwise t-test parallelized with Dask, using the Numba-JIT per-pair kernel.

    Same block-splitting approach as ttest_dask, but each task runs
    _ttest_chunk_numba instead of _ttest_chunk_python.

    Note this configuration only supports Student's t-test - there is no
    `equal_var` parameter, because _ttest_numba_pair does not implement
    Welch's t-test (unlike the reference/vectorized/pure-Dask configs).
    This is a known gap in feature parity across the four implementations;
    it does not affect this project's benchmarks, which all use
    equal_var=True (matching NApy's default) throughout.

    Parameters:
        bin_data: (n_bin, n_samples) matrix of binary (0/1) group labels.
        cont_data: (n_cont, n_samples) matrix of continuous values.
        nan_bin: sentinel marking a missing entry in bin_data.
        nan_cont: sentinel marking a missing entry in cont_data.
        workload: number of feature pairs per Dask task.

    Returns:
        Dict with "t" and "p_unadjusted" keys, each an (n_bin, n_cont) array.
    """
    n_bin = bin_data.shape[0]
    n_cont = cont_data.shape[0]
    stat = np.full((n_bin, n_cont), np.nan, dtype=np.float64)
    pvals = np.full((n_bin, n_cont), np.nan, dtype=np.float64)

    pairs = rectangular_pairs(n_bin, n_cont)
    tasks = [
        delayed(_ttest_chunk_numba)(bin_data, cont_data, chunk, nan_bin, nan_cont)
        for chunk in chunked(pairs, workload)
    ]
    results = compute(*tasks)

    for chunk_result in results:
        for i, j, s, p in chunk_result:
            stat[i, j] = s
            pvals[i, j] = p
    return {"t": stat, "p_unadjusted": pvals}


def ttest_napy(
    bin_data: np.ndarray,
    cont_data: np.ndarray,
    nan_bin: int = NAN_VALUE_CAT,
    nan_cont: float = NAN_VALUE_CONT,
) -> dict[str, np.ndarray]:
    """Pairwise t-test via the real NApy package (ground-truth comparison).

    Thin wrapper around napy.ttest, used as the performance baseline that
    the four hand-written implementations above are benchmarked against
    (and, in tests/test_ttest.py, checked for numerical agreement with).

    Parameters:
        bin_data: (n_bin, n_samples) matrix of binary group labels. Callers
            must pre-convert this to use `nan_cont` as its missing-value
            sentinel (see benchmark_ttest), since NApy's ttest takes a
            single `nan_value` shared by both input matrices.
        cont_data: (n_cont, n_samples) matrix of continuous values.
        nan_bin: unused by NApy directly; kept for signature symmetry with
            the other ttest_* functions.
        nan_cont: sentinel marking a missing entry in either matrix, passed
            to NApy as its single nan_value.

    Returns:
        Dict with "t" and "p_unadjusted" keys, as returned by napy.ttest.

    Raises:
        ImportError: if the napypi package is not installed.
    """
    if napy is None:
        raise ImportError("napypi is not installed.")
    return napy.ttest(
        bin_data,
        cont_data,
        nan_value=float(nan_cont),  # NApy uses one nan value for both matrices
        axis=0,
        threads=1,
        check_data=False,
        return_types=["t", "p_unadjusted"],
        equal_var=True,
    )


# ----------------------------
# Validation
# ----------------------------

def compare_arrays(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> float:
    """Largest absolute difference between two result arrays, ignoring shared NaNs.

    Used to check that two implementations agree: a NaN in both a and b at
    the same position is treated as "they agree it's undefined" and is
    excluded from the comparison, rather than propagating NaN through the
    max/abs/subtract chain (which would otherwise always report the result
    as NaN regardless of how well-defined entries compare).

    Parameters:
        a: first result array.
        b: second result array, same shape as a.
        atol: unused; kept for API symmetry with numpy's isclose family
            (comparisons here just report the max diff for the caller to
            threshold themselves, e.g. via print statements below).

    Returns:
        The maximum absolute elementwise difference over positions where a
        and b are not both NaN. Returns 0.0 if every position is a shared NaN.
    """
    mask = ~(np.isnan(a) & np.isnan(b))
    if not np.any(mask):
        return 0.0
    aa = a[mask]
    bb = b[mask]
    return float(np.nanmax(np.abs(aa - bb)))


def validate_small_examples() -> None:
    """Sanity-check that all four implementations agree, on small simulated data.

    Runs chi2_reference/vectorized/dask/dask_numba on one small simulated
    categorical matrix, and ttest_reference/vectorized/dask/dask_numba on
    one small simulated (binary, continuous) pair, printing the maximum
    disagreement (see compare_arrays) between the reference implementation
    and each of the other three. This is called once at the start of
    main() as a quick smoke test before the timed benchmark runs - a large
    printed diff would indicate a correctness bug, not a performance issue.
    """
    rng = np.random.default_rng(RNG_SEED)

    # chi2
    cat = simulate_categorical_matrix(8, 50, n_categories=4, missing_ratio=0.1, rng=rng)
    c_ref = chi2_reference(cat)
    c_vec = chi2_vectorized(cat)
    c_dask = chi2_dask(cat, workload=5)
    c_numba = chi2_dask_numba(cat, workload=5)
    print("chi2 validation max diff ref-vect:", compare_arrays(c_ref["chi2"], c_vec["chi2"]))
    print("chi2 validation max diff ref-dask:", compare_arrays(c_ref["chi2"], c_dask["chi2"]))
    print("chi2 validation max diff ref-dask_numba:", compare_arrays(c_ref["chi2"], c_numba["chi2"]))

    # ttest
    b, c = simulate_binary_continuous_matrices(6, 6, 60, missing_ratio=0.1, rng=rng)
    b = b.astype(np.int64)
    c = c.astype(np.float64)
    t_ref = ttest_reference(b, c)
    t_vec = ttest_vectorized(b, c)
    t_dask = ttest_dask(b, c, workload=5)
    t_numba = ttest_dask_numba(b, c, workload=5)
    print("ttest validation max diff ref-vect:", compare_arrays(t_ref["t"], t_vec["t"]))
    print("ttest validation max diff ref-dask:", compare_arrays(t_ref["t"], t_dask["t"]))
    print("ttest validation max diff ref-dask_numba:", compare_arrays(t_ref["t"], t_numba["t"]))


# ----------------------------
# Benchmarking
# ----------------------------

def time_call(fn: Callable, *args, repeats: int = 3, **kwargs) -> tuple[float, object]:
    """Time a function call, running it `repeats` times and keeping the best.

    The minimum (rather than mean) of several runs is reported because
    wall-clock timings on a shared, non-realtime OS are noise-prone in one
    direction only: background processes, OS scheduling, and (for the Dask
    configurations) process/IPC jitter can only slow a run down, never speed
    it up below its true cost. The fastest observed run is therefore the
    closest single estimate of the function's actual cost.

    Parameters:
        fn: the function to time.
        *args: positional arguments passed to fn on every call.
        repeats: number of times to call fn (every call is timed; this is
            not the same as best-of-N with early stopping).
        **kwargs: keyword arguments passed to fn on every call.

    Returns:
        (best_seconds, last_result): the minimum runtime across all repeats,
        and the return value of the final call to fn (kept only so callers
        can e.g. print/inspect the last result if needed).
    """
    best = float("inf")
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best, last


def benchmark_chi2(
    feature_list: list[int],
    sample_list: list[int],
    missing_ratio: float,
    workers: int,
    workload: int,
    repeats: int,
) -> list[BenchmarkResult]:
    """Time all chi-squared implementations across a grid of feature/sample counts.

    For every (features, samples) combination in feature_list x sample_list,
    simulates one categorical matrix and times every available
    implementation (python_ref, vectorized, dask, dask_numba, plus napy if
    installed) on that exact matrix, so all implementations are compared on
    identical data at each size. Called twice from main() with one of the
    two lists held to a single fixed value, to produce the independent
    "vary features" / "vary samples" sweeps for Task 2.

    Parameters:
        feature_list: feature counts to simulate.
        sample_list: sample counts to simulate.
        missing_ratio: fraction of entries to mark as missing in each
            simulated matrix.
        workers: number of Dask worker processes (recorded in the result
            rows for reference; the LocalCluster itself is created in main()).
        workload: number of feature pairs per Dask task, used by the dask
            and dask_numba implementations.
        repeats: number of timed repeats per implementation (see time_call).

    Returns:
        A list of BenchmarkResult rows, one per (features, samples,
        implementation) combination.
    """
    results: list[BenchmarkResult] = []
    rng = np.random.default_rng(RNG_SEED)

    impls = {
        "python_ref": chi2_reference,
        "vectorized": chi2_vectorized,
        "dask": lambda data: chi2_dask(data, workload=workload),
        "dask_numba": lambda data: chi2_dask_numba(data, workload=workload),
    }
    if napy is not None:
        impls["napy"] = chi2_napy

    for f in feature_list:
        for s in sample_list:
            data = simulate_categorical_matrix(f, s, n_categories=4, missing_ratio=missing_ratio, rng=rng)
            for name, fn in impls.items():
                runtime, _ = time_call(fn, data, repeats=repeats)
                results.append(
                    BenchmarkResult(
                        test_name="chi2",
                        implementation=name,
                        features=f,
                        samples=s,
                        missing_ratio=missing_ratio,
                        workers=workers,
                        workload=workload,
                        runtime_sec=runtime,
                    )
                )
                print(f"[chi2] {name:12s} features={f:4d} samples={s:4d} time={runtime:.4f}s")
    return results


def benchmark_ttest(
    feature_list: list[int],
    sample_list: list[int],
    missing_ratio: float,
    workers: int,
    workload: int,
    repeats: int,
) -> list[BenchmarkResult]:
    """Time all t-test implementations across a grid of feature/sample counts.

    Analogous to benchmark_chi2: for every (features, samples) combination,
    simulates one (n_bin=features, n_cont=features, samples) pair of
    matrices and times every available implementation on that exact data.

    Parameters:
        feature_list: feature counts to simulate (used as both n_bin and
            n_cont, so the grid is square).
        sample_list: sample counts to simulate.
        missing_ratio: fraction of entries to mark as missing in each
            simulated matrix.
        workers: number of Dask worker processes (recorded for reference).
        workload: number of feature pairs per Dask task, used by the dask
            and dask_numba implementations.
        repeats: number of timed repeats per implementation (see time_call).

    Returns:
        A list of BenchmarkResult rows, one per (features, samples,
        implementation) combination.
    """
    results: list[BenchmarkResult] = []
    rng = np.random.default_rng(RNG_SEED)

    impls = {
        "python_ref": lambda b, c: ttest_reference(b, c),
        "vectorized": lambda b, c: ttest_vectorized(b, c),
        "dask": lambda b, c: ttest_dask(b, c, workload=workload),
        "dask_numba": lambda b, c: ttest_dask_numba(b, c, workload=workload),
    }
    if napy is not None:
        impls["napy"] = lambda b, c: ttest_napy(b, c)

    for f in feature_list:
        for s in sample_list:
            bin_data, cont_data = simulate_binary_continuous_matrices(f, f, s, missing_ratio=missing_ratio, rng=rng)
            # To compare fairly with NApy, use one nan sentinel for both matrices.
            bin_for_napy = bin_data.astype(np.float64)
            bin_for_napy[bin_for_napy == NAN_VALUE_CAT] = float(NAN_VALUE_CONT)
            cont_for_napy = cont_data.copy()

            for name, fn in impls.items():
                if name == "napy":
                    runtime, _ = time_call(fn, bin_for_napy, cont_for_napy, repeats=repeats)
                else:
                    runtime, _ = time_call(fn, bin_data, cont_data, repeats=repeats)
                results.append(
                    BenchmarkResult(
                        test_name="ttest",
                        implementation=name,
                        features=f,
                        samples=s,
                        missing_ratio=missing_ratio,
                        workers=workers,
                        workload=workload,
                        runtime_sec=runtime,
                    )
                )
                print(f"[ttest] {name:12s} features={f:4d} samples={s:4d} time={runtime:.4f}s")
    return results


def benchmark_workloads(
    test_name: str,
    features: int,
    samples: int,
    missing_ratio: float,
    workers: int,
    workloads: list[int],
    repeats: int,
) -> list[BenchmarkResult]:
    """Time the Dask and Dask+Numba implementations across a range of chunk sizes (Task 3).

    Simulates ONE fixed-size problem (given by `features`/`samples`) and,
    for each value in `workloads`, times chi2_dask/ttest_dask and
    chi2_dask_numba/ttest_dask_numba with that chunk size. This isolates the
    workload -> runtime relationship from problem-size effects: only the
    number of pairs dispatched per Dask task changes between runs, not the
    data itself. Only the two Dask-based implementations are timed here -
    python_ref/vectorized/napy have no notion of a Dask workload.

    Parameters:
        test_name: "chi2" or "ttest".
        features: fixed feature count for the simulated data.
        samples: fixed sample count for the simulated data.
        missing_ratio: fraction of entries to mark as missing.
        workers: number of Dask worker processes (recorded for reference).
        workloads: list of chunk sizes (feature pairs per Dask task) to sweep.
        repeats: number of timed repeats per workload (see time_call).

    Returns:
        A list of BenchmarkResult rows, one per (workload, implementation) combination.

    Raises:
        ValueError: if test_name is not "chi2" or "ttest".
    """
    rng = np.random.default_rng(RNG_SEED)
    results: list[BenchmarkResult] = []

    if test_name == "chi2":
        data = simulate_categorical_matrix(features, samples, n_categories=4, missing_ratio=missing_ratio, rng=rng)
        for workload in workloads:
            for impl_name, fn in {
                "dask": lambda: chi2_dask(data, workload=workload),
                "dask_numba": lambda: chi2_dask_numba(data, workload=workload),
            }.items():
                runtime, _ = time_call(fn, repeats=repeats)
                results.append(
                    BenchmarkResult(
                        test_name=test_name,
                        implementation=impl_name,
                        features=features,
                        samples=samples,
                        missing_ratio=missing_ratio,
                        workers=workers,
                        workload=workload,
                        runtime_sec=runtime,
                    )
                )
                print(f"[{test_name}] workload={workload:4d} {impl_name:10s} time={runtime:.4f}s")
    elif test_name == "ttest":
        bin_data, cont_data = simulate_binary_continuous_matrices(features, features, samples, missing_ratio=missing_ratio, rng=rng)
        for workload in workloads:
            for impl_name, fn in {
                "dask": lambda: ttest_dask(bin_data, cont_data, workload=workload),
                "dask_numba": lambda: ttest_dask_numba(bin_data, cont_data, workload=workload),
            }.items():
                runtime, _ = time_call(fn, repeats=repeats)
                results.append(
                    BenchmarkResult(
                        test_name=test_name,
                        implementation=impl_name,
                        features=features,
                        samples=samples,
                        missing_ratio=missing_ratio,
                        workers=workers,
                        workload=workload,
                        runtime_sec=runtime,
                    )
                )
                print(f"[{test_name}] workload={workload:4d} {impl_name:10s} time={runtime:.4f}s")
    else:
        raise ValueError("test_name must be 'chi2' or 'ttest'")

    return results


# ----------------------------
# Plotting
# ----------------------------

# Shared styling so every plot uses the same color/marker/label for a given
# implementation, and always draws lines in the same left-to-right legend order.
IMPL_ORDER = ["python_ref", "vectorized", "dask", "dask_numba", "napy"]
IMPL_STYLE = {
    "python_ref": dict(color="#d62728", marker="o"),
    "vectorized": dict(color="#ff7f0e", marker="s"),
    "dask": dict(color="#1f77b4", marker="^"),
    "dask_numba": dict(color="#2ca02c", marker="D"),
    "napy": dict(color="#9467bd", marker="*"),
}
IMPL_LABEL = {
    "python_ref": "Python reference",
    "vectorized": "Vectorized (NumPy)",
    "dask": "Pure Dask",
    "dask_numba": "Dask + Numba",
    "napy": "NApy",
}


def save_runtime_plot(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    xcol: str,
    xlabel: str,
    logy: bool = True,
    logx: bool = False,
) -> None:
    """Draw and save one runtime-vs-x line plot, one line per implementation.

    Used for all three sweep types (features, samples, workload); the
    caller (main()) is responsible for pre-filtering `df` down to the rows
    for one sweep and one test_name before calling this.

    Parameters:
        df: rows to plot, must have "implementation", `xcol`, and
            "runtime_sec" columns. Each implementation present is drawn as
            its own line (styled/labeled/ordered via IMPL_STYLE/IMPL_LABEL/
            IMPL_ORDER), sorted by `xcol` so lines connect left to right.
        outpath: file path to save the PNG to.
        title: plot title.
        xcol: name of the column in `df` to use as the x-axis values.
        xlabel: human-readable x-axis label (independent of `xcol`, so the
            raw column name doesn't have to double as the axis label).
        logy: use a log-scale y-axis. Default True, since NApy is typically
            2-3 orders of magnitude faster than the other implementations -
            a linear axis would flatten it to an indistinguishable line
            near zero.
        logx: use a log-scale x-axis (used for the workload sweep, whose
            values span two orders of magnitude).
    """
    plt.figure(figsize=(8, 5.5))
    present = [impl for impl in IMPL_ORDER if impl in df["implementation"].unique()]
    for impl in present:
        sub = df[df["implementation"] == impl].sort_values(xcol)
        style = IMPL_STYLE.get(impl, {})
        plt.plot(
            sub[xcol],
            sub["runtime_sec"],
            marker=style.get("marker", "o"),
            color=style.get("color"),
            linewidth=1.8,
            markersize=6,
            label=IMPL_LABEL.get(impl, impl),
        )
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel("Runtime (s, log scale)" if logy else "Runtime (s)", fontsize=11)
    if logy:
        plt.yscale("log")
    if logx:
        plt.xscale("log")
    plt.title(title, fontsize=12)
    plt.legend(title="Implementation", fontsize=9, title_fontsize=9, framealpha=0.9)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# Main driver
# ----------------------------

def main() -> None:
    """CLI entry point: run the correctness smoke check, then Task 2 and Task 3 benchmarks.

    Parses --test/--mode/--workers/--outdir/--repeats (see the module
    docstring and README.md for flag meanings), starts a local Dask cluster,
    runs validate_small_examples() as a quick correctness check, then runs:

      - Task 2: two independent size sweeps (vary features with samples
        fixed, then vary samples with features fixed) via benchmark_chi2/
        benchmark_ttest, covering all implementations including NApy.
      - Task 3: a workload sweep via benchmark_workloads, covering only the
        Dask and Dask+Numba implementations at one fixed problem size.

    All rows are tagged with which sweep produced them (see the `tag`
    closure below and BenchmarkResult.sweep), written to a single CSV, and
    plotted per test_name/sweep combination with save_runtime_plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["chi2", "ttest", "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "benchmark"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Dask client
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1, processes=True, dashboard_address=None)
    client = Client(cluster)
    print(client)

    try:
        validate_small_examples()

        missing_ratio = 0.1

        if args.mode == "smoke":
            # Task 2: two independent size sweeps (one dimension varies, the other is fixed)
            feature_sweep = [10, 20, 40]
            samples_fixed_for_feature_sweep = 100
            sample_sweep = [100, 200]
            features_fixed_for_sample_sweep = 10
            base_workload = 10
            # Task 3: workload sweep at a fixed, moderately-sized problem
            workload_features, workload_samples = 20, 100
            workloads = [5, 10, 20]
        else:
            feature_sweep = [20, 50, 100, 150, 200, 300]
            samples_fixed_for_feature_sweep = 500
            sample_sweep = [100, 250, 500, 1000, 2000]
            features_fixed_for_sample_sweep = 100
            base_workload = 50
            workload_features, workload_samples = 200, 500
            workloads = [5, 10, 25, 50, 100, 200, 400]

        all_results: list[BenchmarkResult] = []

        def tag(rows: list[BenchmarkResult], sweep: str) -> list[BenchmarkResult]:
            for r in rows:
                r.sweep = sweep
            return rows

        if args.test in ("chi2", "all"):
            all_results.extend(tag(benchmark_chi2(
                feature_list=feature_sweep,
                sample_list=[samples_fixed_for_feature_sweep],
                missing_ratio=missing_ratio,
                workers=args.workers,
                workload=base_workload,
                repeats=args.repeats,
            ), "features"))
            all_results.extend(tag(benchmark_chi2(
                feature_list=[features_fixed_for_sample_sweep],
                sample_list=sample_sweep,
                missing_ratio=missing_ratio,
                workers=args.workers,
                workload=base_workload,
                repeats=args.repeats,
            ), "samples"))
            all_results.extend(tag(benchmark_workloads(
                test_name="chi2",
                features=workload_features,
                samples=workload_samples,
                missing_ratio=missing_ratio,
                workers=args.workers,
                workloads=workloads,
                repeats=args.repeats,
            ), "workload"))

        if args.test in ("ttest", "all"):
            all_results.extend(tag(benchmark_ttest(
                feature_list=feature_sweep,
                sample_list=[samples_fixed_for_feature_sweep],
                missing_ratio=missing_ratio,
                workers=args.workers,
                workload=base_workload,
                repeats=args.repeats,
            ), "features"))
            all_results.extend(tag(benchmark_ttest(
                feature_list=[features_fixed_for_sample_sweep],
                sample_list=sample_sweep,
                missing_ratio=missing_ratio,
                workers=args.workers,
                workload=base_workload,
                repeats=args.repeats,
            ), "samples"))
            all_results.extend(tag(benchmark_workloads(
                test_name="ttest",
                features=workload_features,
                samples=workload_samples,
                missing_ratio=missing_ratio,
                workers=args.workers,
                workloads=workloads,
                repeats=args.repeats,
            ), "workload"))

        df = pd.DataFrame([r.__dict__ for r in all_results])
        csv_path = outdir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

        for test_name in sorted(df["test_name"].unique()):
            fplot = df[(df["test_name"] == test_name) & (df["sweep"] == "features")]
            if not fplot.empty:
                save_runtime_plot(
                    fplot,
                    outdir / f"{test_name}_runtime_vs_features.png",
                    f"{test_name}: runtime vs. number of features (samples={samples_fixed_for_feature_sweep})",
                    "features",
                    xlabel="Number of features (F)",
                )

            splot = df[(df["test_name"] == test_name) & (df["sweep"] == "samples")]
            if not splot.empty:
                save_runtime_plot(
                    splot,
                    outdir / f"{test_name}_runtime_vs_samples.png",
                    f"{test_name}: runtime vs. number of samples (features={features_fixed_for_sample_sweep})",
                    "samples",
                    xlabel="Number of samples (S)",
                )

            wplot = df[(df["test_name"] == test_name) & (df["sweep"] == "workload")]
            if not wplot.empty:
                save_runtime_plot(
                    wplot,
                    outdir / f"{test_name}_runtime_vs_workload.png",
                    f"{test_name}: runtime vs. Dask workload (pairs/task; F={workload_features}, S={workload_samples})",
                    "workload",
                    xlabel="Workload (pairs per Dask task)",
                    logx=True,
                )

        print("Done.")
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
