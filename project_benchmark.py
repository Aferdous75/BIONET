
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
    """Feature x sample categorical matrix."""
    rng = np.random.default_rng() if rng is None else rng
    data = rng.integers(0, n_categories, size=(n_features, n_samples), dtype=np.int64)
    if missing_ratio > 0:
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
    """Feature x sample matrices for t-test."""
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
    return list(combinations(range(n_features), 2))


def rectangular_pairs(n_left: int, n_right: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_left) for j in range(n_right)]


def chunked(seq: list[tuple[int, int]], chunk_size: int) -> list[list[tuple[int, int]]]:
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


# ----------------------------
# Chi-squared implementations
# ----------------------------

def chi2_one_pair_python(x: np.ndarray, y: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> tuple[float, float]:
    valid_pairs: list[tuple[int, int]] = []
    categories_x = set()
    categories_y = set()

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
    n_features = data.shape[0]
    stat = np.full((n_features, n_features), np.nan, dtype=np.float64)
    pvals = np.full((n_features, n_features), np.nan, dtype=np.float64)

    for i, j in upper_triangle_pairs(n_features):
        s, p = chi2_one_pair_python(data[i], data[j], nan_value=nan_value)
        stat[i, j] = stat[j, i] = s
        pvals[i, j] = pvals[j, i] = p
    return {"chi2": stat, "p_unadjusted": pvals}


def chi2_one_pair_vectorized(x: np.ndarray, y: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> tuple[float, float]:
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

    if np.unique(xv).size != categories_x.size or np.unique(yv).size != categories_y.size:
        return np.nan, np.nan

    n_cat_x = int(categories_x.max()) + 1
    n_cat_y = int(categories_y.max()) + 1

    flat = np.bincount(xv * n_cat_y + yv, minlength=n_cat_x * n_cat_y)
    contingency = flat.reshape(n_cat_x, n_cat_y).astype(np.float64)

    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    total = contingency.sum()

    if total == 0:
        return np.nan, np.nan

    expected = row_sums @ col_sums / total
    if np.any(expected == 0):
        return np.nan, np.nan

    chi2 = float(((contingency - expected) ** 2 / expected).sum())
    dof = (n_cat_x - 1) * (n_cat_y - 1)
    p = float(chi2_dist.sf(chi2, dof))
    return chi2, p


def chi2_vectorized(data: np.ndarray, nan_value: int = NAN_VALUE_CAT) -> dict[str, np.ndarray]:
    n_features = data.shape[0]
    stat = np.full((n_features, n_features), np.nan, dtype=np.float64)
    pvals = np.full((n_features, n_features), np.nan, dtype=np.float64)

    for i, j in upper_triangle_pairs(n_features):
        s, p = chi2_one_pair_vectorized(data[i], data[j], nan_value=nan_value)
        stat[i, j] = stat[j, i] = s
        pvals[i, j] = pvals[j, i] = p
    return {"chi2": stat, "p_unadjusted": pvals}


def _chi2_chunk_python(data: np.ndarray, pair_chunk: list[tuple[int, int]], nan_value: int) -> list[tuple[int, int, float, float]]:
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
    max_x = -1
    max_y = -1
    total_nonmiss_x = 0
    total_nonmiss_y = 0
    valid_count = 0

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
    seen_x = np.zeros(n_cat_x, dtype=np.int64)
    seen_y = np.zeros(n_cat_y, dtype=np.int64)
    present_x_count = 0
    present_y_count = 0

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
    group0 = []
    group1 = []

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
    n0 = 0
    n1 = 0
    sum0 = 0.0
    sum1 = 0.0

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
    mask = ~(np.isnan(a) & np.isnan(b))
    if not np.any(mask):
        return 0.0
    aa = a[mask]
    bb = b[mask]
    return float(np.nanmax(np.abs(aa - bb)))


def validate_small_examples() -> None:
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
    c[b == NAN_VALUE_CAT] = c[b == NAN_VALUE_CAT]  # no-op, keeps shapes clear
    # make same missing sentinel for NApy comparison later if needed
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
