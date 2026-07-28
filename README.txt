BIONETS project: chi-squared test and t-test benchmarking against NApy
=======================================================================

Files
-----
- project_benchmark.py : all four implementations (Python reference,
  vectorized NumPy, pure Dask, Dask + Numba) for both chi-squared and
  t-test, plus the NApy comparison, benchmarking, and plotting.
- tests/               : pytest correctness suite. Checks that all four
  implementations agree with each other, match hand-computed values
  from scipy (chi2_contingency, ttest_ind), match the real napypi
  package's output, and handle edge cases correctly (missing values,
  categories disappearing after pairwise deletion, too few samples,
  zero variance).
- Test.py               : tiny standalone napypi smoke check (unrelated
  to the main benchmark).
- requirements.txt      : exact package versions used (pip freeze from
  the "bionets" conda environment).
- results/              : output CSV + PNG plots from the benchmark run.

Environment setup
------------------
1) conda create -n bionets python=3.11
2) conda activate bionets
3) pip install -r requirements.txt

(napypi pulls in its own pinned numpy/scipy/numba/torch versions, which
is why requirements.txt looks stricter than a typical project.)

Running the correctness tests
------------------------------
    pytest tests/ -v

23 tests, covering both statistical tests across all four
implementations plus the real napypi package.

Running the benchmarks
------------------------
    python project_benchmark.py --test all --mode smoke   --workers 4 --repeats 2
    python project_benchmark.py --test all --mode benchmark --workers 4 --repeats 3

--test        chi2 | ttest | all
--mode        smoke (small, fast sanity check) | benchmark (full sweep)
--workers     number of Dask worker processes
--repeats     timing repeats per configuration (best-of-N is not used;
              all N runs are timed and recorded)
--outdir      output directory (default: results)

Benchmark design
------------------
Task 2 (runtime vs. problem size): two independent sweeps per test, so
that each plot varies exactly one dimension at a time instead of
averaging over a mixed grid:
  - vary number of features, samples held fixed  -> *_runtime_vs_features.png
  - vary number of samples, features held fixed  -> *_runtime_vs_samples.png
Both include all four implementations plus NApy.

Task 3 (runtime vs. Dask workload): for a single fixed problem size,
vary the number of feature pairs dispatched per Dask task and measure
runtime for the pure Dask and Dask + Numba configurations only.
  -> *_runtime_vs_workload.png

All plots use a log-scale runtime axis, since NApy is typically 2-3
orders of magnitude faster than the Python-level implementations.

The "sweep" column in results/benchmark_results.csv tags which of the
three sweeps (features / samples / workload) each row belongs to.
