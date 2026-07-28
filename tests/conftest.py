"""Shared pytest config: make project_benchmark.py importable from tests/.

project_benchmark.py lives at the repo root, one directory above this
tests/ package, so it is not on sys.path by default when pytest is invoked
from elsewhere. Inserting the parent directory here (once, for the whole
test session) lets every test module do `from project_benchmark import ...`
without needing to install the project as a package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
