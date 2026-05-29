"""Package init for the test suite.

Pytest's default file-mode collection (e.g. `pytest code/tests/`) walks the
directory and finds every `test_*.py` regardless of this file. But when
pytest is invoked against this `__init__.py` directly (e.g.
`pytest code/tests/__init__.py`) — or anything else that treats the test
folder as a package and reads its `__init__` — it sees only what's
re-exported here.

To keep both invocation modes consistent (and to stop missing new test
files when they're added), this module auto-imports the public test
symbols from every sibling `test_*.py`. Add a new test file with the
standard naming convention and it'll be picked up automatically — no
manual maintenance of this file.
"""

import importlib
import sys
from pathlib import Path

# Only run the import sweep when imported as a package (i.e. by pytest).
# Skipped when this file is run as a script via `python __init__.py`,
# which falls through to run_tests() at the bottom.
#
# We mirror `from <module> import *` semantics here (respect __all__ if
# defined, otherwise export every public name) rather than filtering on
# `test_*` / `Test*` prefixes alone. Fixtures (e.g. `sample_rows`,
# `processed_results` in test_pipeline.py) don't follow that prefix
# convention but ARE needed by the tests that consume them — without
# them, those tests collect-but-error with "fixture not found" when
# pytest is invoked against this __init__.py directly.
if __name__ != "__main__":
    _here = Path(__file__).parent
    for _path in sorted(_here.glob("test_*.py")):
        _module = importlib.import_module(f"{__name__}.{_path.stem}")
        if hasattr(_module, "__all__"):
            _names = _module.__all__
        else:
            _names = [n for n in dir(_module) if not n.startswith("_")]
        for _name in _names:
            globals()[_name] = getattr(_module, _name)


def run_tests():
    """Convenience: `python code/tests/__init__.py` runs pytest on this dir."""
    import pytest
    tests_dir = str(Path(__file__).parent)
    return pytest.main([tests_dir, "-v"])


if __name__ == "__main__":
    sys.exit(run_tests())
