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
if __name__ != "__main__":
    _here = Path(__file__).parent
    for _path in sorted(_here.glob("test_*.py")):
        _module = importlib.import_module(f"{__name__}.{_path.stem}")
        for _name in dir(_module):
            if _name.startswith("test_") or _name.startswith("Test"):
                globals()[_name] = getattr(_module, _name)


def run_tests():
    """Convenience: `python code/tests/__init__.py` runs pytest on this dir."""
    import pytest
    tests_dir = str(Path(__file__).parent)
    return pytest.main([tests_dir, "-v"])


if __name__ == "__main__":
    sys.exit(run_tests())
