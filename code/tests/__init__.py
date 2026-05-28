import pytest
import sys
from pathlib import Path

# Only import relative test functions if imported as a package (e.g. by pytest)
if __name__ != "__main__":
    from .test_llm import *
    from .test_pii import *
    from .test_safety import *
    from .test_classifier import *
    from .test_retriever import *
    from .test_escalation import *
    from .test_generator import *
    from .test_pipeline import *

def run_tests():
    # Run pytest on the tests directory
    tests_dir = str(Path(__file__).parent)
    return pytest.main([tests_dir, "-v"])

if __name__ == "__main__":
    sys.exit(run_tests())
