#!/usr/bin/env python3
"""Test runner for Hybrid RAG"""

import subprocess
import sys

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--asyncio-mode=auto"],
        cwd=".",
    )
    sys.exit(result.returncode)
