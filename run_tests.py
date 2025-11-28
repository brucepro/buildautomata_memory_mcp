#!/usr/bin/env python3
"""
Test Runner for BuildAutomata Memory System
Copyright 2025 Jurden Bruce

Quick test runner script for the root directory.
Simply runs the comprehensive test suite.

Usage:
    python run_tests.py
    python run_tests.py --verbose
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the test suite"""
    test_file = Path(__file__).parent / "tests" / "test_memory_system.py"

    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)

    # Pass through any arguments (like --verbose)
    cmd = [sys.executable, str(test_file)] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
