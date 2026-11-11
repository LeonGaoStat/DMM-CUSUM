#!/usr/bin/env python3
"""Proposed-method launcher; no competitor study is dispatched."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / 'localization/run_table5_proposed.py'
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
