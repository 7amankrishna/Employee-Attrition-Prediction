#!/usr/bin/env python3
"""Run the full analysis pipeline: EDA -> modeling -> report.

    python3 analysis/run_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eda
import modeling
import report


def main():
    print("=== stage 1/3: EDA ===")
    eda.main()
    print("=== stage 2/3: modeling ===")
    modeling.main()
    print("=== stage 3/3: report ===")
    report.main()
    print("done. report: reports/attrition-research-report.md")


if __name__ == "__main__":
    main()
