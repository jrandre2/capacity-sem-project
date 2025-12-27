#!/usr/bin/env python3
"""
Run Phase 2 Week 3: Multi-Stage Efficiency Analysis

Tests if velocity effects differ by bottleneck location in the
administrative pipeline (obligate→disburse→expend).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stages.s06_alternatives import run_multistage_efficiency_analysis

if __name__ == "__main__":
    print("Running Phase 2 Week 3: Multi-Stage Efficiency Analysis")
    print()

    results = run_multistage_efficiency_analysis()

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(results.to_string(index=False))
    print()
