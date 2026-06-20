"""PrismLang Benchmark Suite — run all 3 domains and print comparison report.

Usage:
    python -m benchmarks.run_all                    # all domains
    python -m benchmarks.run_all --domain healthcare
    python -m benchmarks.run_all --domain finance
    python -m benchmarks.run_all --domain trade_market

Results are stored in bench.run_results in prismLangDB.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import BENCH_DSN
from benchmarks.runner import BenchmarkRunner
from benchmarks.report import print_comparison
from benchmarks.domains import healthcare, finance, trade_market

DOMAINS = {
    "healthcare":   healthcare.benchmark,
    "finance":      finance.benchmark,
    "trade_market": trade_market.benchmark,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="PrismLang Benchmark Suite")
    parser.add_argument("--domain", choices=list(DOMAINS.keys()), default=None,
                        help="Run a single domain (default: all)")
    args = parser.parse_args()

    runner = BenchmarkRunner(BENCH_DSN)
    pairs = []

    domains_to_run = {args.domain: DOMAINS[args.domain]} if args.domain else DOMAINS

    print("\n" + "=" * 80)
    print("  PrismLang Benchmark Suite — Insight IT Solutions LLC")
    print("=" * 80)

    for name, bench_fn in domains_to_run.items():
        print(f"\n[{name.upper()}]")
        std_result, prism_result = runner.run_pair(
            standard_fn=lambda fn=bench_fn: fn()[0],
            prismlang_fn=lambda fn=bench_fn: fn()[1],
        )
        pairs.append((std_result, prism_result))

    runner.close()

    print_comparison(pairs, title="PrismLang vs Standard LangGraph — Benchmark Results")


if __name__ == "__main__":
    main()
