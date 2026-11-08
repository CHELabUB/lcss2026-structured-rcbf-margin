#!/usr/bin/env python3
"""Reproduce everything: unit tests, simulations and certification sweep, printed numbers, figures.

    python run_all.py [--step 0.05] [--skip-tests]

Order: tests/test_model, test_margin_duality, test_cert_set (model-level, seconds) ->
generate_results.py (results/runs.npz, results/results.json; minutes) -> stats.py (the numbers of
Section V next to the printed values) -> plotter.py --extended (Fig. 1 and the extended figures) ->
tests/test_results (every printed number against results.json).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    print("\n$", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--step", type=float, default=None, help="grid spacing of the certification sweep (default 0.05)")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    if not args.skip_tests:
        run("-m", "unittest", "tests.test_model", "tests.test_margin_duality", "tests.test_cert_set")
    run("generate_results.py", *(["--step", str(args.step)] if args.step else []))
    run("stats.py")
    run("plotter.py", "--extended")
    if not args.skip_tests:
        run("-m", "unittest", "tests.test_results")


if __name__ == "__main__":
    main()
