"""One-shot cleanup helper for auto diagnostics logs.

Usage:
    python cleanup_diagnostics.py
    python cleanup_diagnostics.py --all
"""

import argparse

from auto_diagnostics import cleanup_auto_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Delete diagnostics logs created for crash debugging.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also delete the entire run_diagnostics directory.",
    )
    args = parser.parse_args()

    cleanup_auto_diagnostics(remove_all_run_diagnostics=args.all)
    if args.all:
        print("Deleted run_diagnostics (all diagnostics).")
    else:
        print("Deleted run_diagnostics/auto_global (automatic diagnostics only).")


if __name__ == "__main__":
    main()
