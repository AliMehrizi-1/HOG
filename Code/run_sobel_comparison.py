"""Run qualitative, quantitative, and MultiCue Sobel comparisons in order."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = Path("configs/sobel_comparison.json")
EXPERIMENTS = (
    "qualitative_sobel_comparison.py",
    "quantitative_sobel_comparison.py",
    "multicue_edge_evaluation.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute all final Sobel comparison experiments in order."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Shared JSON config path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute each experiment and stop on a real failure."""
    args = parse_args(argv)
    for script in EXPERIMENTS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run(
            [sys.executable, script, "--config", str(args.config)],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
