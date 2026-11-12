"""Run the full early-warning analysis.

    python run_analysis.py --data data/sensor.csv --out reports
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from early_warning import Config, run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/sensor.csv", help="path to the pump sensor CSV")
    p.add_argument("--out", default="reports", help="folder for charts and results")
    p.add_argument("--z", type=float, default=Config.z_threshold, help="band threshold in robust SDs")
    p.add_argument("--window-h", type=int, default=Config.warning_window_h, help="warning window in hours")
    args = p.parse_args()

    cfg = Config(z_threshold=args.z, warning_window_h=args.window_h)
    summary = run(args.data, args.out, cfg)
    print(summary.to_string(index=False))
    print(f"\nCharts and full results written to {args.out}/")


if __name__ == "__main__":
    main()
