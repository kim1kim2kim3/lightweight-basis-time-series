"""Convert raw electricity.txt / traffic.txt (comma-separated, no header) to Informer-style CSV."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path


def txt_to_csv(txt_path: Path, csv_path: Path) -> None:
    rows: list[list[float]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",")])
    if not rows:
        raise SystemExit(f"empty: {txt_path}")
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise SystemExit(f"ragged rows in {txt_path}")

    start = datetime(2016, 7, 1, 2, 0, 0)
    header = ["date"] + [str(i) for i in range(n - 1)] + ["OT"]
    with open(csv_path, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(header)
        for i, r in enumerate(rows):
            ts = start + timedelta(hours=i)
            w.writerow(
                [ts.strftime("%Y-%m-%d %H:%M:%S")]
                + [f"{x:.6f}" for x in r[:-1]]
                + [f"{r[-1]:.6f}"]
            )


def main() -> None:
    here = Path(__file__).resolve().parent
    txt_to_csv(here / "electricity.txt", here / "electricity.csv")
    txt_to_csv(here / "traffic.txt", here / "traffic.csv")
    ecl = here / "ecl.csv"
    if ecl.exists():
        ecl.unlink()
    os.link(here / "electricity.csv", ecl)


if __name__ == "__main__":
    main()
