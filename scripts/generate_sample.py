"""Regenerate data/sample_franke_sales.csv."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franforecast.synthetic import generate_sales  # noqa: E402

OUTPUT = ROOT / "data" / "sample_franke_sales.csv"


def main() -> None:
    frame = generate_sales()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(
        f"Wrote {len(frame):,} rows, {frame['Item_Code'].nunique()} items x "
        f"{frame['Region'].nunique()} regions, {frame['Date'].min()} to {frame['Date'].max()} -> {OUTPUT}"
    )


if __name__ == "__main__":
    main()
