from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train_and_compare


if __name__ == "__main__":
    report = train_and_compare()
    print(report["selected_model"]["name"])
