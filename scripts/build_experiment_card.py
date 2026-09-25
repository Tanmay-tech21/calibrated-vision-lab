"""Rebuild the experiment card and its machine-readable result snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrated_vision import build_experiment_snapshot, render_experiment_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    arguments = parser.parse_args()

    snapshot = build_experiment_snapshot()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = arguments.output_dir / "experiment_summary.json"
    card_path = arguments.output_dir / "EXPERIMENT_CARD.md"
    json_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    card_path.write_text(render_experiment_card(snapshot))
    print(json.dumps({"experiment_card": str(card_path), "summary": str(json_path)}))


if __name__ == "__main__":
    main()
