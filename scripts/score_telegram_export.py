from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dialog_emo_demo.pipeline import MODEL_REGISTRY, score_dialog, write_scored_csv
from dialog_emo_demo.telegram import load_telegram_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a Telegram JSON export into a dialog CSV.")
    parser.add_argument("--input", required=True, type=Path, help="result.json from Telegram Desktop")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV path")
    parser.add_argument(
        "--model",
        default="uniform",
        choices=sorted(MODEL_REGISTRY),
        help="Registered model name (see MODEL_REGISTRY in pipeline.py)",
    )
    parser.add_argument(
        "--parsed-output",
        type=Path,
        default=None,
        help="Optional: dump the parsed dialog (without emotion columns) here for inspection",
    )
    args = parser.parse_args()

    frame = load_telegram_export(args.input)
    if args.parsed_output is not None:
        args.parsed_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.parsed_output, index=False)
        print(f"parsed {len(frame)} rows -> {args.parsed_output}")

    scored = score_dialog(frame, MODEL_REGISTRY[args.model]())
    written = write_scored_csv(scored, args.output)
    print(f"wrote {len(scored)} rows -> {written}")


if __name__ == "__main__":
    main()
