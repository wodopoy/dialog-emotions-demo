from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dialog_emo_demo.schema import DEFAULT_DATA_PATH
from dialog_emo_demo.synthetic import write_synthetic_dialog


def main() -> None:
    path = write_synthetic_dialog(DEFAULT_DATA_PATH)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
