"""Export a trained model. Notebook section 6.1 (ONNX).

Example: python scripts/12_export.py --format onnx
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=config.ROOT / "best.pt")
    parser.add_argument("--format", default="onnx")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.export(format=args.format)


if __name__ == "__main__":
    main()
