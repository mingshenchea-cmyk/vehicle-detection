"""Train a YOLO12 model on BARD. Notebook section 3.2 (n/s/m/x cells merged).

Example: python scripts/09_train.py --model s
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["n", "s", "m", "x"], default="s")
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(f"yolo12{args.model}.pt")  # COCO-pretrained
    model.train(
        data=str(config.BARD / "BARD.yaml"),
        imgsz=args.imgsz,
        patience=5,
        amp=True,
        project=str(config.RUNS),
        name=f"yolo12{args.model}",
    )


if __name__ == "__main__":
    main()
