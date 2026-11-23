"""Validate or test a trained model, overall and per weather.

Notebook sections 3.3 / 3.4 (raw) and 5.1 / 5.2 (post-NMS: --conf 0.25 --iou 0.5).

Examples:
    python scripts/10_evaluate.py --split val
    python scripts/10_evaluate.py --split test --weather fog
    python scripts/10_evaluate.py --split val --conf 0.25 --iou 0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=config.ROOT / "best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--weather", choices=["all"] + config.WEATHERS, default="all")
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold (post-NMS: 0.25)")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold (post-NMS: 0.5)")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    thresholds = {k: v for k, v in (("conf", args.conf), ("iou", args.iou)) if v is not None}

    runs = ["overall"] + config.WEATHERS if args.weather == "all" else [args.weather]
    for weather in runs:
        yaml_name = "BARD.yaml" if weather == "overall" else f"BARD_{weather}.yaml"
        model.val(
            data=str(config.BARD / yaml_name),
            split=args.split,
            project=str(config.RUNS),
            name=f"{weather}_{args.split}",
            **thresholds,
        )


if __name__ == "__main__":
    main()
