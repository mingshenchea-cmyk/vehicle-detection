"""Visual checks: original / ground-truth / prediction panels, or saved metric curves.

Notebook sections 3.5 + 4.1 (panels; --conf 0.25 --iou 0.5 for the post-NMS view)
and 5.3 (--curves RUN_DIR to display the PNGs ultralytics saved).

Examples:
    python scripts/11_visualize.py
    python scripts/11_visualize.py --conf 0.25 --iou 0.5
    python scripts/11_visualize.py --curves runs/overall_val
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

DEFAULT_IMAGES = [
    (config.BARD / "split/clear/val/8ed43a59-574c5f4d.jpg", "Clear"),
    (config.BARD / "split/fog/val/TJ_Google_390.png", "Fog"),
    (config.BARD / "split/night/test/GP010397_frame_000810_rgb_anon.png", "Night"),
    (config.BARD / "split/rain/test/GP020402_frame_000521_rgb_anon.png", "Rain"),
]
CURVE_FILES = ["BoxF1_curve.png", "BoxP_curve.png", "BoxPR_curve.png", "BoxR_curve.png",
               "confusion_matrix.png", "confusion_matrix_normalized.png"]


def ground_truth_image(image_path, class_names):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    label_path = image_path.with_suffix(".txt")
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_names.get(cls_id, str(cls_id)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def show_panels(model, images, thresholds):
    for img_path, title in images:
        if not img_path.exists():
            print(f"WARNING: {img_path} not found, skipping.")
            continue
        original = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(ground_truth_image(img_path, model.names), cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(model(source=str(img_path), **thresholds)[0].plot(), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle(title, fontsize=20, fontweight="bold")
        for ax, img, subtitle in zip(axes, (original, gt, pred),
                                     ("Original Image", "Ground Truth", "Prediction")):
            ax.imshow(img)
            ax.set_title(subtitle)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


def show_curves(run_dir):
    for name in CURVE_FILES:
        path = run_dir / name
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=config.ROOT / "best.pt")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--images", type=Path, nargs="*", help="image paths (default: one per weather)")
    parser.add_argument("--curves", type=Path, help="results dir whose curve PNGs to display")
    args = parser.parse_args()

    if args.curves:
        show_curves(args.curves)
        return

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    thresholds = {k: v for k, v in (("conf", args.conf), ("iou", args.iou)) if v is not None}
    images = [(p, p.parent.parent.name) for p in args.images] if args.images else DEFAULT_IMAGES
    show_panels(model, images, thresholds)


if __name__ == "__main__":
    main()
