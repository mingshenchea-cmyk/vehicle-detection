"""Fill minority classes up to per-split targets with cropped + augmented instances.

Notebook sections 2.3.1-2.3.4 merged. Counts are read live from the split labels
(the notebook hardcoded them). On the original data this augments 'train' in
clear/fog/night/rain and 'motorcycle' in rain. Writes into BARD/split/{weather}/{split}.
"""
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

LABEL_DIR = config.BARD / "sampled_labels"  # original boxes come from here
TARGET_COUNTS = {"train": 280, "val": 80, "test": 40}


def read_boxes(label_path, class_id):
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        if int(parts[0]) == class_id:
                            boxes.append(tuple(map(float, parts)))
                    except ValueError:
                        continue
    return boxes


def augment(img):
    """Notebook's augmentation stack, each step applied with p=0.5."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    if random.random() < 0.5:
        h, w = img.shape[:2]
        crop_w = random.randint(int(0.8 * w), w)
        crop_h = random.randint(int(0.8 * h), h)
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        crop = img[y1:y1 + crop_h, x1:x1 + crop_w]
        if crop.size == 0:
            crop = img
        img = cv2.resize(crop, (w, h))

    if random.random() < 0.5:
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), random.uniform(-5, 5), 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 1] *= random.uniform(0.7, 1.3)  # saturation
        hsv[..., 2] *= random.uniform(0.7, 1.3)  # brightness
        hsv = np.clip(hsv, 0, 255)
        img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    if random.random() < 0.5:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.uniform(-30, 30))

    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    if random.random() < 0.5:
        k = random.choice([5, 7])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1
        kernel /= k
        img = cv2.filter2D(img, -1, kernel)

    return img


def augment_class(split_dir, class_id, class_name, current_count, target):
    shortfall = target - current_count
    if shortfall <= 0:
        return 0
    print(f"\nProcessing {split_dir.parent.name}/{split_dir.name}: "
          f"Need {shortfall} more instances of '{class_name}'.")

    images = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTS)
    created = 0
    index = 0
    pass_start_created = 0
    while created < shortfall and images:
        img_path = images[index % len(images)]
        # A full pass with no new instances means no usable source boxes remain
        if index > 0 and index % len(images) == 0:
            if created == pass_start_created:
                print(f"WARNING: no more '{class_name}' boxes available; stopping at {created}/{shortfall}.")
                break
            pass_start_created = created

        for box in read_boxes(LABEL_DIR / (img_path.stem + ".txt"), class_id):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"WARNING: cannot read {img_path}")
                break
            h, w = img.shape[:2]
            _, x_c, y_c, bw, bh = box
            x1 = max(0, int((x_c - bw / 2) * w))
            y1 = max(0, int((y_c - bh / 2) * h))
            x2 = min(w, int((x_c + bw / 2) * w))
            y2 = min(h, int((y_c + bh / 2) * h))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            new_name = f"{img_path.stem}_aug{created + current_count}"
            cv2.imwrite(str(split_dir / f"{new_name}.jpg"), augment(crop.copy()))
            (split_dir / f"{new_name}.txt").write_text(f"{class_id} 0.500000 0.500000 1.000000 1.000000\n")
            created += 1
            if created >= shortfall:
                break
        index += 1

    return created


def main():
    total_new = 0
    for weather in config.WEATHERS:
        for split in config.SPLITS:
            split_dir = config.BARD / "split" / weather / split
            if not split_dir.is_dir():
                print(f"Directory {split_dir} not found. Skipping.")
                continue

            # Live instance counts from the labels actually in this split
            counts = Counter()
            for label in split_dir.glob("*.txt"):
                counts.update(config.label_counts(label))

            for class_id, class_name in config.CLASS_MAP.items():
                created = augment_class(split_dir, class_id, class_name,
                                        counts[class_name], TARGET_COUNTS[split])
                counts[class_name] += created
                total_new += created

    print(f"\nAugmentation Complete\nTotal new instances created: {total_new}")


if __name__ == "__main__":
    main()
