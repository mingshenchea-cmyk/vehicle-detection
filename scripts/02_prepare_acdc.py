"""ACDC (COCO json) -> BARD YOLO labels, sorted into weather folders.

Notebook section 1.2. Expects data/gt_detection_trainval*, data/rgb_anon_trainvaltest.
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

JSON_FILES = [
    config.DATA / "gt_detection_trainval/gt_detection/instancesonly_train_gt_detection.json",
    config.DATA / "gt_detection_trainval/gt_detection/instancesonly_val_gt_detection.json",
    config.DATA / "gt_detection_trainval_ref/gt_detection/instancesonly_train_gt_ref_detection.json",
    config.DATA / "gt_detection_trainval_ref/gt_detection/instancesonly_val_gt_ref_detection.json",
]
IMAGES_DIR = config.DATA / "rgb_anon_trainvaltest/rgb_anon"
DST_IMAGES = config.BARD / "images"
DST_LABELS = config.BARD / "labels"

# ACDC category id -> BARD class id (person 24 / rider 25 are dropped)
CLASS_REMAP = {26: 2, 27: 5, 28: 1, 31: 4, 32: 3, 33: 0}


def weather_of(file_name, image_name):
    if "_ref" in image_name:
        return "clear"
    folder = file_name.split("/")[0].lower()
    for weather in ("fog", "rain", "snow", "night"):
        if weather in folder:
            return weather
    return "unknown"


def main():
    DST_LABELS.mkdir(parents=True, exist_ok=True)
    categories = {}
    image_count = label_count = 0

    for json_file in JSON_FILES:
        data = json.loads(json_file.read_text())
        for item in data["categories"]:
            categories[item["id"]] = item["name"]

        # Index annotations by image id once (same result as the notebook's nested scan)
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        for img in data["images"]:
            file_name = img["file_name"]
            image_name = Path(file_name).name
            width, height = img["width"], img["height"]

            weather_dir = DST_IMAGES / weather_of(file_name, image_name)
            weather_dir.mkdir(parents=True, exist_ok=True)

            lines = []
            for ann in annotations[img["id"]]:
                if ann["category_id"] not in CLASS_REMAP:
                    continue
                x, y, w, h = ann["bbox"]
                lines.append(
                    f"{CLASS_REMAP[ann['category_id']]} {(x + w / 2) / width} {(y + h / 2) / height} {w / width} {h / height}"
                )

            label_file = DST_LABELS / image_name.replace(".png", ".txt")
            label_file.write_text("\n".join(lines))
            label_count += 1

            shutil.copy(IMAGES_DIR / file_name, weather_dir / image_name)
            image_count += 1

    print("Classes:", categories)
    print(f"{image_count} images and {label_count} labels converted to {config.BARD}.")


if __name__ == "__main__":
    main()
