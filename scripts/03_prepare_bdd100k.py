"""BDD100K (per-image json) -> BARD YOLO labels, sorted into weather folders.

Notebook section 1.3. Expects data/bdd100k_labels/100k and data/bdd100k_images_100k/100k.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

JSON_DIRS = [config.DATA / "bdd100k_labels/100k" / s for s in ("test", "train", "val")]
IMAGE_DIRS = [config.DATA / "bdd100k_images_100k/100k" / s for s in ("test", "train", "val")]
DST_IMAGES = config.BARD / "images"
DST_LABELS = config.BARD / "labels"

# BDD100K category -> BARD class id (everything else is dropped)
CLASS_REMAP = {"car": 2, "truck": 5, "bus": 1, "train": 4, "motor": 3, "bike": 0}

# BDD100K images are a fixed 1280x720
IMG_W, IMG_H = 1280, 720


def weather_of(attributes):
    if attributes.get("timeofday") == "night":
        return "night"
    weather = attributes.get("weather", "")
    if weather == "foggy":
        return "fog"
    if weather == "rainy":
        return "rain"
    if weather == "snowy":
        return "snow"
    return "clear"  # clear / partly cloudy / overcast / undefined


def main():
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    # Class check + json lookup by image stem
    categories = set()
    json_lookup = {}
    for json_dir in JSON_DIRS:
        for file in sorted(json_dir.rglob("*.json")):
            json_lookup[file.stem] = file
            data = json.loads(file.read_text())
            for frame in data.get("frames", []):
                for obj in frame.get("objects", []):
                    if "category" in obj:
                        categories.add(obj["category"])
    print("Classes:", categories)

    image_count = label_count = 0
    for images_dir in IMAGE_DIRS:
        for image_path in sorted(images_dir.rglob("*.jpg")):
            stem = image_path.stem
            data = json.loads(json_lookup[stem].read_text())

            weather_dir = DST_IMAGES / weather_of(data.get("attributes", {}))
            weather_dir.mkdir(parents=True, exist_ok=True)

            lines = []
            for frame in data.get("frames", []):
                for obj in frame.get("objects", []):
                    if "box2d" not in obj or obj["category"] not in CLASS_REMAP:
                        continue
                    box = obj["box2d"]
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    lines.append(
                        f"{CLASS_REMAP[obj['category']]} {((x1 + x2) / 2) / IMG_W} {((y1 + y2) / 2) / IMG_H} "
                        f"{(x2 - x1) / IMG_W} {(y2 - y1) / IMG_H}"
                    )

            (DST_LABELS / f"{stem}.txt").write_text("\n".join(lines))
            label_count += 1

            shutil.copy(image_path, weather_dir / image_path.name)
            image_count += 1

    print(f"{image_count} images and {label_count} labels converted to {config.BARD}.")


if __name__ == "__main__":
    main()
