"""DAWN (YOLO txt labels) -> BARD: remap classes, sort images into weather folders.

Notebook section 1.1. Run once; expects data/DAWN/{images,labels}.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SRC_IMAGES = config.DATA / "DAWN" / "images"
SRC_LABELS = config.DATA / "DAWN" / "labels"
DST_IMAGES = config.BARD / "images"
DST_LABELS = config.BARD / "labels"

# DAWN class id -> BARD class id (person is dropped)
CLASS_REMAP = {2: 0, 3: 2, 4: 3, 6: 1, 7: 4, 8: 5}

WEATHER_PREFIXES = {
    "sand": ["dusttornado", "sand_storm_g2_", "sand_storm_g2", "sand_storm"],
    "fog": ["foggy", "haze", "mist"],
    "rain": ["rain_storm"],
    "snow": ["snow_storm"],
}


def weather_of(filename):
    prefix = filename.split("-")[0].lower()
    for weather, prefixes in WEATHER_PREFIXES.items():
        if prefix in prefixes:
            return weather
    return None


def main():
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    # Class check: first label file seen per class id
    class_file = {}
    for file in sorted(SRC_LABELS.iterdir()):
        if file.suffix != ".txt":
            continue
        with open(file) as f:
            for line in f:
                if line.strip():
                    class_file.setdefault(int(line.split()[0]), file.name)
    print("Class IDs found and example file:")
    for cls, name in sorted(class_file.items()):
        print(f"  Class {cls}: {name}")

    image_count = label_count = 0
    for img in sorted(SRC_IMAGES.iterdir()):
        weather = weather_of(img.name)
        if weather is None:
            print(f"WARNING: unknown weather prefix, skipping {img.name}")
            continue
        dst_dir = DST_IMAGES / weather
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dst_dir / img.name)
        image_count += 1

    for label in sorted(SRC_LABELS.iterdir()):
        if label.suffix != ".txt":
            continue
        lines = []
        with open(label) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_class = int(parts[0])
                if old_class not in CLASS_REMAP:
                    continue
                parts[0] = str(CLASS_REMAP[old_class])
                lines.append(" ".join(parts) + "\n")
        with open(DST_LABELS / label.name, "w") as f:
            f.writelines(lines)
        label_count += 1

    print(f"{image_count} images and {label_count} labels copied to {config.BARD}.")


if __name__ == "__main__":
    main()
