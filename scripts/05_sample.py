"""Balanced sampling: pick images per weather until each class has TARGET instances.

Notebook sections 2.1.2-2.1.5 (one cell per weather, identical logic) merged.
Images that would overshoot a class target fall back to single-object crops.
Output: BARD/sampled_images/{weather} + BARD/sampled_labels.
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

TARGET = 400  # instances per class per weather
TARGET_ORDER = ["bicycle", "bus", "car", "motorcycle", "train", "truck"]


def sample_weather(weather):
    src_img = config.BARD / "images" / weather
    dst_img = config.BARD / "sampled_images" / weather
    src_lbl = config.BARD / "labels"
    dst_lbl = config.BARD / "sampled_labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    targets = {name: TARGET for name in TARGET_ORDER}
    class_counts = Counter()
    used_images = set()
    image_count = 0
    crop_counter = 0

    # Load images with their class counts, fewest cars first (full images before dense ones)
    images = []
    for fname in sorted(p.name for p in src_img.iterdir() if p.suffix.lower() in config.IMAGE_EXTS):
        counts = config.label_counts(src_lbl / (Path(fname).stem + ".txt"))
        if counts:
            images.append((fname, counts))
    images.sort(key=lambda item: item[1]["car"])

    def crop_and_relabel(img_path, lbl_path):
        """Crop each still-needed object into its own full-frame image + label."""
        nonlocal crop_counter
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return 0, 0

        img_w, img_h = img.size
        base_name = img_path.stem
        added = 0
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])
                except ValueError:
                    continue
                name = config.CLASS_MAP.get(class_id)
                if name not in targets or class_counts[name] >= targets[name]:
                    continue

                x1 = max(0, int((x_c - w / 2) * img_w))
                y1 = max(0, int((y_c - h / 2) * img_h))
                x2 = min(img_w, int((x_c + w / 2) * img_w))
                y2 = min(img_h, int((y_c + h / 2) * img_h))
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue

                crop_counter += 1
                new_name = f"{base_name}_crop_{crop_counter}"
                img.crop((x1, y1, x2, y2)).save(dst_img / f"{new_name}.png")
                (dst_lbl / f"{new_name}.txt").write_text(f"{class_id} 0.500000 0.500000 1.000000 1.000000\n")
                class_counts[name] += 1
                added += 1
        return 1, added

    for target in TARGET_ORDER:
        print(f"\n=== [{weather}] Sampling class '{target}' until {targets[target]} instances ===")
        made_progress = True
        while class_counts[target] < targets[target] and made_progress:
            made_progress = False
            for fname, counts in images:
                if fname in used_images or counts[target] == 0:
                    continue
                img_counts = Counter({c: n for c, n in counts.items() if c in targets})
                if not img_counts:
                    continue

                img_path = src_img / fname
                lbl_path = src_lbl / (Path(fname).stem + ".txt")

                # If this image would make ANY class exceed its target, crop instead
                if any(class_counts[c] + n > targets[c] for c, n in img_counts.items()):
                    added_images, added_instances = crop_and_relabel(img_path, lbl_path)
                    if added_instances > 0:
                        used_images.add(fname)
                        image_count += added_images
                        made_progress = True
                        if class_counts[target] >= targets[target]:
                            break
                    continue

                shutil.copy(img_path, dst_img / fname)
                shutil.copy(lbl_path, dst_lbl / lbl_path.name)
                class_counts.update(img_counts)
                used_images.add(fname)
                image_count += 1
                made_progress = True
                if class_counts[target] >= targets[target]:
                    break

            if not made_progress:
                print(f"WARNING: stalled while sampling '{target}'. "
                      "No further unused images can be added without overshooting.")
                break

    print(f"\n=== [{weather}] Sampling completed ===")
    print(f"Total sampled images: {image_count}")
    total = sum(class_counts[c] for c in TARGET_ORDER)
    print("Final class counts with percentages:")
    for name in TARGET_ORDER:
        pct = class_counts[name] / total * 100 if total else 0
        print(f"{name:12}: {class_counts[name]:5} / {targets[name]}   ({pct:6.2f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", choices=config.WEATHERS, help="sample one weather only (default: all)")
    args = parser.parse_args()
    for weather in [args.weather] if args.weather else config.WEATHERS:
        sample_weather(weather)


if __name__ == "__main__":
    main()
