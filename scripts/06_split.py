"""Instance-balanced 70/20/10 split of the sampled data, per weather.

Notebook section 2.2.1. Greedy assignment with a swap/fallback window so each
split's class counts stay close to target. Copies into BARD/split/{split} and
BARD/split/{weather}/{split}. Run analyze.py --split afterwards to verify.
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

IMAGE_ROOT = config.BARD / "sampled_images"
LABEL_DIR = config.BARD / "sampled_labels"
OUTPUT_DIR = config.BARD / "split"

TRAIN_RATIO, VAL_RATIO = 0.7, 0.2  # test gets the remainder
SWAP_SEARCH_WINDOW = 50


def copy_files(file_list, split):
    main_dir = OUTPUT_DIR / split
    main_dir.mkdir(parents=True, exist_ok=True)
    for img_path, weather in file_list:
        label_name = img_path.stem + ".txt"
        src_lbl = LABEL_DIR / label_name

        shutil.copy(img_path, main_dir / img_path.name)
        if src_lbl.exists():
            shutil.copy(src_lbl, main_dir / label_name)

        weather_dir = OUTPUT_DIR / weather / split
        weather_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, weather_dir / img_path.name)
        if src_lbl.exists():
            shutil.copy(src_lbl, weather_dir / label_name)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_split_files = {s: [] for s in config.SPLITS}

    for weather_dir in sorted(IMAGE_ROOT.iterdir()):
        if not weather_dir.is_dir():
            continue
        weather = weather_dir.name
        print(f"\nProcessing Weather: {weather}")

        weather_images = []  # (img_path, counts, unique_id)
        weather_totals = Counter()
        for idx, img_path in enumerate(sorted(weather_dir.iterdir())):
            if img_path.suffix.lower() not in config.IMAGE_EXTS:
                continue
            counts = config.label_counts(LABEL_DIR / (img_path.stem + ".txt"))
            if counts:
                weather_images.append((img_path, counts, f"{weather}_{idx}"))
                weather_totals.update(counts)

        if not weather_images:
            print(f"No images found or labels missing in {weather}. Skipping.")
            continue

        # Full images first, cropped images last
        weather_images.sort(key=lambda item: 1 if "_crop_" in item[0].name.lower() else 0)

        targets = {}
        for name, total in weather_totals.items():
            train_target = int(total * TRAIN_RATIO)
            val_target = int(total * VAL_RATIO)
            targets[name] = {"train": train_target, "val": val_target,
                             "test": total - train_target - val_target}

        split_files = {s: [] for s in config.SPLITS}
        current = {s: Counter() for s in config.SPLITS}
        unassigned = list(weather_images)

        while unassigned:
            img_path, img_counts, _uid = unassigned.pop(0)

            # Strict greedy: best split that does not overshoot any class target
            best_split, best_score = None, -1
            for split in config.SPLITS:
                if any(current[split][c] + n > targets[c][split] for c, n in img_counts.items()):
                    continue
                score = sum(min(n, targets[c][split] - current[split][c]) for c, n in img_counts.items())
                if score > best_score:
                    best_split, best_score = split, score

            if best_split:
                split_files[best_split].append((img_path, weather))
                current[best_split].update(img_counts)
                continue

            # Swap/fallback: find the least complete split for this image's classes
            completeness = {}
            for split in config.SPLITS:
                target_sum = sum(targets[c][split] for c in img_counts if c in targets)
                if target_sum == 0:
                    completeness[split] = len(weather_images) * 100
                else:
                    completeness[split] = sum(current[split][c] for c in img_counts if c in targets) / target_sum
            needy_split = min(completeness, key=completeness.get)

            # Look for a replacement image near the end of the queue that fits
            pool = unassigned[-SWAP_SEARCH_WINDOW:]
            replacement = None
            for i, (r_path, r_counts, r_uid) in enumerate(pool):
                if all(current[needy_split][c] + n <= targets[c][needy_split] for c, n in r_counts.items()):
                    replacement = (r_path, r_counts, r_uid)
                    replacement_index = len(unassigned) - len(pool) + i
                    break

            if replacement:
                r_path, r_counts, _ = replacement
                split_files[needy_split].append((r_path, weather))
                current[needy_split].update(r_counts)
                unassigned.pop(replacement_index)
                unassigned.append((img_path, img_counts, _uid))
            else:
                # No swap possible: force assignment, accepting overshoot
                split_files[needy_split].append((img_path, weather))
                current[needy_split].update(img_counts)

        for split in config.SPLITS:
            global_split_files[split].extend(split_files[split])
        print(f"Images Split: Train={len(split_files['train'])}, "
              f"Val={len(split_files['val'])}, Test={len(split_files['test'])}")

    for split in config.SPLITS:
        copy_files(global_split_files[split], split)

    print("\nFinal Image Counts:")
    print(", ".join(f"{s.capitalize()}={len(global_split_files[s])}" for s in config.SPLITS))


if __name__ == "__main__":
    main()
