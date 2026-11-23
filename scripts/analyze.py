"""Class-distribution tables for any pipeline stage.

Notebook sections 2.1.1 / 2.1.6 (weather x class table) and 2.2.2 / 2.3.5
(per-weather split tables, --split).

Examples:
    python scripts/analyze.py                          # BARD/images vs BARD/labels
    python scripts/analyze.py --images BARD/sampled_images --labels BARD/sampled_labels
    python scripts/analyze.py --split                  # verify BARD/split
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def weather_table(images_dir, labels_dir):
    weather_class_counts = defaultdict(Counter)
    for weather_dir in sorted(images_dir.iterdir()):
        if not weather_dir.is_dir():
            continue
        for image in weather_dir.iterdir():
            if image.suffix.lower() in config.IMAGE_EXTS:
                counts = config.label_counts(labels_dir / (image.stem + ".txt"))
                weather_class_counts[weather_dir.name].update(counts)

    all_classes = sorted({c for counts in weather_class_counts.values() for c in counts})
    table = []
    column_totals = Counter()
    for weather in sorted(weather_class_counts):
        counts = weather_class_counts[weather]
        total = sum(counts.values())
        table.append([weather] + [counts[c] for c in all_classes] + [total])
        table.append([f"{weather} (%)"] +
                     [f"{counts[c] / total * 100:.1f}%" if total else "0%" for c in all_classes] +
                     ["100%"])
        column_totals.update(counts)
        column_totals["Total"] += total

    table.append(["Total"] + [column_totals[c] for c in all_classes] + [column_totals["Total"]])
    overall = column_totals["Total"]
    table.append(["Total (%)"] +
                 [f"{column_totals[c] / overall * 100:.1f}%" if overall else "0%" for c in all_classes] +
                 ["100%"])
    print(tabulate(table, headers=["Weather"] + all_classes + ["Total"]))


def split_table(split_root):
    weather_dirs = sorted(d for d in split_root.iterdir()
                          if d.is_dir() and d.name not in config.SPLITS)
    for weather_dir in weather_dirs:
        counts = {s: Counter() for s in config.SPLITS}
        for split in config.SPLITS:
            split_path = weather_dir / split
            if not split_path.is_dir():
                continue
            for image in split_path.iterdir():
                if image.suffix.lower() in config.IMAGE_EXTS:
                    counts[split].update(config.label_counts(split_path / (image.stem + ".txt")))

        all_classes = sorted({c for counter in counts.values() for c in counter})
        print(f"\n{weather_dir.name} weather")
        header = f"| {'Class':12} | {'Train':8} | {'Val':8} | {'Test':8} | {'Total':8} |"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for name in all_classes:
            a, b, c = (counts[s][name] for s in config.SPLITS)
            print(f"| {name:12} | {a:8} | {b:8} | {c:8} | {a + b + c:8} |")
        print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=config.BARD / "images")
    parser.add_argument("--labels", type=Path, default=config.BARD / "labels")
    parser.add_argument("--split", action="store_true",
                        help="show per-weather train/val/test tables from BARD/split")
    args = parser.parse_args()
    if args.split:
        split_table(config.BARD / "split")
    else:
        weather_table(args.images, args.labels)


if __name__ == "__main__":
    main()
