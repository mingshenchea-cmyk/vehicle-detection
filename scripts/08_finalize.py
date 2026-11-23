"""Finalize the dataset: merge weather splits into the main split, resize, write YAMLs.

Notebook sections 2.3.6 + 2.4 + 2.5. (The Colab zip/mount/unzip steps were manual
workflow and are intentionally not part of the pipeline.)
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def copy_to_main_split():
    count = 0
    for weather in config.WEATHERS:
        for split in config.SPLITS:
            src = config.BARD / "split" / weather / split
            dst = config.BARD / "split" / split
            dst.mkdir(parents=True, exist_ok=True)
            for file in sorted(src.iterdir()):
                shutil.copy2(file, dst / file.name)
                count += 1
    print(f"{count} files copied into the main split directories.")


def resize_images(image_dir, size):
    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() in (".jpg", ".png"):
            img = cv2.imread(str(path))
            if img is None:
                print(f"WARNING: cannot read {path}")
                continue
            cv2.imwrite(str(path), cv2.resize(img, (size, size)))
    print(f"Image resize to {size}x{size} completed.")


def write_configs():
    def dump(data, path):
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"{path.name} created successfully!")

    dump({"train": "./split/train", "val": "./split/val", "test": "./split/test",
          "nc": len(config.CLASS_NAMES), "names": config.CLASS_NAMES},
         config.BARD / "BARD.yaml")

    for weather in config.WEATHERS:
        dump({"train": f"./split/{weather}/train", "val": f"./split/{weather}/val",
              "test": f"./split/{weather}/test",
              "nc": len(config.CLASS_NAMES), "names": config.CLASS_NAMES},
             config.BARD / f"BARD_{weather}.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=480, help="resize images to NxN (default 480)")
    parser.add_argument("--dir", type=Path, default=config.BARD / "split",
                        help="split root to resize (default BARD/split)")
    args = parser.parse_args()

    copy_to_main_split()
    resize_images(args.dir, args.size)
    write_configs()


if __name__ == "__main__":
    main()
