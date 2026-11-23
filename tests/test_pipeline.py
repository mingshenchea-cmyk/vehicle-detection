"""Self-check: run the data pipeline (05->06->07->08 + analyze) on tiny synthetic data.

    python tests/test_pipeline.py

Uses a temp dir via VD_DATA; never touches real data. Targets are scaled down.
"""
import importlib.util
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="vd_test_"))
os.environ["VD_DATA"] = str(TMP)
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

import config


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_data():
    random.seed(0)
    (config.BARD / "labels").mkdir(parents=True, exist_ok=True)
    for weather in config.WEATHERS:
        img_dir = config.BARD / "images" / weather
        img_dir.mkdir(parents=True)
        for i in range(20):
            stem = f"{weather}_{i}"
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
            lines = []
            for _ in range(random.randint(1, 4)):
                cls = random.choice(list(config.CLASS_MAP))
                cx, cy = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
                lines.append(f"{cls} {cx} {cy} 0.2 0.2")
            (config.BARD / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    (config.BARD / "labels").mkdir(parents=True, exist_ok=True)


def split_counts(split_dir):
    counts = Counter()
    for label in split_dir.glob("*.txt"):
        counts.update(config.label_counts(label))
    return counts


def main():
    make_data()

    sample = load_script("05_sample")
    sample.TARGET = 8
    sample.main()
    for weather in config.WEATHERS:
        sampled = config.BARD / "sampled_images" / weather
        assert sampled.is_dir(), f"no sampled images for {weather}"
        counts = Counter()
        for img in sampled.iterdir():
            counts.update(config.label_counts(config.BARD / "sampled_labels" / (img.stem + ".txt")))
        assert all(n <= 8 for n in counts.values()), f"overshoot in {weather}: {counts}"

    split = load_script("06_split")
    split.main()
    main_files = []
    for s in config.SPLITS:
        main_files += [p.name for p in (config.BARD / "split" / s).iterdir() if p.suffix in config.IMAGE_EXTS]
    weather_files = [p.name for w in config.WEATHERS for s in config.SPLITS
                     for p in (config.BARD / "split" / w / s).iterdir() if p.suffix in config.IMAGE_EXTS]
    assert len(main_files) == len(set(main_files)), "image assigned twice in main split"
    assert sorted(main_files) == sorted(weather_files), "main split != weather splits"

    augment = load_script("07_augment")
    augment.TARGET_COUNTS = {"train": 6, "val": 3, "test": 2}
    # The split step may force-overshoot targets (notebook behavior); augmentation
    # must never add beyond max(target, what was already there).
    pre = {(w, s): split_counts(config.BARD / "split" / w / s)
           for w in config.WEATHERS for s in config.SPLITS}
    augment.main()
    aug_found = False
    for weather in config.WEATHERS:
        for s in config.SPLITS:
            d = config.BARD / "split" / weather / s
            counts = split_counts(d)
            for name, n in counts.items():
                limit = max(augment.TARGET_COUNTS[s], pre[(weather, s)][name])
                assert n <= limit, f"aug overshoot {weather}/{s}: {counts}"
            aug_found |= any("_aug" in p.name for p in d.iterdir())
    assert aug_found, "no augmented images created"

    sys.argv = ["08_finalize.py", "--size", "480"]
    finalize = load_script("08_finalize")
    finalize.main()
    data = yaml.safe_load((config.BARD / "BARD.yaml").read_text())
    assert data["nc"] == 6 and data["names"] == config.CLASS_NAMES
    for weather in config.WEATHERS:
        assert (config.BARD / f"BARD_{weather}.yaml").exists()
    sample_img = next(p for p in (config.BARD / "split" / "train").iterdir() if p.suffix in config.IMAGE_EXTS)
    assert cv2.imread(str(sample_img)).shape[:2] == (480, 480), "resize failed"

    sys.argv = ["analyze.py", "--split"]
    load_script("analyze").main()

    print(f"\nALL CHECKS PASSED (synthetic data in {TMP})")


if __name__ == "__main__":
    main()
