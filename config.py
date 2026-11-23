"""Shared paths and constants for the vehicle-detection pipeline."""
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Raw datasets (DAWN/, RTTS/, bdd100k_*, ...) and the merged BARD/ dataset live here.
# Override with the VD_DATA environment variable if your data lives elsewhere.
DATA = Path(os.environ.get("VD_DATA", ROOT / "data"))
BARD = DATA / "BARD"
RUNS = ROOT / "runs"

CLASS_MAP = {0: "bicycle", 1: "bus", 2: "car", 3: "motorcycle", 4: "train", 5: "truck"}
CLASS_NAMES = ["Bicycle", "Bus", "Car", "Motorcycle", "Train", "Truck"]  # YOLO yaml order
WEATHERS = ["clear", "fog", "night", "rain"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def label_counts(label_path):
    """Counter of class names in a YOLO label file (empty Counter if missing)."""
    counts = Counter()
    path = Path(label_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    name = CLASS_MAP.get(int(parts[0]))
                except ValueError:
                    continue
                if name:
                    counts[name] += 1
    return counts
