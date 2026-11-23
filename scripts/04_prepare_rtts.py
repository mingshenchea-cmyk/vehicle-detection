"""RTTS (VOC xml) -> BARD YOLO labels. All RTTS images are foggy.

Notebook section 1.4. Expects data/RTTS/{Annotations,JPEGImages}.
"""
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

XML_DIR = config.DATA / "RTTS" / "Annotations"
IMAGES_DIR = config.DATA / "RTTS" / "JPEGImages"
DST_IMAGES = config.BARD / "images" / "fog"
DST_LABELS = config.BARD / "labels"

# RTTS name -> BARD class id (person is dropped)
CLASS_REMAP = {"car": 2, "bus": 1, "motorbike": 3, "bicycle": 0}


def main():
    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    categories = set()
    image_count = label_count = 0

    for xml_file in sorted(XML_DIR.iterdir()):
        root = ET.parse(xml_file).getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            categories.add(name)
            if name not in CLASS_REMAP:
                continue
            box = obj.find("bndbox")
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)
            lines.append(
                f"{CLASS_REMAP[name]} {(xmin + xmax) / 2 / width} {(ymin + ymax) / 2 / height} "
                f"{(xmax - xmin) / width} {(ymax - ymin) / height}"
            )

        stem = xml_file.stem
        (DST_LABELS / f"{stem}.txt").write_text("\n".join(lines))
        label_count += 1

        shutil.copy(IMAGES_DIR / f"{stem}.png", DST_IMAGES / f"{stem}.png")
        image_count += 1

    print("Classes:", categories)
    print(f"{image_count} images and {label_count} labels converted to {config.BARD}.")


if __name__ == "__main__":
    main()
