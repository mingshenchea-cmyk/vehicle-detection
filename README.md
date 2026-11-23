# Vehicle Detection (YOLO12)

Vehicle detection in adverse weather (clear / fog / night / rain), converted from
`vehicle_detection.ipynb` (kept for reference). `best.pt` is the trained YOLO12s
model (imgsz 480, 6 classes: Bicycle, Bus, Car, Motorcycle, Train, Truck).

## Setup

```bash
pip install -r requirements.txt
```

All data lives under `data/` (or set the `VD_DATA` environment variable to point elsewhere).
Choose your starting point:

### Option A — combined BARD dataset (skip steps 1–4)

Download the ready-combined dataset (~12 GB, all 4 sources already merged and
converted to YOLO format) from
[github.com/mingshenchea-cmyk/BARD-Vehicle-Detection](https://github.com/mingshenchea-cmyk/BARD-Vehicle-Detection)
([Kaggle download](https://www.kaggle.com/datasets/mingshen0118/bard-vehicle-detection)).

Extract it so the merged folders sit at `data/BARD/`:

```
data/BARD/
├── images/
│   ├── clear/  fog/  night/  rain/  sand/  snow/
└── labels/          # flat YOLO .txt files, one per image
```

(If the archive extracts under a different top-level folder name, rename/move it
to `data/BARD`.) Then **skip pipeline steps 1–4** and start at step 5.

### Option B — the 4 raw datasets (run everything)

Place each source under `data/` exactly as below, then run steps 1–4 to build
`data/BARD/` yourself. Nothing can be skipped.

```
data/
├── DAWN/{images,labels}
├── gt_detection_trainval/  gt_detection_trainval_ref/  rgb_anon_trainvaltest/   # ACDC
├── bdd100k_labels/100k/  bdd100k_images_100k/100k/
└── RTTS/{Annotations,JPEGImages}
```

## Pipeline (run in order)

| Step | Command | Purpose |
|---|---|---|
| 1 | `python scripts/01_prepare_dawn.py` | DAWN → BARD (class remap + weather folders) |
| 2 | `python scripts/02_prepare_acdc.py` | ACDC COCO json → YOLO |
| 3 | `python scripts/03_prepare_bdd100k.py` | BDD100K json → YOLO |
| 4 | `python scripts/04_prepare_rtts.py` | RTTS VOC xml → YOLO (all fog) |
| — | `python scripts/analyze.py` | weather × class distribution |
| 5 | `python scripts/05_sample.py` | balanced sampling, 400 instances/class/weather |
| 6 | `python scripts/06_split.py` | instance-balanced 70/20/10 split |
| — | `python scripts/analyze.py --split` | verify split counts |
| 7 | `python scripts/07_augment.py` | fill minority classes (train/motorcycle) with crops |
| 8 | `python scripts/08_finalize.py --size 480` | merge main split, resize, write YAML configs |
| 9 | `python scripts/09_train.py --model s` | train YOLO12 (n/s/m/x), results in `runs/` |
| 10 | `python scripts/10_evaluate.py --split val` | validate overall + per weather |
| 10 | `python scripts/10_evaluate.py --split test --conf 0.25 --iou 0.5` | post-NMS test |
| 11 | `python scripts/11_visualize.py --conf 0.25 --iou 0.5` | original / GT / prediction panels |
| 11 | `python scripts/11_visualize.py --curves runs/overall_val` | display metric curves |
| 12 | `python scripts/12_export.py` | export ONNX |

**Which steps can you skip?**
- Option A (combined BARD dataset): skip **1–4**, start at step 5.
- Option B (raw datasets): run everything in order.
- Already have `BARD/split` + the YAML configs (e.g. from a previous run):
  skip **1–8**, start at step 9.
- Just want to use the trained model (`best.pt`): no dataset needed — run
  `app/dashcam.py`, `11_visualize.py` or `12_export.py` directly.

Notes:
- The notebook's zip → Google Drive → unzip → Colab steps were manual workflow
  and are not part of this pipeline. To train on Colab, upload `data/BARD` yourself.
- For 640/320 variants (notebook's BARD2/BARD3), copy `BARD/split` and run
  `08_finalize.py --dir <copy> --size 640`.
- Video demo (notebook 5.4) is a one-liner:
  `yolo predict model=best.pt source=video.mp4 conf=0.25 iou=0.5 save=True`

## Dashcam app

```bash
python app/dashcam.py                      # webcam
python app/dashcam.py --source video/rain1.mp4
```

Live detection with vehicle count, measured FPS and a Record button
(saves annotated `.avi` files to `recordings/`).
