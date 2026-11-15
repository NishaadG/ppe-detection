# PPE Detection

Object detection for construction-site safety. The model finds hard hats, safety vests, goggles and masks, **and flags when one is missing**, so a photo or camera frame can be checked for violations.

**Demo:** _Hugging Face Space link (add after deploying)_

## Classes

| Worn | Missing |
|---|---|
| `hardhat` | `no-hardhat` |
| `vest` | `no-vest` |
| `goggles` | `no-goggles` |
| `mask` | `no-mask` |

## Data

Four public Kaggle datasets, merged into one 8-class dataset in YOLO format by `prepare_data.py`:

| Source | Format | Used for |
|---|---|---|
| [Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) | Pascal VOC XML | hardhat, no-hardhat |
| [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) | Pascal VOC XML | mask, no-mask |
| [Safety Vests Detection](https://www.kaggle.com/datasets/adilshamim8/safety-vests-detection-dataset) | CSV (Roboflow export) | vest, no-vest |
| [PPE Kit Detection](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers) | YOLO | goggles, no-goggles, hardhat, vest, no-hardhat |

`prepare_data.py`:
- converts Pascal VOC and CSV boxes to normalised YOLO boxes, clipping boxes that run outside the image
- maps each source's labels onto the shared class list and drops the rest
- removes exact duplicate images and prefixes file names by source so nothing is overwritten
- makes a **stratified 70/20/10 split** (images grouped by the rarest class they contain), so rare classes reach the test set
- writes **repeat-factor sampling** lists (Gupta et al., LVIS 2019) so a second training run sees images with rare classes more often
- saves class counts per split (`stats.json`, `class_distribution.png`)

## Training and evaluation

`train_colab.ipynb` runs everything on a free Colab T4 and saves results to Google Drive (`train_kaggle.ipynb` does the same on Kaggle):

1. **Baseline:** YOLOv8s on the plain training split
2. **Balanced:** YOLOv8s with repeat-factor sampling for the rare classes
3. **Optional:** YOLOv5s on the same split, to compare architectures

`evaluate.py` scores each run on the held-out test split (mAP@0.5, mAP@0.5:0.95, precision, recall, AP per class, inference time) and writes `results/comparison.md`.

## Results

### v2 (this repo, stratified split)

25 epochs per run, imgsz 640, batch 16, seed 42, on a Kaggle T4 (`train_kaggle.ipynb`). Dataset: 11,126 images (7,788 train / 2,225 valid / 1,113 test), 41,887 objects.

| Metric (test split, 1,113 images) | YOLOv8s baseline | YOLOv8s balanced | YOLOv5s |
|---|---|---|---|
| mAP@0.5 | **0.756** | 0.755 | 0.755 |
| mAP@0.5:0.95 | **0.484** | 0.480 | 0.471 |
| Precision | 0.747 | 0.752 | 0.874 |
| Recall | 0.703 | 0.732 | 0.706 |
| AP@0.5 hardhat | 0.959 | 0.960 | 0.956 |
| AP@0.5 vest | 0.931 | 0.933 | 0.920 |
| AP@0.5 goggles | 0.738 | 0.719 | 0.704 |
| AP@0.5 mask | 0.936 | 0.938 | 0.933 |
| AP@0.5 no-hardhat | 0.860 | 0.852 | 0.862 |
| AP@0.5 no-vest | 0.821 | 0.807 | 0.796 |
| AP@0.5 no-goggles | 0.045 | 0.040 | 0.040 |
| AP@0.5 no-mask | 0.758 | 0.788 | 0.826 |
| Inference ms/image (T4) | 10.9 | 11.2 | 9.2 |

- Best run: YOLOv8s baseline. `goggles` improved from 0.63 in v1 to 0.74.
- Repeat-factor sampling raised recall (0.703 to 0.732) and `no-mask` AP but did not change overall mAP.
- `no-goggles` is still not learned (45 test objects; the model labels them background).
- All three runs scored their best validation mAP at the last epoch, so they were retrained for longer (below).

**Longer training** (same settings, `EPOCHS = 75`, early stopping with patience 10). Both runs stopped early: baseline after 60 epochs (best at 50), balanced after 61 (best at 51), about 5.5 h on a T4 in total.

| Metric (test split, 1,113 images) | Baseline, 25 ep | Baseline, 60 ep | Balanced, 25 ep | Balanced, 61 ep |
|---|---|---|---|---|
| mAP@0.5 | 0.756 | **0.762** | 0.755 | 0.755 |
| mAP@0.5:0.95 | **0.484** | 0.483 | 0.480 | 0.476 |
| Precision | 0.747 | 0.752 | 0.752 | 0.766 |
| Recall | 0.703 | 0.731 | 0.732 | 0.709 |
| AP@0.5 goggles | 0.738 | 0.731 | 0.719 | 0.729 |
| AP@0.5 no-vest | 0.821 | 0.861 | 0.807 | 0.858 |
| AP@0.5 no-goggles | 0.045 | 0.052 | 0.040 | 0.044 |
| AP@0.5 no-mask | 0.758 | 0.766 | 0.788 | 0.743 |

- Training longer gave small gains for the baseline (recall 0.703 to 0.731, `no-vest` 0.821 to 0.861) and left mAP@0.5:0.95 flat; the models converge around epoch 50.
- `no-goggles` stays near zero however long the model trains, so it needs more labelled examples rather than more epochs.

### v1 (first version: YOLOv5s, 12 epochs, random 70/20/10 split of 7,093 images)

| Metric (test split, 710 images) | Value |
|---|---|
| mAP@0.5 | 0.714 |
| Precision | 0.895 |
| Recall | 0.656 |
| AP@0.5 hardhat / mask / no-hardhat / vest | 0.937 / 0.923 / 0.880 / 0.874 |
| Inference | about 11 ms per image (T4) |

In v1, `goggles` (0.63) and especially `no-goggles` (0.02, only 32 test objects) were the weak classes. v2's stratified split and repeat-factor sampling target exactly that.

## Run it yourself

```bash
pip install -r requirements.txt
# download the four datasets into raw/helmet, raw/mask, raw/vest, raw/goggles
python prepare_data.py --raw raw --out data
yolo train model=yolov8s.pt data=data/ppe.yaml epochs=25 imgsz=640
python evaluate.py --data data/ppe.yaml --runs runs/detect/train --out results
```

## Demo app

`space/` holds a Gradio app for Hugging Face Spaces. It draws the detections and lists any missing gear. To deploy, copy `best.pt` into `space/` and push the folder to a new Gradio Space.
