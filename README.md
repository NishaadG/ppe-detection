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

`train_colab.ipynb` runs everything on a free Colab T4 and saves results to Google Drive:

1. **Baseline:** YOLOv8s on the plain training split
2. **Balanced:** YOLOv8s with repeat-factor sampling for the rare classes
3. **Optional:** YOLOv5s on the same split, to compare architectures

`evaluate.py` scores each run on the held-out test split (mAP@0.5, mAP@0.5:0.95, precision, recall, AP per class, inference time) and writes `results/comparison.md`.

## Results

### v2 (this repo, stratified split)
_Table from `results/comparison.md` goes here._

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
