"""Build one 8-class PPE dataset in YOLO format from four public Kaggle datasets.

Expected layout after downloading (see train_colab.ipynb):
    raw/helmet   andrewmvd/hard-hat-detection                       (Pascal VOC XML)
    raw/mask     andrewmvd/face-mask-detection                      (Pascal VOC XML)
    raw/vest     adilshamim8/safety-vests-detection-dataset         (Roboflow CSV: <split>/<split>/_annotations.csv)
    raw/goggles  ketakichalke/ppe-kit-detection-construction-site-workers (YOLO txt)

Usage:
    python prepare_data.py --raw raw --out data
"""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["hardhat", "vest", "goggles", "mask", "no-hardhat", "no-vest", "no-goggles", "no-mask"]
CID = {name: i for i, name in enumerate(CLASSES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Source label -> unified class. Unmapped source classes are dropped.
VOC_MAP = {
    "helmet": {"helmet": "hardhat", "head": "no-hardhat"},
    "mask": {"with_mask": "mask", "without_mask": "no-mask"},
}
# The vest dataset is a Roboflow "TensorFlow CSV" export: one _annotations.csv per split with
# class names, not ids ("NO-Safety Vest" and "Safety Vest"; ids 0 and 1 in its YOLO export).
CSV_MAP = {
    "vest": {"Safety Vest": "vest", "NO-Safety Vest": "no-vest"},
}
# ppe-kit ids (from the Kaggle description): 0 Helmet, 1 Gloves, 2 Vest, 3 Boots, 4 Goggles, 5 none,
# 6 Person, 7 no_helmet, 8 no_goggle, 9 no_gloves, 10 no_boots
YOLO_MAP = {
    "goggles": {0: "hardhat", 2: "vest", 4: "goggles", 7: "no-hardhat", 8: "no-goggles"},
}


def box_to_yolo(cls, x1, y1, x2, y2, w, h):
    """Pixel corners -> normalised YOLO box, clipped to the image. None if the box is degenerate."""
    x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def voc_to_yolo(xml_path, cmap):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = float(size.find("width").text), float(size.find("height").text)
    if w <= 0 or h <= 0:
        return []
    out = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in cmap:
            continue
        b = obj.find("bndbox")
        box = box_to_yolo(CID[cmap[name]], *(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")), w, h)
        if box:
            out.append(box)
    return out


def read_csv(csv_path, cmap):
    """Roboflow _annotations.csv -> {image_path: labels}. Images sit next to the CSV."""
    out = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["class"].strip()
            if name not in cmap:
                continue
            w, h = float(row["width"]), float(row["height"])
            if w <= 0 or h <= 0:
                continue
            box = box_to_yolo(CID[cmap[name]], *(float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")), w, h)
            if box:
                out[csv_path.parent / row["filename"]].append(box)
    return out


def read_yolo(txt_path, cmap):
    out = []
    for line in txt_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        src = int(float(p[0]))
        if src not in cmap:
            continue
        xc, yc, bw, bh = (min(max(float(v), 0.0), 1.0) for v in p[1:])
        if bw <= 0 or bh <= 0:
            continue
        out.append((CID[cmap[src]], xc, yc, bw, bh))
    return out


def collect(raw):
    """Return a list of (source, image_path, labels)."""
    items = []
    for src, cmap in VOC_MAP.items():
        for xml in sorted((raw / src).rglob("*.xml")):
            img = next((p for p in (xml.parent.parent / "images").glob(xml.stem + ".*") if p.suffix.lower() in IMG_EXT), None)
            if img is None:
                continue
            labels = voc_to_yolo(xml, cmap)
            if labels:
                items.append((src, img, labels))
    for src, cmap in CSV_MAP.items():
        for csv_path in sorted((raw / src).rglob("_annotations.csv")):
            for img, labels in sorted(read_csv(csv_path, cmap).items()):
                if img.exists() and img.suffix.lower() in IMG_EXT:
                    items.append((src, img, labels))
    for src, cmap in YOLO_MAP.items():
        for img in sorted((raw / src).rglob("*")):
            if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
                continue
            parts = list(img.parts)
            i = len(parts) - 1 - parts[::-1].index("images")
            parts[i] = "labels"
            txt = Path(*parts).with_suffix(".txt")
            if not txt.exists():
                continue
            labels = read_yolo(txt, cmap)
            if labels:
                items.append((src, img, labels))
    return items


def source_class_names(raw):
    """Print each CSV/YOLO source dataset's own classes so the mappings above can be checked."""
    for src in CSV_MAP:
        names = Counter()
        for csv_path in (raw / src).rglob("_annotations.csv"):
            with open(csv_path, newline="", encoding="utf-8") as f:
                names.update(row["class"].strip() for row in csv.DictReader(f))
        print(f"[{src}] class names in _annotations.csv (objects): {dict(sorted(names.items()))}")
    for src in YOLO_MAP:
        yamls = list((raw / src).rglob("*.yaml"))
        for y in yamls:
            print(f"[{src}] {y.relative_to(raw)}:")
            print("   ", y.read_text().strip().replace("\n", "\n    ")[:600])
        if not yamls:  # e.g. ppe-kit ships no yaml; its names are in the Kaggle description
            ids = Counter()
            for txt in (raw / src).rglob("*.txt"):
                if "labels" in txt.parts:
                    ids.update(int(float(l.split()[0])) for l in txt.read_text().splitlines() if l.strip())
            print(f"[{src}] no yaml with class names; source ids (objects): {dict(sorted(ids.items()))}")


def stratified_split(items, seed, fractions=(0.7, 0.2, 0.1)):
    """Group images by the rarest class they contain, then split each group 70/20/10,
    so rare classes appear in train, validation and test."""
    freq = Counter(c for _, _, labels in items for c in {l[0] for l in labels})
    groups = defaultdict(list)
    for it in items:
        rarest = min({l[0] for l in it[2]}, key=lambda c: freq[c])
        groups[rarest].append(it)
    rng = random.Random(seed)
    splits = {"train": [], "valid": [], "test": []}
    for c in sorted(groups):
        g = groups[c]
        rng.shuffle(g)
        n_tr = round(len(g) * fractions[0])
        n_va = round(len(g) * fractions[1])
        splits["train"] += g[:n_tr]
        splits["valid"] += g[n_tr:n_tr + n_va]
        splits["test"] += g[n_tr + n_va:]
    return splits


def repeat_factor_list(train_items, out_dir, threshold, seed):
    """Repeat-factor sampling (Gupta et al., LVIS 2019): images holding rare classes are
    listed more than once in the training file, so the model sees them more often."""
    n = len(train_items)
    img_freq = Counter(c for _, _, labels in train_items for c in {l[0] for l in labels})
    r_class = {c: max(1.0, math.sqrt(threshold / (img_freq[c] / n))) for c in img_freq}
    rng = random.Random(seed)
    lines, factors = [], []
    for _, img_name, labels in train_items:
        r = max(r_class[l[0]] for l in labels)
        k = int(r) + (1 if rng.random() < r - int(r) else 0)
        factors.append(k)
        lines += [str((out_dir / "images" / "train" / img_name).resolve())] * k
    rng.shuffle(lines)
    (out_dir / "train_balanced.txt").write_text("\n".join(lines) + "\n")
    return {CLASSES[c]: round(r, 2) for c, r in sorted(r_class.items())}, len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rfs-threshold", type=float, default=0.15)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)

    source_class_names(raw)
    items = collect(raw)
    if not items:
        raise SystemExit(f"No labelled images found under {raw}")
    # Drop exact duplicate images (same bytes), which some Kaggle zips contain twice
    seen, unique = set(), []
    for it in items:
        h = hashlib.md5(it[1].read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(it)
    print(f"Removed {len(items) - len(unique)} duplicate images")
    items = unique

    if out.exists():
        shutil.rmtree(out)
    named, used = [], set()
    for src, img, labels in items:
        stem = img.stem
        if len(stem) > 80:  # some Roboflow names exceed 200 chars, too long for Windows paths
            stem = f"{stem[:60]}_{hashlib.md5(stem.encode()).hexdigest()[:8]}"
        stem = f"{src}_{stem}"  # prefix avoids clashes between datasets
        k = 1
        while stem.lower() in used:  # same stem in two folders of one source would overwrite labels
            k += 1
            stem = f"{src}_{img.stem[:60]}_{k}"
        used.add(stem.lower())
        named.append((src, img, f"{stem}{img.suffix.lower()}", labels))
    splits = stratified_split([(src, (img, name), labels) for src, img, name, labels in named], a.seed)

    counts = {s: Counter() for s in splits}
    images = {s: 0 for s in splits}
    per_source = Counter()
    train_items = []
    for split, its in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for src, (img, name), labels in its:
            shutil.copy(img, out / "images" / split / name)
            (out / "labels" / split / (Path(name).stem + ".txt")).write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels) + "\n")
            images[split] += 1
            per_source[src] += 1
            counts[split].update(l[0] for l in labels)
            if split == "train":
                train_items.append((src, name, labels))

    rfs, n_balanced = repeat_factor_list(train_items, out, a.rfs_threshold, a.seed)

    common = f"path: {out.resolve()}\nval: images/valid\ntest: images/test\nnames:\n" + "".join(
        f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    (out / "ppe.yaml").write_text("train: images/train\n" + common)
    (out / "ppe_balanced.yaml").write_text("train: train_balanced.txt\n" + common)

    stats = {
        "images": images,
        "images_total": sum(images.values()),
        "objects_total": sum(sum(c.values()) for c in counts.values()),
        "images_per_source": dict(per_source),
        "objects_per_class": {s: {CLASSES[c]: counts[s][c] for c in range(len(CLASSES))} for s in counts},
        "repeat_factor_per_class": rfs,
        "balanced_train_list_length": n_balanced,
        "seed": a.seed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nImages: {images} (total {stats['images_total']}), objects: {stats['objects_total']}")
    print("Images per source:", dict(per_source))
    print(f"{'class':12s}" + "".join(f"{s:>8s}" for s in counts))
    for c in range(len(CLASSES)):
        print(f"{CLASSES[c]:12s}" + "".join(f"{counts[s][c]:8d}" for s in counts))
    print("Repeat factors:", rfs, f"-> balanced train list has {n_balanced} entries")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        w = 0.27
        for k, s in enumerate(counts):
            ax.bar([i + (k - 1) * w for i in range(len(CLASSES))], [counts[s][c] for c in range(len(CLASSES))], w, label=s)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("objects (log scale)")
        ax.set_title("Objects per class and split")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "class_distribution.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
