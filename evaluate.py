"""Evaluate trained runs on the held-out test split and write a comparison table.

Usage:
    python evaluate.py --data data/ppe.yaml --runs runs/baseline runs/balanced --out results
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights, data, out_dir, name):
    model = YOLO(weights)
    r = model.val(data=data, split="test", imgsz=640, batch=16, plots=True,
                  project=str(out_dir), name=f"{name}_test", exist_ok=True, verbose=False)
    names = r.names
    per_class = {names[int(c)]: round(float(ap), 4) for c, ap in zip(r.ap_class_index, r.box.ap50)}
    res = {
        "run": name,
        "weights": str(weights),
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "ap50_per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in r.speed.items()},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2))
    return res


def table(results):
    classes = sorted({c for r in results for c in r["ap50_per_class"]})
    head = "| Metric (test split) | " + " | ".join(r["run"] for r in results) + " |"
    sep = "|---" * (len(results) + 1) + "|"
    rows = [head, sep]
    for key, label in [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("precision", "Precision"), ("recall", "Recall")]:
        rows.append(f"| {label} | " + " | ".join(f"{r[key]:.3f}" for r in results) + " |")
    for c in classes:
        rows.append(f"| AP@0.5 {c} | " + " | ".join(
            f"{r['ap50_per_class'][c]:.3f}" if c in r["ap50_per_class"] else "-" for r in results) + " |")
    rows.append("| Inference ms/image | " + " | ".join(f"{r['speed_ms_per_image'].get('inference', 0):.1f}" for r in results) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders that contain weights/best.pt")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for run in a.runs:
        w = Path(run) / "weights" / "best.pt"
        if not w.exists():
            print(f"skip {run}: no weights/best.pt")
            continue
        results.append(evaluate(w, a.data, out, Path(run).name))
    if not results:
        raise SystemExit("No run folder had weights/best.pt, nothing to evaluate")
    md = table(results)
    (out / "comparison.md").write_text(md + "\n")
    print(md)
    best = max(results, key=lambda r: r["mAP50"])
    (out / "best_run.txt").write_text(best["run"] + "\n")
    print(f"\nBest run on test mAP@0.5: {best['run']} ({best['mAP50']:.3f})")


if __name__ == "__main__":
    main()
