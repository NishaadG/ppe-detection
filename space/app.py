"""Gradio demo: detect PPE (hard hats, vests, goggles, masks) and flag missing gear."""
from collections import Counter
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

model = YOLO(Path(__file__).with_name("best.pt"))  # works from any working directory

GEAR = ["hardhat", "vest", "goggles", "mask"]
MISSING = {"no-hardhat": "hard hat", "no-vest": "safety vest", "no-goggles": "goggles", "no-mask": "mask"}


def detect(image, conf):
    if image is None:
        return None, "Upload an image first."
    r = model.predict(image, conf=conf, imgsz=640, verbose=False)[0]
    annotated = r.plot()[:, :, ::-1]  # BGR -> RGB
    counts = Counter(r.names[int(c)] for c in r.boxes.cls)
    worn = [f"{counts[g]} x {g}" for g in GEAR if counts[g]]
    missing = [f"{counts[k]} x missing {v}" for k, v in MISSING.items() if counts[k]]
    if missing:
        status = "### Violation found\n" + "\n".join(f"- {m}" for m in missing)
    elif worn:
        status = "### No violations found"
    else:
        status = "### No PPE or people detected"
    if worn:
        status += "\n\n**Gear detected:** " + ", ".join(worn)
    status += f"\n\n_{len(r.boxes)} objects, {r.speed['inference']:.0f} ms inference_"
    return annotated, status


demo = gr.Interface(
    fn=detect,
    inputs=[gr.Image(type="pil", label="Site photo"),
            gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence threshold")],
    outputs=[gr.Image(label="Detections"), gr.Markdown(label="Compliance")],
    title="PPE Detection",
    description="YOLOv8 model trained on 4 merged Kaggle datasets (8 classes: hard hat, vest, goggles, mask, and each one missing). "
                "Upload a construction-site photo to see detected gear and any missing items.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
