# sweep_thresholds.py
from ultralytics import YOLO
import numpy as np

model = YOLO("runs/ewaste-3class/weights/best.pt")
best = None
for conf in np.linspace(0.35, 0.60, 6):
    for iou in np.linspace(0.45, 0.60, 4):
        m = model.val(data="datasets/ewaste/data.yaml", split="val",
                      conf=float(conf), iou=float(iou), imgsz=640, verbose=False, plots=False)
        P, R, map50 = m.box.mp, m.box.mr, m.box.map50
        if (P>=0.85 and R>=0.80 and map50>=0.85) and (best is None or map50>best["map50"]):
            best = dict(conf=conf, iou=iou, P=float(P), R=float(R), map50=float(map50))
print("Best:", best)
