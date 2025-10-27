from ultralytics import YOLO

model = YOLO("runs/ewaste-3class/weights/best.pt")
metrics = model.val(
    data="datasets/ewaste/data.yaml",
    imgsz=640,
    split="test",
    conf=0.001,  # to build full PR curves
    iou=0.5,
    save_json=True,
    plots=True
)

print(f"Precision (mean): {metrics.box.mp:.3f}")
print(f"Recall (mean):    {metrics.box.mr:.3f}")
print(f"mAP@0.5:          {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95:     {metrics.box.map:.3f}")

for i, name in enumerate(metrics.names):
    ap50 = metrics.box.maps[i]
    print(f"AP@0.5 [{name}]:  {ap50:.3f}")
