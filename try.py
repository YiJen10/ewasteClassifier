from ultralytics import YOLO
model = YOLO("runs/ewaste-3class-v3/weights/best.pt")

res = model.predict(
    source="test_image/sample4.png",
    conf=0.25,          # lower to boost recall
    iou=0.60,           # slightly higher IoU to avoid over-suppressing nearby boxes
    imgsz=704,          # helps thin/small items (cables, small PCBs)
    max_det=300,        # just in case there are many objects
    agnostic_nms=False, # keep per-class NMS
    save=True, project="runs/predict", name="triage"
)[0]

print("Detected:", len(res.boxes))
for b in res.boxes:
    print(int(b.cls[0]), model.names[int(b.cls[0])], float(b.conf[0]))

res[0].show()