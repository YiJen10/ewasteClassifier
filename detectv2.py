# predict_once.py
from ultralytics import YOLO
import glob, os, cv2

# 1) auto-pick newest best.pt under runs/**/weights/
cands = glob.glob("runs/**/weights/best.pt", recursive=True)
assert cands, "No best.pt found. Train first."
weights = max(cands, key=os.path.getmtime)
print("Using weights:", weights)

# 2) run prediction with a more forgiving threshold
model = YOLO(weights)
res = model.predict(
    source="test_image/sample10.jpg",
    conf=0.3,          # lower -> more detections
    iou=0.50,
    imgsz=640,
    max_det=300,
    save=True,          # will save to runs/detect/predict*/
    project="runs/predict",
    name="ewaste-v2",
    exist_ok=True,
    verbose=True
)[0]

# 3) print what the model actually found
print("Detections:", len(res.boxes))
for b in res.boxes:
    cls = int(b.cls[0])
    print(model.names[cls], float(b.conf[0]), b.xyxy[0].tolist())

# 4) explicitly draw & save annotated image (in case GUI show() is flaky)
img = res.plot()  # returns numpy array with boxes+labels drawn
cv2.imwrite("runs/predict/ewaste-v2/sample10_annotated.png", img)
print("Saved:", "runs/predict/ewaste-v2/sample10_annotated.png")
