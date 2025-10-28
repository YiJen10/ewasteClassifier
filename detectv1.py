from ultralytics import YOLO

# Load trained model
model = YOLO("runs/ewaste-3class/weights/best.pt")

# Run detection on an image
# results = model("test_images/sample4.jpg", imgsz=640, save=True)

# Lower confidence to 0.25 to show weak detections
results = model.predict(source="test_image/sample9.jpg", conf=0.5, save=True)

# Optionally print results
results[0].show()  # Show image with boxes (for local GUI environments)