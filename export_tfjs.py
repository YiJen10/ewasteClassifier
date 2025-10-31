from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/ewaste-3class-v6/weights/best.pt")

# Export to TensorFlow.js
model.export(format="tfjs")

print("✅ Export complete! Check your 'runs/ewaste-3class-v6/weights' folder for TFJS model files.")
