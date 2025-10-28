from ultralytics import YOLO
import torch
import os

# --- Configuration ---
# NOTE: Update this path after your training run 'ewaste-v3-final' completes.
# This assumes the new model is saved in the 'runs/detect' folder.
MODEL_PATH = "runs/ewaste-3class-v3/weights/best.pt" 

# Set the path to the test image you want to check
# Example: Using the previously problematic sample1.jpg
IMAGE_FILE = "test_image/sample10.jpg" 

# Optimal Post-Processing Settings for v3
CONF_THRESHOLD = 0.40  # Increased to prune false positives (Targeting >= 85% Precision)
IOU_THRESHOLD = 0.45   # Decreased for stricter NMS to handle clutter/overlap (Fixing misclassification)
IMG_SIZE = 640         # Match your training resolution

# --- Detection Logic ---
def run_image_detection():
    # Check for model existence before proceeding
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure you have run python Train_model.py and that the path is correct.")
        return
        
    # Check for test image existence
    if not os.path.exists(IMAGE_FILE):
        print(f"Error: Test image file not found at {IMAGE_FILE}")
        return

    try:
        # Load the model
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully.")
        print(f"Running detection on: {IMAGE_FILE}")
        
        # Run prediction using optimized parameters
        results = model.predict(
            source=IMAGE_FILE, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            show=True,              # Display the resulting image with boxes
            save=True,              # Save results to runs/predict/
            project="runs/predict", 
            name="v3_image_check",
            device=0,               # Use GPU
            verbose=False
        )

        # Print detection summary, similar to your original code
        res = results[0]
        print("-" * 50)
        print(f"Detected Objects ({len(res.boxes)} total):")
        
        for b in res.boxes:
            class_id = int(b.cls[0])
            confidence = float(b.conf[0])
            
            # Print class name and confidence score
            print(f"  {model.names[class_id]}: {confidence:.4f}")
        
        print("-" * 50)
        print(f"Results saved to runs/predict/v3_image_check/")

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")

if __name__ == '__main__':
    # Ensure you are running this within your active VENV
    if not os.environ.get('VIRTUAL_ENV'):
        print("WARNING: Not running in a virtual environment. Ensure CUDA PyTorch is accessible globally.")
    
    run_image_detection()