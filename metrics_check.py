from ultralytics import YOLO
import torch

# --- Configuration ---
# NOTE: The model path MUST be updated after your training completes.
MODEL_PATH = "runs/ewaste-3class-v3/weights/best.pt" 

# Set the path to your Roboflow YAML file for validation
# This file contains the path to your 'test' folder
DATA_YAML_PATH = "datasets/ewaste_v3/data.yaml" 

# The names of your classes (based on common YOLO convention)
# Adjust these if your Roboflow names are different (e.g., '0': 'Battery', '1': 'Cable', '2': 'PCB')
CLASS_NAMES = ['Battery', 'Cable', 'PCB'] 

# --- Metric Check Logic ---
def check_final_metrics():
    # 1. Load the trained model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the path to the best.pt file is correct after your training run.")
        return

    # 2. Run the validation mode on the entire test set
    print("\nStarting evaluation on the Test Set (This may take a few minutes on the GPU)...")
    
    # Running validation calculates P, R, F1, and mAP for overall and per-class performance.
    metrics = model.val(
        data=DATA_YAML_PATH,  # Uses the test set defined in the YAML file
        imgsz=640,            # Match training resolution
        split='test',         # Explicitly run on the Test split
        device=0,             # Use GPU
        save_json=False,      # Set to True if you want a detailed JSON output file
        conf=0.001,           # Low confidence to calculate the full PR curve
        iou=0.60              # Standard IOU for COCO-style metrics
    )

# 3. Print Overall Results (Project Targets)
    print("\n" + "="*70)
    print("           ✨ V3 MODEL OVERALL PERFORMANCE METRICS ✨")
    print("="*70)
    print(f"Target Precision: >= 0.85 | Target Recall: >= 0.80 | Target mAP@0.5: >= 0.85")
    print("-" * 70)
    
    # We use the printed 'all' row values for the summary
    print(f"OVERALL Precision (P): {metrics.box.mp:.4f}")
    print(f"OVERALL Recall (R):    {metrics.box.mr:.4f}")
    print(f"OVERALL mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"OVERALL mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print("="*70)

    # 4. Print Per-Class Results (FIXED)
    print("\n" + "="*70)
    print("        📊 PER-CLASS mAP@0.5 BREAKDOWN (Crucial for Balance) 📊")
    print("="*70)

    # FIX: Use metrics.box.ap50 which returns the AP@0.5 for each class (a NumPy array)
    map50_per_class = metrics.box.ap50 
    
    # Class order is inferred from the original YAML/training run
    # If your YAML defines classes in a different order, adjust CLASS_NAMES list
    CLASS_NAMES = ['Battery', 'Cable', 'PCB'] 
    
    # Print header
    print(f"{'Class':<10}{'mAP@0.5':>10}")
    print("-" * 20)
    
    # Print results for each class
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(map50_per_class):
            # Check if the value is a scalar before formatting
            score = map50_per_class[i] if isinstance(map50_per_class[i], (float, int)) else map50_per_class[i].item()
            print(f"{class_name:<10}{score:>10.4f}")
        
    print("="*70)


if __name__ == '__main__':
    # Ensure you are running this within your active VENV and CUDA is available
    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA not available. Please ensure your VENV is active.")
    else:
        check_final_metrics()