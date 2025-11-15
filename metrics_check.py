from ultralytics import YOLO
import torch

# --- Configuration ---
MODEL_PATH = "runs/ewaste-3class-v10/weights/best.pt" 
DATA_YAML_PATH = "datasets/ewaste_v9/data.yaml" 
# ADDED: Define your output file name
OUTPUT_FILE = "evaluation_report.txt" 

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
    print("\nStarting evaluation on the Test Set...")
    
    metrics = model.val(
        data=DATA_YAML_PATH, 
        imgsz=640,          
        split='test',       
        device=0,           
        save_json=False,    
        conf=0.001,         
        iou=0.60            
    )

    # 3. Open the output file to start writing
    print(f"\nWriting results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    
        # 4. Print & Write Overall Results (Project Targets)
        s1 = "\n" + "="*70
        s2 = "               ✨ V7 MODEL OVERALL PERFORMANCE METRICS ✨"
        s3 = "="*70
        s4 = "Target Precision: >= 0.85 | Target Recall: >= 0.80 | Target mAP@0.5: >= 0.85"
        s5 = "-" * 70
        
        print(s1); f.write(s1 + "\n")
        print(s2); f.write(s2 + "\n")
        print(s3); f.write(s3 + "\n")
        print(s4); f.write(s4 + "\n")
        print(s5); f.write(s5 + "\n")
        
        # We use the printed 'all' row values for the summary
        line_p = f"OVERALL Precision (P): {metrics.box.mp:.4f}"
        line_r = f"OVERALL Recall (R):    {metrics.box.mr:.4f}"
        line_m50 = f"OVERALL mAP@0.5:       {metrics.box.map50:.4f}"
        line_map = f"OVERALL mAP@0.5:0.95:  {metrics.box.map:.4f}"
        s6 = "="*70
        
        print(line_p); f.write(line_p + "\n")
        print(line_r); f.write(line_r + "\n")
        print(line_m50); f.write(line_m50 + "\n")
        print(line_map); f.write(line_map + "\n")
        print(s6); f.write(s6 + "\n")

        # 5. Print & Write Per-Class Detailed Results
        s7 = "\n" + "="*70
        s8 = "                   📊 DETAILED PER-CLASS METRICS 📊"
        s9 = "="*70
        
        print(s7); f.write(s7 + "\n")
        print(s8); f.write(s8 + "\n")
        print(s9); f.write(s9 + "\n")

        # Get the map of class names {0: 'Battery', 1: 'Cable', 2: 'PCB'}
        class_names_map = metrics.names

        # Get the per-class metric arrays
        per_class_precision = metrics.box.p
        per_class_recall = metrics.box.r
        per_class_map50 = metrics.box.ap50
        per_class_map_full = metrics.box.ap # This is mAP@0.5:0.95

        # Print header
        table_head = f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'mAP@0.5':>10}{'mAP@.5-.95':>12}"
        table_sep = "-" * 57
        
        print(table_head); f.write(table_head + "\n")
        print(table_sep); f.write(table_sep + "\n")
        
        # Loop through each class and print its metrics
        for i, class_name in class_names_map.items():
            # Get the metrics for class 'i'
            p = per_class_precision[i]
            r = per_class_recall[i]
            ap50 = per_class_map50[i]
            ap = per_class_map_full[i]
            
            # Format the row
            row = f"{class_name:<15}{p:>10.4f}{r:>10.4f}{ap50:>10.4f}{ap:>12.4f}"
            
            # Print and write the row
            print(row); f.write(row + "\n")
            
        s10 = "="*70
        print(s10); f.write(s10 + "\n")

    print(f"\n✅ Successfully saved evaluation results to {OUTPUT_FILE}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA not available. Please ensure your VENV is active.")
    else:
        check_final_metrics()