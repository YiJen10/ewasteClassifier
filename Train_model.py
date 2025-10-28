from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    # 1. Load the YOLOv8 Nano model (recommended for speed and efficiency)
    model = YOLO("yolov8n.pt") 

    # 2. Define and run the training parameters
    results = model.train(
        data="datasets/ewaste_v3/data.yaml",        # <--- CONFIRM THIS IS YOUR NEW YAML FILE NAME
        epochs=150,                   # <--- Increased Epochs for Deeper Learning (from initial 50)
        imgsz=640,                    # <--- Use 640 or match your Roboflow resize (e.g., 704)
        
        # --- Critical GPU/Speed Parameters ---
        batch=-1,                     # <--- AUTO BATCH SIZE (Maximizes VRAM usage on your 3050)
        device=0,                     # <--- USE GPU (0 is the device index for your RTX 3050)
        patience=50,                  # <--- Stops training if no mAP improvement after 50 epochs
        
        name="ewaste-3class-v3",       # <--- Unique name for this run
    )

if __name__ == "__main__":
    mp.freeze_support()                 # required on Windows if subprocesses spawn
    mp.set_start_method("spawn", force=True)  # extra safety on some setups
    main()