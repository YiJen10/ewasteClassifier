from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    # 1. Load the YOLOv8 Nano model 
    model = YOLO("runs/ewaste-3class-v9/weights/best.pt") 

    # 2. Define and run the training parameters
    results = model.train(
        data="datasets/ewaste_v9/data.yaml",        
        epochs=75,                    # <--- Increased Epochs for Deeper Learning (from initial 50)
        lr0=0.001,                    # <--- CRITICAL: Set a low learning rate
        imgsz=640,                    # <--- Use 640 to match Roboflow resize
        
        # --- Critical GPU/Speed Parameters ---
        batch=-1,                     # <--- AUTO BATCH SIZE (Maximizes VRAM usage on 3050)
        device=0,                     # <--- USE GPU (0 is the device index for laptop's RTX 3050)
        patience=50,                  # <--- Stops training if no mAP improvement after 50 epochs
        
        name="ewaste-3class-v10",       
    )

if __name__ == "__main__":
    mp.freeze_support()                 # required on Windows if subprocesses spawn
    mp.set_start_method("spawn", force=True)  # extra safety on some setups
    main()