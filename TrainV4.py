from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    # Load your *existing* best model, not the base yolov8n.pt
    model = YOLO("runs/ewaste-3class-v4-with-negatives/weights/best.pt") 

    # Train the model
    results = model.train(
        data="datasets/ewaste_v4/data.yaml",  
        epochs=50,                      # Train for 50 new epochs
        imgsz=640,
        name="runs/ewaste-3class-v5" # Give it a new name
    )

if __name__ == "__main__":
    mp.freeze_support()                 # required on Windows if subprocesses spawn
    mp.set_start_method("spawn", force=True)  # extra safety on some setups
    main()