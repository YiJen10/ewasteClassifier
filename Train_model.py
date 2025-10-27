# Train_model.py
from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    model = YOLO("yolov8n.pt")   # ok if it downloads yolo11n under the hood
    model.train(
        data="datasets/ewaste_v2/data.yaml",
        epochs=60,
        imgsz=640,
        batch=-1,
        device=0,          # you have GPU working 🎉
        workers=2,         # ↓ reduce workers on Windows to avoid spawn issues
        cache=True,
        rect=True,
        patience=15,
        optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        degrees=10, translate=0.08, scale=0.4, shear=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=1.0, mixup=0.15, flipud=0.2, fliplr=0.5,
        project="runs", name="ewaste-3class-v2", exist_ok=False
    )

if __name__ == "__main__":
    mp.freeze_support()                 # required on Windows if subprocesses spawn
    mp.set_start_method("spawn", force=True)  # extra safety on some setups
    main()
