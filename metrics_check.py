from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    model = YOLO("runs/ewaste-3class/weights/best.pt")  # <-- adjust if different
    metrics = model.val(
        data="datasets/ewaste/data.yaml",
        split="test",
        imgsz=640,
        conf=0.001,      # build full PR curves
        iou=0.5,
        device=0,        # use GPU
        workers=0,       # <<< IMPORTANT on Windows; try 0 or 2
        batch=8,         # small, safe batch for 4GB VRAM
        plots=True,
        save_json=True
    )

    print(f"Precision:    {metrics.box.mp:.3f}")
    print(f"Recall:       {metrics.box.mr:.3f}")
    print(f"mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    for i, name in enumerate(metrics.names):
        print(f"AP@0.5 [{name}]: {metrics.box.maps[i]:.3f}")

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()