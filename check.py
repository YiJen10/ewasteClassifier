import yaml, glob

# a) what does the model believe?
from ultralytics import YOLO
m = YOLO("runs/ewaste-3class-v2/weights/best.pt")
print("model.names:", m.names)  # expect {0:'battery',1:'cable',2:'pcb'} in that exact order

# b) what does your data.yaml say?
with open("datasets/ewaste_v2/data.yaml","r") as f:
    d = yaml.safe_load(f)
print("data.yaml names:", d.get("names"))

# c) detect any segmentation-style labels slipped in
bad = []
for p in glob.glob("datasets/ewaste_v2/labels/**/*.txt", recursive=True):
    with open(p) as f:
        for line in f:
            nums = line.strip().split()
            if len(nums) > 6:  # seg masks have lots of numbers; boxes have 5
                bad.append(p); break
print("seg-like label files:", len(bad))
