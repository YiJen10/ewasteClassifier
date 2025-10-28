import torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch location:", torch.__file__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))