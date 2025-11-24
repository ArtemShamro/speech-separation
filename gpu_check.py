import torch
import subprocess

print("=== PyTorch CUDA Check ===")
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cudnn.enabled:", torch.backends.cudnn.enabled)

num_gpus = torch.cuda.device_count()
print("GPU count:", num_gpus)

if num_gpus > 0:
    for i in range(num_gpus):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))

    # Test tensor allocation
    try:
        x = torch.randn(1000, 1000).to('cuda')
        print("Tensor successfully moved to GPU.")
    except Exception as e:
        print("Error moving tensor to GPU:", e)

print("\n=== nvidia-smi ===")
try:
    print(subprocess.check_output("nvidia-smi", shell=True).decode())
except:
    print("nvidia-smi not found (driver issue or no NVIDIA GPU).")
