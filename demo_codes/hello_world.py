# hello_world.py
import torch
import os

# Print basic hello world
print("Hello World from NYU Greene!")

# Print some system information
print(f"Running on node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'unknown')}")

# Check if CUDA is available (if you requested a GPU)
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Running on CPU only.")

