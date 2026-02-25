"""
Test if we can load the checkpoint without creating environments
"""
import glob
import os

print("Finding checkpoint...")
checkpoints = glob.glob("./models_ultimate/*.zip")
if not checkpoints:
    print("No checkpoints found!")
    exit(1)

checkpoints.sort(key=os.path.getmtime)
latest = checkpoints[-1]
print(f"Latest checkpoint: {latest}")

# Try to inspect the checkpoint
from stable_baselines3 import PPO
print("Loading checkpoint data...")

# Load without env (will fail but tells us if file is corrupt)
try:
    import torch
    data = torch.load(latest)
    print(f"✓ Checkpoint file is valid")
    print(f"  Keys: {list(data.keys())}")
except Exception as e:
    print(f"✗ Error loading checkpoint: {e}")
