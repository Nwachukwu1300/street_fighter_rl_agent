#!/usr/bin/env python3
"""
Diagnostic script to check DIAMBRA and Stable Baselines3 installation
"""
import sys

print("Python version:", sys.version)
print("=" * 60)

# Check imports
print("\nChecking imports...")
try:
    import diambra.arena
    print("✓ diambra.arena imported successfully")
    print(f"  Version: {diambra.arena.__version__ if hasattr(diambra.arena, '__version__') else 'unknown'}")
except Exception as e:
    print(f"✗ Failed to import diambra.arena: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    print("✓ stable_baselines3 imported successfully")
    import stable_baselines3
    print(f"  Version: {stable_baselines3.__version__}")
except Exception as e:
    print(f"✗ Failed to import stable_baselines3: {e}")
    print("\nYou need to install stable-baselines3:")
    print("  pip install stable-baselines3")
    sys.exit(1)

try:
    from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
    print("✓ make_sb3_env imported successfully")
except Exception as e:
    print(f"✗ Failed to import make_sb3_env: {e}")
    print("\nYou may need to install DIAMBRA with SB3 support:")
    print("  pip install diambra-arena[stable-baselines3]")
    sys.exit(1)

print("\n" + "=" * 60)
print("All checks passed! Your installation looks good.")
print("=" * 60)
