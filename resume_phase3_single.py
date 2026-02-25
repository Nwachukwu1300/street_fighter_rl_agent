"""
Resume Phase 3 Training - Single Environment (More Stable)
Uses 1 environment instead of 4 parallel to avoid multiprocessing issues

Run with: caffeinate -d diambra run python resume_phase3_single.py
"""
print("=" * 70)
print("RESUMING PHASE 3 TRAINING (Single Environment)")
print("=" * 70)
print("\nStarting...")

import diambra.arena
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import glob
import os

print("✓ Imports successful")


def find_latest_checkpoint(checkpoint_dir="./models_phase3/"):
    """Find the most recent checkpoint"""
    checkpoints = glob.glob(f"{checkpoint_dir}*.zip")
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]


def extract_step_count(checkpoint_path):
    """Extract step count from checkpoint filename"""
    import re
    match = re.search(r'(\d+)_steps', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


def linear_schedule(initial_value, final_value):
    """Learning rate that decays linearly over training"""
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func


def main():
    checkpoint_dir = "./models_phase3/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Find latest checkpoint
    print("\n[1/5] Finding latest checkpoint...")
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if not latest_checkpoint:
        print("❌ No checkpoints found in ./models_phase3/")
        print("Run train_phase3.py instead")
        return

    checkpoint_name = os.path.basename(latest_checkpoint)
    steps_completed = extract_step_count(latest_checkpoint)
    steps_remaining = 5_000_000 - (steps_completed - 2_878_200)

    print(f"✓ Found checkpoint: {checkpoint_name}")
    print(f"  Total steps completed: {steps_completed:,}")
    print(f"  Phase 3 steps completed: {steps_completed - 2_878_200:,} / 5,000,000")
    print(f"  Phase 3 steps remaining: {steps_remaining:,}")
    print(f"  Progress: {((steps_completed - 2_878_200) / 5_000_000 * 100):.1f}%")

    # Environment settings - DIFFICULTY 3
    print("\n[2/5] Configuring environment...")
    env_settings = diambra.arena.EnvironmentSettings()
    env_settings.difficulty = 3  # HARD difficulty
    env_settings.characters = ("Ken", "Ryu", "Chun-Li")
    env_settings.frame_shape = (84, 84, 1)
    env_settings.step_ratio = 6

    # Wrapper settings
    wrappers_settings = diambra.arena.WrappersSettings()
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.flatten = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.filter_keys = ["stage", "timer"]

    print("✓ Settings configured")
    print("\n[3/5] Creating single environment...")
    print("(Using 1 environment for stability - will be slower but more reliable)")

    # Create single environment (no multiprocessing)
    env = diambra.arena.make(
        "sfiii3n",
        env_settings,
        wrappers_settings
    )

    print("✓ Environment created")

    # Load checkpoint
    print(f"\n[4/5] Loading checkpoint...")
    print(f"Loading from: {checkpoint_name}")

    model = PPO.load(
        latest_checkpoint.replace('.zip', ''),
        env=env,
        device="auto"
    )
    print("✓ Checkpoint loaded successfully")

    # Update learning rate schedule
    print("\n[5/5] Updating hyperparameters...")
    model.learning_rate = linear_schedule(2e-4, 5e-6)
    print("✓ Learning rate schedule updated")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,  # Every 200k steps (no division by num_envs)
        save_path=checkpoint_dir,
        name_prefix="sfiii_phase3"
    )

    # Train
    print("\nResuming Phase 3 training...")
    print("=" * 70)
    print(f"RESUMING: {steps_remaining:,} timesteps remaining")
    print("Difficulty: 3 (Hard)")
    print(f"Resuming from: {checkpoint_name}")
    estimated_hours = (steps_remaining / 5_000_000) * 5.5 * 4  # 4x slower with 1 env
    print(f"Expected time: {estimated_hours:.1f} hours (single environment)")
    print("=" * 70)
    print()

    try:
        model.learn(
            total_timesteps=steps_remaining,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=5,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            print("(Environment already closed)")

    # Save final model
    final_path = "sfiii_agent_phase3_final"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("PHASE 3 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✓ Model saved: {final_path}.zip")
    print(f"  Location: {os.path.abspath(final_path + '.zip')}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print()
    print("Total training steps across all phases:")
    print(f"  Phase 1: 2,000,000 steps (difficulty 1)")
    print(f"  Phase 2: ~2,880,000 steps (difficulty 2)")
    print(f"  Phase 3: 5,000,000 steps (difficulty 3)")
    print(f"  TOTAL: ~9,880,000 steps")
    print("=" * 70)


if __name__ == "__main__":
    main()