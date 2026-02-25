"""
Resume Phase 3 Training from Last Checkpoint
Automatically finds the latest checkpoint and continues training

Run with: caffeinate -d diambra run -s=4 python resume_phase3.py
"""
print("=" * 70)
print("RESUMING PHASE 3 TRAINING")
print("=" * 70)
print("\nStarting...")

import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
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
    # Example: sfiii_phase3_4078200_steps.zip -> 4078200
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
    steps_remaining = 5_000_000 - (steps_completed - 2_878_200)  # Subtract Phase 2 steps

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

    # Wrapper settings (same as training)
    wrappers_settings = diambra.arena.WrappersSettings()
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.flatten = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.filter_keys = ["stage", "timer"]

    print("✓ Settings configured")
    print("\nCreating parallel environments (4 workers)...")
    print("(This may take 1-2 minutes)")

    # Create environments - use DummyVecEnv (more stable on macOS Python 3.12)
    env, num_envs = make_sb3_env(
        "sfiii3n",
        env_settings=env_settings,
        wrappers_settings=wrappers_settings,
        use_subprocess=False,  # DummyVecEnv instead of SubprocVecEnv
        seed=42
    )

    print(f"✓ Created {num_envs} parallel environments (DummyVecEnv)")

    # Load checkpoint
    print(f"\n[3/5] Loading checkpoint...")
    print(f"Loading from: {checkpoint_name}")

    model = PPO.load(
        latest_checkpoint.replace('.zip', ''),
        env=env,
        device="auto"
    )
    print("✓ Checkpoint loaded successfully")

    # Update learning rate schedule
    print("\n[4/5] Updating hyperparameters...")
    model.learning_rate = linear_schedule(2e-4, 5e-6)
    print("✓ Learning rate schedule updated")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000 // num_envs,
        save_path=checkpoint_dir,
        name_prefix="sfiii_phase3"
    )

    # Train
    print("\n[5/5] Resuming Phase 3 training...")
    print("=" * 70)
    print(f"RESUMING: {steps_remaining:,} timesteps remaining")
    print("Difficulty: 3 (Hard)")
    print(f"Resuming from: {checkpoint_name}")
    estimated_hours = (steps_remaining / 5_000_000) * 5.5
    print(f"Expected time: {estimated_hours:.1f} hours")
    print("=" * 70)
    print()

    try:
        model.learn(
            total_timesteps=steps_remaining,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=5,
            reset_num_timesteps=False  # Continue counting
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
    print()
    print("Expected performance:")
    print("  vs Difficulty 1: 90-95% win rate")
    print("  vs Difficulty 2: 80-85% win rate")
    print("  vs Difficulty 3: 60-65% win rate")
    print("=" * 70)


if __name__ == "__main__":
    main()