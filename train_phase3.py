"""
Train Phase 3 - Hard Difficulty
Loads Phase 2 model and trains on difficulty 3

Run with: caffeinate -d diambra run -s=4 python train_phase3.py
"""
print("=" * 70)
print("PHASE 3 TRAINING - Hard Difficulty")
print("=" * 70)
print("\nStarting...")

import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os

print("✓ Imports successful")


def linear_schedule(initial_value, final_value):
    """Learning rate that decays linearly over training"""
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func


def main():
    print("\nPhase 3: Loading Phase 2 model and training on difficulty 3")
    print("Expected final performance: 60-65% win rate vs difficulty 3")
    print("=" * 70)

    checkpoint_dir = "./models_phase3/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Environment settings - DIFFICULTY 3
    print("\n[1/4] Configuring environment...")
    env_settings = diambra.arena.EnvironmentSettings()
    env_settings.difficulty = 3  # HARD difficulty
    env_settings.characters = ("Ken", "Ryu", "Chun-Li")
    env_settings.frame_shape = (84, 84, 1)
    env_settings.step_ratio = 6

    # Wrapper settings (same as Phase 2)
    wrappers_settings = diambra.arena.WrappersSettings()
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.flatten = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.filter_keys = ["stage", "timer"]

    print("✓ Settings configured")
    print("\nCreating parallel environments (4 workers)...")
    print("(This may take 1-2 minutes)")

    # Create environments
    env, num_envs = make_sb3_env(
        "sfiii3n",
        env_settings=env_settings,
        wrappers_settings=wrappers_settings,
        use_subprocess=True,
        seed=42
    )

    print(f"✓ Created {num_envs} parallel environments")

    # Load Phase 2 model
    print("\n[2/4] Loading Phase 2 model...")
    phase2_model_path = "sfiii_agent_phase2_default_rewards"
    print(f"Loading from: {phase2_model_path}.zip")

    model = PPO.load(
        phase2_model_path,
        env=env,
        device="auto"
    )
    print("✓ Phase 2 model loaded successfully")

    # Update learning rate schedule for Phase 3
    print("\n[3/4] Updating hyperparameters for Phase 3...")
    model.learning_rate = linear_schedule(2e-4, 5e-6)  # Lower learning rate for fine-tuning
    print("✓ Learning rate schedule updated")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000 // num_envs,  # Save every 200k steps
        save_path=checkpoint_dir,
        name_prefix="sfiii_phase3"
    )

    # Train
    print("\n[4/4] Starting Phase 3 training...")
    print("=" * 70)
    print("TRAINING: 5,000,000 timesteps")
    print("Difficulty: 3 (Hard)")
    print("Starting from: Phase 2 checkpoint (~2.88M steps)")
    print("Expected time: 5-6 hours")
    print("=" * 70)
    print()

    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=5,
            reset_num_timesteps=False  # Continue counting from Phase 2
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
