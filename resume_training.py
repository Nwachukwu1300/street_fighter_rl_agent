"""
Resume Training from Last Checkpoint
Continues curriculum training from where it left off

Run with: caffeinate -d diambra run -s=4 python resume_training.py
"""
import diambra.arena
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os
import glob
from custom_rewards import HealthBasedRewardWrapper


def make_env(rank=0, difficulty=3, step_ratio=6):
    """Create environment with optimal settings"""
    def _init():
        env_settings = diambra.arena.EnvironmentSettings()
        env_settings.difficulty = difficulty
        env_settings.characters = ("Ken", "Ryu", "Chun-Li")
        env_settings.frame_shape = (84, 84, 1)
        env_settings.rank = rank
        env_settings.step_ratio = step_ratio

        wrappers_settings = diambra.arena.WrappersSettings()
        wrappers_settings.no_attack_buttons_combinations = True
        wrappers_settings.stack_frames = 4
        wrappers_settings.flatten = True
        wrappers_settings.filter_keys = [
            "stage", "timer",
            "P1_health", "P2_health",
            "P1_super_bar", "P2_super_bar",
            "P1_side", "P2_side",
            "P1_character", "P2_character"
        ]
        wrappers_settings.normalize_reward = False

        env = diambra.arena.make("sfiii3n", env_settings, wrappers_settings)
        env = HealthBasedRewardWrapper(env)
        return env

    return _init


def find_latest_checkpoint(checkpoint_dir="./models_ultimate/"):
    """Find the most recent checkpoint"""
    checkpoints = glob.glob(f"{checkpoint_dir}*.zip")
    if not checkpoints:
        return None
    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]


def train_phase_continue(phase_name, difficulty, total_steps, checkpoint_path,
                         num_envs=4, step_ratio=6, checkpoint_dir="./models_ultimate/"):
    """Continue training from checkpoint"""

    print(f"\n{'=' * 70}")
    print(f"RESUMING: {phase_name}")
    print(f"  From checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Timesteps: {total_steps:,}")
    print(f"{'=' * 70}\n")

    # Create environments
    env = DummyVecEnv([make_env(i, difficulty, step_ratio) for i in range(num_envs)])
    env = VecMonitor(env)

    # Load model from checkpoint
    print("Loading model from checkpoint...")
    model = PPO.load(checkpoint_path, env=env)
    print("✓ Model loaded successfully")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_envs,
        save_path=checkpoint_dir,
        name_prefix=f"sfiii_{phase_name.lower()}"
    )

    # Train
    print("Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_steps,
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

    return model


def main():
    print("=" * 70)
    print("RESUME TRAINING - Continue from Last Checkpoint")
    print("=" * 70)

    checkpoint_dir = "./models_ultimate/"

    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is None:
        print("\n✗ No checkpoints found!")
        print("Please run train_ultimate.py first")
        return

    print(f"\n✓ Found checkpoint: {os.path.basename(latest_checkpoint)}")

    # Determine which phase to resume
    if "phase1" in latest_checkpoint.lower():
        print("\n✓ Phase 1 completed!")
        print("Starting Phase 2: Medium difficulty (3M steps)")

        # PHASE 2
        model = train_phase_continue(
            phase_name="Phase2_Medium",
            difficulty=2,
            total_steps=3_000_000,
            checkpoint_path=latest_checkpoint,
            num_envs=4,
            step_ratio=6,
            checkpoint_dir=checkpoint_dir
        )

        # Save after Phase 2
        phase2_path = "sfiii_agent_phase2_complete"
        model.save(phase2_path)
        print(f"\n✓ Phase 2 complete! Saved as: {phase2_path}.zip")

        # PHASE 3
        print("\n" + "=" * 70)
        print("Starting Phase 3: Hard difficulty (5M steps)")
        print("=" * 70)

        model = train_phase_continue(
            phase_name="Phase3_Hard",
            difficulty=3,
            total_steps=5_000_000,
            checkpoint_path=phase2_path + ".zip",
            num_envs=4,
            step_ratio=6,
            checkpoint_dir=checkpoint_dir
        )

        final_path = "sfiii_agent_ultimate_curriculum"

    elif "phase2" in latest_checkpoint.lower():
        print("\n✓ Phase 2 completed!")
        print("Starting Phase 3: Hard difficulty (5M steps)")

        # PHASE 3 only
        model = train_phase_continue(
            phase_name="Phase3_Hard",
            difficulty=3,
            total_steps=5_000_000,
            checkpoint_path=latest_checkpoint,
            num_envs=4,
            step_ratio=6,
            checkpoint_dir=checkpoint_dir
        )

        final_path = "sfiii_agent_ultimate_curriculum"

    else:
        print("\n⚠ Unrecognized checkpoint phase")
        print("Continuing training at difficulty 3...")

        model = train_phase_continue(
            phase_name="Resume",
            difficulty=3,
            total_steps=5_000_000,
            checkpoint_path=latest_checkpoint,
            num_envs=4,
            step_ratio=6,
            checkpoint_dir=checkpoint_dir
        )

        final_path = "sfiii_agent_resumed"

    # Save final model
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    model.save(final_path)
    print(f"✓ Final model saved: {final_path}.zip")
    print(f"  Location: {os.path.abspath(final_path + '.zip')}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
