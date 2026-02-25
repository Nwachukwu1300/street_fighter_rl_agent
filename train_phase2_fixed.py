"""
Train Phase 2 - Using DIAMBRA's make_sb3_env
Fixed version that doesn't hang on environment creation

Run with: caffeinate -d diambra run -s=4 python train_phase2_fixed.py
"""
print("=" * 70)
print("PHASE 2 TRAINING - Medium Difficulty (Fixed)")
print("=" * 70)
print("\nStarting imports...")

import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os

print("✓ Imports successful")

from custom_rewards import HealthBasedRewardWrapper

print("✓ Custom rewards imported")


def linear_schedule(initial_value, final_value):
    """Learning rate that decays linearly over training"""
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func


def main():
    print("\nThis trains Phase 2 from scratch (difficulty 2)")
    print("Expected performance: 70-75% win rate")
    print("=" * 70)

    checkpoint_dir = "./models_phase2/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Environment settings
    print("\n[1/3] Configuring environments...")
    env_settings = diambra.arena.EnvironmentSettings()
    env_settings.difficulty = 2  # Medium difficulty
    env_settings.characters = ("Ken", "Ryu", "Chun-Li")
    env_settings.frame_shape = (84, 84, 1)
    env_settings.step_ratio = 6

    # Wrapper settings
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

    print("✓ Settings configured")
    print("\nCreating 4 parallel environments...")
    print("(This may take 1-2 minutes - please wait)")

    # Create environments using DIAMBRA's function
    env, num_envs = make_sb3_env(
        "sfiii3n",
        env_settings=env_settings,
        wrappers_settings=wrappers_settings,
        wrapper_kwargs=dict(wrappers=[HealthBasedRewardWrapper]),
        num_env=4,
        seed=42
    )

    print(f"✓ Created {num_envs} environments")

    # Create model
    print("\n[2/3] Creating PPO agent...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,

        # LEARNING RATE SCHEDULE
        learning_rate=linear_schedule(3e-4, 1e-5),

        # LARGE BATCHES
        n_steps=4096,
        batch_size=256,
        n_epochs=10,

        # PPO PARAMETERS
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,

        # DISCOUNT & GAE
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,

        # LARGE NETWORK
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 512], vf=[512, 512])],
            activation_fn=torch.nn.ReLU,
        ),

        device="auto",
    )
    print("✓ Agent created")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_envs,
        save_path=checkpoint_dir,
        name_prefix="sfiii_phase2"
    )

    # Train
    print("\n[3/3] Starting training...")
    print("=" * 70)
    print("TRAINING: 3,000,000 timesteps")
    print("Difficulty: 2 (Medium)")
    print("Expected time: 3-4 hours")
    print("=" * 70)
    print()

    try:
        model.learn(
            total_timesteps=3_000_000,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=5
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

    # Save
    final_path = "sfiii_agent_phase2_standalone"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("PHASE 2 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✓ Model saved: {final_path}.zip")
    print(f"  Location: {os.path.abspath(final_path + '.zip')}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print()
    print("Expected performance:")
    print("  vs Difficulty 1: 85-90% win rate")
    print("  vs Difficulty 2: 70-75% win rate")
    print("  vs Difficulty 3: 55-60% win rate")
    print("=" * 70)


if __name__ == "__main__":
    main()
