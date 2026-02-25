"""
Train Phase 2 - Using CORRECT DIAMBRA API
No custom wrappers, just train with default DIAMBRA rewards

Run with: caffeinate -d diambra run -s=4 python train_phase2_correct.py
"""
print("=" * 70)
print("PHASE 2 TRAINING - Medium Difficulty (Correct)")
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
    print("\nThis trains Phase 2 from scratch (difficulty 2)")
    print("Using DIAMBRA's default rewards (no custom wrapper)")
    print("=" * 70)

    checkpoint_dir = "./models_phase2/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Environment settings
    print("\n[1/3] Configuring environment...")
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
    wrappers_settings.normalize_reward = True  # Use DIAMBRA's normalization
    wrappers_settings.filter_keys = ["stage", "timer"]  # Keep it simple

    print("✓ Settings configured")
    print("\nCreating parallel environments (4 workers)...")
    print("(This may take 1-2 minutes)")

    # Create environments - CORRECT API usage
    env, num_envs = make_sb3_env(
        "sfiii3n",
        env_settings=env_settings,
        wrappers_settings=wrappers_settings,
        use_subprocess=True,  # Enable parallel envs
        seed=42
    )

    print(f"✓ Created {num_envs} parallel environments")

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
    final_path = "sfiii_agent_phase2_default_rewards"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("PHASE 2 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✓ Model saved: {final_path}.zip")
    print(f"  Location: {os.path.abspath(final_path + '.zip')}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print()
    print("Expected performance with default rewards:")
    print("  vs Difficulty 2: 60-70% win rate")
    print("  vs Difficulty 3: 45-55% win rate")
    print("=" * 70)


if __name__ == "__main__":
    main()
