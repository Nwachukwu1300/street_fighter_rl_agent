"""
DIAMBRA Arena Street Fighter III Training Script
Uses PPO from Stable Baselines3 with parallel environments

Run with: diambra run -s=4 python train_v1.py
"""
import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
import os

def main():
    print("=" * 60)
    print("DIAMBRA Arena - Street Fighter III Training")
    print("=" * 60)

    # Environment settings - use DIAMBRA's EnvironmentSettings class
    env_settings = diambra.arena.EnvironmentSettings()
    env_settings.difficulty = 3
    env_settings.characters = ("Ken", "Ryu", "Chun-Li")
    env_settings.frame_shape = (84, 84, 1)  # Smaller grayscale frames for faster training

    # Wrapper settings - flatten to create shallow dict (not nested)
    # Based on official DIAMBRA docs for SB3 compatibility
    wrappers_settings = diambra.arena.WrappersSettings()
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 4  # Stack 4 frames for motion
    wrappers_settings.flatten = True  # Creates shallow dict from nested dict
    # Filter to keep only essential non-image observations (frame is auto-included)
    wrappers_settings.filter_keys = ["stage", "timer"]

    # Create vectorized environment using DIAMBRA's native function
    # This will automatically use the number of environments specified by -s flag
    print("\n[1/4] Creating environments...")
    env, num_envs = make_sb3_env(
        "sfiii3n",
        env_settings=env_settings,
        wrappers_settings=wrappers_settings,
        use_subprocess=True,
        seed=42,
        log_dir_base="./logs/",
        allow_early_resets=True
    )

    print(f"✓ Successfully created {num_envs} parallel environment(s)")

    # Create PPO agent with optimized hyperparameters
    print("\n[2/4] Initializing PPO agent...")
    model = PPO(
        "MultiInputPolicy",  # For flattened dict observation spaces
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    print("✓ PPO agent initialized")

    # Train for 1 million steps
    print("\n[3/4] Starting training for 1,000,000 timesteps...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=1_000_000,
            progress_bar=True,
            log_interval=10
        )

        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving current progress...")

    except Exception as e:
        print(f"\n\n✗ Error during training: {e}")
        print("Saving current progress...")

    # Save the model
    print("\n[4/4] Saving model...")
    model_path = "sfiii_agent_v1"
    model.save(model_path)
    print(f"✓ Model saved as: {model_path}.zip")

    # Cleanup
    env.close()
    print("\n" + "=" * 60)
    print("Training session complete!")
    print(f"Model location: {os.path.abspath(model_path + '.zip')}")
    print("=" * 60)

if __name__ == "__main__":
    main()