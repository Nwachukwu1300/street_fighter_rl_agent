"""
DIAMBRA Arena Street Fighter III Training Script - COLOR VERSION
Uses RGB frames instead of grayscale for better visual quality

Run with: diambra run -s=4 python train_v2_color.py
"""
import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
import os

def main():
    print("=" * 60)
    print("DIAMBRA Arena - Street Fighter III Training (COLOR)")
    print("=" * 60)

    # Environment settings - USE COLOR FRAMES
    env_settings = diambra.arena.EnvironmentSettings()
    env_settings.difficulty = 3
    env_settings.characters = ("Ken", "Ryu", "Chun-Li")
    env_settings.frame_shape = (84, 84, 3)  # RGB - 3 channels for color!

    # Wrapper settings
    wrappers_settings = diambra.arena.WrappersSettings()
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ["stage", "timer"]

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

    print("\n[2/4] Initializing PPO agent...")
    model = PPO(
        "MultiInputPolicy",
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
    model_path = "sfiii_agent_v2_color"
    model.save(model_path)
    print(f"✓ Model saved as: {model_path}.zip")

    env.close()
    print("\n" + "=" * 60)
    print("Training session complete!")
    print(f"Model location: {os.path.abspath(model_path + '.zip')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
