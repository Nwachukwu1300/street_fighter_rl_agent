"""
Test your trained DIAMBRA agent
Loads the trained model and watches it play with visual rendering

Run with: diambra run python test_agent.py
"""
import diambra.arena
from stable_baselines3 import PPO

def main():
    print("=" * 60)
    print("Testing Trained Agent")
    print("=" * 60)

    # Load the trained model
    print("\nLoading trained model...")
    model_path = "sfiii_agent_phase2_default_rewards"
    print(f"Loading from: {model_path}.zip")
    model = PPO.load(model_path, device="cpu")
    print(f"✓ Model loaded successfully (Phase 2 - 96% complete, ~2.88M steps)")

    # Create environment with SAME settings as training but with rendering
    env_settings = diambra.arena.EnvironmentSettings()
    env_settings.difficulty = 2  # Test on same difficulty as training first
    env_settings.characters = ("Ken", "Ryu", "Chun-Li")
    env_settings.frame_shape = (84, 84, 1)  # Must match training settings
    env_settings.step_ratio = 6  # Same as training

    # Same wrappers as training
    wrappers_settings = diambra.arena.WrappersSettings()
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ["stage", "timer"]

    # Create environment with HUMAN rendering to watch the agent
    print("\nCreating environment with visual rendering...")
    env = diambra.arena.make("sfiii3n", env_settings, wrappers_settings, render_mode="human")
    print("✓ Environment created")

    # Run test episodes
    num_episodes = 3
    print(f"\n{'=' * 60}")
    print(f"Running {num_episodes} test episodes...")
    print(f"{'=' * 60}\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        print(f"Episode {episode + 1}/{num_episodes} - Starting...")

        while not done:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step_count += 1

            # Render the game (visual window)
            env.render()

        print(f"Episode {episode + 1} finished!")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Result: {info.get('round_done', 'Unknown')}")
        print()

    env.close()
    print("=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
