"""
Evaluate Phase 2 Model Performance
Tests the 96%-complete model to see if it's ready for hackathon submission

Run with: diambra run python evaluate_phase2.py
"""
import diambra.arena
from stable_baselines3 import PPO
import numpy as np

def main():
    print("=" * 70)
    print("EVALUATING PHASE 2 MODEL (96% complete, ~2.88M steps)")
    print("=" * 70)

    # Load the model
    print("\nLoading model...")
    model_path = "sfiii_agent_phase2_default_rewards"
    model = PPO.load(model_path, device="cpu")
    print(f"✓ Loaded: {model_path}.zip")

    # Test on multiple difficulties
    difficulties = [1, 2, 3]

    for difficulty in difficulties:
        print(f"\n{'=' * 70}")
        print(f"Testing on Difficulty {difficulty}")
        print("=" * 70)

        # Create environment with same settings as training
        env_settings = diambra.arena.EnvironmentSettings()
        env_settings.difficulty = difficulty
        env_settings.characters = ("Ken", "Ryu", "Chun-Li")
        env_settings.frame_shape = (84, 84, 1)
        env_settings.step_ratio = 6

        wrappers_settings = diambra.arena.WrappersSettings()
        wrappers_settings.no_attack_buttons_combinations = True
        wrappers_settings.normalize_reward = True
        wrappers_settings.stack_frames = 4
        wrappers_settings.flatten = True
        wrappers_settings.filter_keys = ["stage", "timer"]

        env = diambra.arena.make("sfiii3n", env_settings, wrappers_settings)

        # Run evaluation episodes
        num_episodes = 10
        wins = 0
        total_reward = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            total_reward.append(episode_reward)

            # Check if won (simple heuristic: positive reward means likely won)
            if episode_reward > 0:
                wins += 1

            print(f"  Episode {episode + 1}/10: Reward = {episode_reward:.2f}")

        env.close()

        # Statistics
        win_rate = (wins / num_episodes) * 100
        avg_reward = np.mean(total_reward)
        std_reward = np.std(total_reward)

        print(f"\n{'─' * 70}")
        print(f"Difficulty {difficulty} Results:")
        print(f"  Win Rate: {win_rate:.1f}% ({wins}/{num_episodes})")
        print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"{'─' * 70}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nExpected Performance (Phase 2 with default DIAMBRA rewards):")
    print("  Difficulty 1: 75-85% win rate")
    print("  Difficulty 2: 60-70% win rate")
    print("  Difficulty 3: 45-55% win rate")
    print("\nIf results are within these ranges, the 96% model is GOOD ENOUGH!")
    print("=" * 70)

if __name__ == "__main__":
    main()
