"""
DIAMBRA Competition Agent Submission
Street Fighter III: 3rd Strike PPO Agent
"""
import diambra.arena
from stable_baselines3 import PPO
import os


class DiambraAgent:
    """Competition-ready agent wrapper for DIAMBRA submission"""

    def __init__(self):
        self.model = None
        self.env = None
        self.load_model()

    def load_model(self):
        """Load the trained PPO model"""
        model_path = "./models_phase3/sfiii_phase3_5678200_steps.zip"
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path)
        print("Model loaded successfully!")

    def setup_environment(self, difficulty=3):
        """Configure environment with same settings as training"""
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

        self.env = diambra.arena.make(
            "sfiii3n",
            env_settings=env_settings,
            wrappers_settings=wrappers_settings
        )

    def predict(self, observation):
        """Get action prediction from model"""
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def run_episode(self, num_episodes=1):
        """Run agent for N episodes"""
        if self.env is None:
            self.setup_environment()

        results = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1

            results.append({
                "episode": episode + 1,
                "reward": episode_reward,
                "steps": steps
            })

        return results


def main():
    """Test the agent before submission"""
    print("=" * 70)
    print("DIAMBRA SUBMISSION - Agent Test")
    print("=" * 70)

    agent = DiambraAgent()

    print("\nTesting agent on 5 episodes...")
    results = agent.run_episode(num_episodes=5)

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    for result in results:
        print(f"Episode {result['episode']}: Reward={result['reward']:.2f}, Steps={result['steps']}")

    avg_reward = sum(r['reward'] for r in results) / len(results)
    print(f"\nAverage Reward: {avg_reward:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()