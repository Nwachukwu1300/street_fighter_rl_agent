"""
Simplest possible training script - no checkpoints, minimal imports
Run with: caffeinate -d diambra run -s=4 python train_simple.py
"""
import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO

print("Creating environment...")

# Environment settings
env_settings = diambra.arena.EnvironmentSettings()
env_settings.difficulty = 3
env_settings.characters = ("Ken",)
env_settings.frame_shape = (84, 84, 1)
env_settings.step_ratio = 6

# Wrapper settings
wrappers_settings = diambra.arena.WrappersSettings()
wrappers_settings.no_attack_buttons_combinations = True
wrappers_settings.stack_frames = 4

# Create environment
env, num_envs = make_sb3_env(
    "sfiii3n",
    env_settings=env_settings,
    wrappers_settings=wrappers_settings,
    num_env=4,
    seed=42
)

print(f"Environment created with {num_envs} parallel instances")

# Create simple PPO model
print("Creating PPO agent...")
model = PPO("CnnPolicy", env, verbose=1, device="cpu")
print("Agent created")

# Train
print("Starting training (5M steps, ~5 hours)...")
try:
    model.learn(total_timesteps=5_000_000, progress_bar=True)
    model.save("sfiii_simple_agent")
    print("Training complete! Saved as sfiii_simple_agent.zip")
except KeyboardInterrupt:
    print("\nInterrupted - saving progress...")
    model.save("sfiii_simple_agent_interrupted")
    print("Saved as sfiii_simple_agent_interrupted.zip")
finally:
    env.close()
