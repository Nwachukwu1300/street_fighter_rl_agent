import diambra.arena
from stable_baselines3 import PPO

# Model path and game ID
MODEL_PATH = "./sfiii_phase3_5678200_steps.zip"
GAME_ID = "sfiii3n"

# Load the trained agent
agent = PPO.load(MODEL_PATH)

# Environment settings setup
env_settings = diambra.arena.EnvironmentSettings()
env_settings.characters = ("Ken", "Ryu", "Chun-Li")
env_settings.frame_shape = (84, 84, 1)
env_settings.step_ratio = 6

# Wrapper settings
wrappers_settings = diambra.arena.WrappersSettings()
wrappers_settings.no_attack_buttons_combinations = True
wrappers_settings.normalize_reward = True
wrappers_settings.stack_frames = 4
wrappers_settings.flatten = True
wrappers_settings.filter_keys = ["stage", "timer"]

# Environment creation
env = diambra.arena.make(GAME_ID, env_settings, wrappers_settings)

# Agent-Environment loop
obs, info = env.reset()

while True:
    # Predict the next action using the trained agent
    action, _ = agent.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        if info["env_done"]:
            break

# Close the environment
env.close()