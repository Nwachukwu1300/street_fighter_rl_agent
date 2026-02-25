"""Check DIAMBRA observation space structure"""
import diambra.arena

env = diambra.arena.make("sfiii3n")
print("Observation space:")
print(env.observation_space)
print("\nObservation space keys:")
if hasattr(env.observation_space, 'spaces'):
    for key, space in env.observation_space.spaces.items():
        print(f"  {key}: {space}")
        if hasattr(space, 'spaces'):
            print(f"    ^ NESTED DICT - has sub-keys:")
            for subkey in space.spaces.keys():
                print(f"      - {subkey}")
env.close()
