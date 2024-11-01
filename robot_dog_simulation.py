import gymnasium as gym
import mani_skill.envs
import time
import numpy as np
import os
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.agents.controllers import PDEEPoseControllerConfig
from robot_dog_task import QuadrupedReachEnv

# Create videos directory if it doesn't exist
os.makedirs("Videos", exist_ok=True)

# Create the base environment
env = gym.make("AnymalC-Move-v1", render_mode="rgb_array")

# Wrap the environment with RecordEpisode
env = RecordEpisode(
    env,
    output_dir="Videos",
    save_trajectory=False,
    save_video=True,
    video_fps=30,
    max_steps_per_video=100,
    info_on_video=True  # This will overlay episode information on the video
)

# Reset the environment
obs, _ = env.reset(seed=0)

# Print environment details
env.unwrapped.print_sim_details()

# Run the episode
done = False
truncated = False
start_time = time.time()
total_reward = 0

while not done and not truncated:
    # Sample a random action
    action = env.action_space.sample()
    
    # Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    # Force render each frame (important for video recording)
    env.render()

# Calculate FPS
end_time = time.time()
elapsed_time = end_time - start_time
N = info["elapsed_steps"].item()
FPS = N / elapsed_time

print(f"Episode finished!")
print(f"Total steps: {N}")
print(f"Total reward: {total_reward}")
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Frames Per Second = {FPS:.2f}")

# Make sure to close the environment
env.close()