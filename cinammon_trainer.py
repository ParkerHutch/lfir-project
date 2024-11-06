# adapted from https://github.com/haosulab/ManiSkill/blob/61076748d9e2c7254b1863220615c120f5919d7f/examples/baselines/stable_baselines3/example.py#L13
from math import e
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sympy import det
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode

from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

# from reach_task import AnymalCReachEnv
# from reach_task import QuadrupedReachEnv

environment_name = 'AnymalC-Reach-v1'
def main():
    ms3_vec_env = gym.make(environment_name, num_envs=1024)
    max_episode_steps = gym_utils.find_max_episode_steps_value(ms3_vec_env)
    vec_env = ManiSkillSB3VectorEnv(ms3_vec_env)

    model = PPO("MlpPolicy", vec_env, gamma=0.99, gae_lambda=0.95, n_steps=200, batch_size=32, n_epochs=8, verbose=1)
    print("Starting training")
    model.learn(total_timesteps=25_000_000)
    model.save("ppo_walk")
    vec_env.close()
    print("Finished training")
    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_walk")
    
    eval_vec_env = gym.make(environment_name, num_envs=16, render_mode="rgb_array")
    eval_vec_env = RecordEpisode(eval_vec_env, output_dir="eval_videos", save_video=True, trajectory_name="eval_trajectory", max_steps_per_video=max_episode_steps)
    eval_vec_env = ManiSkillSB3VectorEnv(eval_vec_env)
    obs = eval_vec_env.reset()
    print("Now testing")
    for i in range(max_episode_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_vec_env.step(action)
if __name__ == "__main__":
    main()