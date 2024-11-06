import gymnasium as gym
import mani_skill.envs
import os
from stable_baselines3 import PPO
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
#from robot_dog_task import QuadrupedReachEnv

# Create videos directory if it doesn't exist
os.makedirs("Videos", exist_ok=True)

def train():
    # Create the base environment with fewer environments for initial testing
    print("Creating environment...")
    env = gym.make("AnymalC-Reach-v1", render_mode="rgb_array", num_envs = 700)
    print("Environment created successfully")

    print("Creating vector environment wrapper...")
    vec_env = ManiSkillSB3VectorEnv(env)
    print("Vector environment wrapper created successfully")

    # Define PPO agent and train the model
    print("Creating PPO model...")
    model = PPO("MlpPolicy", vec_env, gamma=0.5, gae_lambda=0.5, n_steps=100, batch_size=128,
            n_epochs=8, verbose=1)
    # model = PPO.load("walk", env=vec_env)
    print("PPO model created successfully")

    # Start learning
    print("Starting training...")
    model.learn(total_timesteps=500_000)
    print("Training completed")

    print("Saving model...")
    model.save("walk")
    print("Model saved successfully")

    vec_env.close()
    env.close()
    del model

def eval():
    print("Starting evaluation...")
    model = PPO.load("walk")
    eval_env = gym.make("AnymalC-Reach-v1", num_envs=1, render_mode="rgb_array", sim_backend = 'gpu')
    eval_env = RecordEpisode(
        eval_env,
        output_dir="Videos",
        save_video=True,
        save_trajectory=False,
        max_steps_per_video=200
    )
    eval_env = ManiSkillSB3VectorEnv(eval_env)
    
    success_count = 0
    num_of_trials = 10
    
    for trial in range(num_of_trials):
        print(f"Starting trial {trial + 1}/{num_of_trials}")
        obs = eval_env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = eval_env.step(action)
            done = dones[0]
            
            if info[0].get("success", False):
                success_count += 1
                print(f"Trial {trial + 1} succeeded!")
                break
    
    print(f"Success Rate Over {num_of_trials} Trials: {(success_count/num_of_trials) * 100}%")
    eval_env.close()

if __name__ == "__main__":
    train()
    #eval()