# Train Cinnamon in the obstacle environment (without penalty, unless you follow the step in the comment below), starting from the model pretrained to walk in an empty environment
python ppo.py --env_id="Cinnamon-Reach-v1" --num_envs=1024 --update_epochs=8 --num_minibatches=32 --total_timesteps=25_000_000 --num-steps=200 --num-eval-steps=200 --gamma=0.99 --gae_lambda=0.95 --checkpoint ./runs/Cinnamon_Walking/final_ckpt.pt
# => eval_success_at_end_mean=0.25

# to train with the obstacle penalty, modify ppo.py to import the environment from cinnamon_task_obstacle_penalty and also use Cinnamon-Reach-v2 as the value for the env_id parameter in the command
python ppo.py --env_id="Cinnamon-Reach-v2" --num_envs=1024 --update_epochs=8 --num_minibatches=32 --total_timesteps=25_000_000 --num-steps=200 --num-eval-steps=200 --gamma=0.99 --gae_lambda=0.95 --checkpoint ./runs/Cinnamon_Walking/final_ckpt.pt

# Successful training of Cinnamon for obstacle 1 parameters:
# to train with the obstacle penalty, modify ppo.py to import the environment from cinnamon_task_obstacle_penalty and also use Cinnamon-Reach-v2 as the value for the env_id parameter in the command
python ppo.py --env_id="Cinnamon-Reach-v2" --num_envs=1024 --update_epochs=8 --num_minibatches=32 --total_timesteps=53_000_000 --num-steps=200 --num-eval-steps=200 --gamma=0.995 --gae_lambda=0.95 --checkpoint ./runs/Cinnamon_Walking/final_ckpt.pt

# Training of Cinnamon for obstacle 3 parameters:
# to train with the obstacle penalty, modify ppo.py to import the environment from cinnamon_task_obstacle_penalty and also use Cinnamon-Reach-v2 as the value for the env_id parameter in the command
python ppo.py --env_id="Cinnamon-Reach-v2" --num_envs=1024 --update_epochs=8 --num_minibatches=32 --total_timesteps=53_000_000 --num-steps=200 --num-eval-steps=200 --gamma=0.995 --gae_lambda=0.95 --checkpoint ./runs/Obstacle1_Optimized_Cinnamon/final_ckpt.pt


# use this command to evaluate a trained model
# model_checkpoint=./runs/Cinnamon-Reach-v2__ppo__1__1732461164/final_ckpt.pt
model_checkpoint=./runs/Obstacle1_Optimized_Cinnamon/final_ckpt.pt
python ppo.py --env_id="Cinnamon-Reach-v2" --num_envs=1024 --num-steps=200 --num-eval-steps=200 --checkpoint $model_checkpoint --evaluate

# The commands will now train cinnamon on the 3 obstacle environment. To go back to the single obstacle modify ppo.py to import the QuadrupedReachEnv from cinnamon_task_obstacle_penalty