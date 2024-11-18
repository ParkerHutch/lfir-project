# Train Cinnamon in the obstacle environment, starting from the model pretrained to walk in an empty environment
python ppo.py --env_id="Cinnamon-Reach-v1" --num_envs=1024 --update_epochs=8 --num_minibatches=32 --total_timesteps=25_000_000 --num-steps=200 --num-eval-steps=200 --gamma=0.99 --gae_lambda=0.95 --checkpoint ./runs/Cinnamon_Walking/final_ckpt.pt
# => eval_success_at_end_mean=0.25

# to train with the obstacle penalty, modify ppo.py to import the environment from cinnamon_task_obstacle_penalty and also use Cinnamon-Reach-v2 as the value for the env_id parameter in the command