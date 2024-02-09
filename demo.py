from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.monitor import Monitor
from env import cartpoleenv

import os
from datetime import datetime

# ---------------- Path definition
model_path = "./Model/saved_models"
# ---------------- Create environment
env = cartpoleenv.CartPoleEnv(action_type='discrete') # action_type can be set as discrete or continuous
check_env(env)

# ---------------- Callback functions
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)


# ---------------- Model
# Option 1: create a new model
print("create a new model")
model = A2C(policy='MlpPolicy', env=env, learning_rate=7e-4, verbose=True, tensorboard_log="./data/")

# Option 2: load the model from files (note that the loaded model can be learned again)
# print("load the model from files")
# model = A2C.load("../CartPole/saved_models/tmp/best_model", env=env)
# model.learning_rate = 1e-4

# Option 3: load the pre-trained model from files
# print("load the pre-trained model from files")
# if env.unwrapped.action_type == 'discrete':
#     model = A2C.load(os.path.join(model_path, "best_model_discrete"), env=env)
# else:
#     model = A2C.load(os.path.join(model_path, "best_model_continuous"), env=env)


# ---------------- Learning
# Use tensorboard to monitor the training process
print('Learning the model')
model.learn(total_timesteps=2000) # 'MlpPolicy' = Actor Critic Policy
print('Finished')
model_name = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
model.save(os.path.join(model_path, model_name))
print("Model saved as " + model_name)
# del model # delete the model and load the best model to predict
# model = A2C.load("Model/saved_models/tmp/best_model", env=env) # This is a discrete model


# ---------------- Prediction
print('Prediction')

for _ in range(10):
    observation, info = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, terminated, info = env.step(action)
        episode_reward += reward
    
    print([episode_reward, env.counts])

env.close()