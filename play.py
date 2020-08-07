import retro
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines import PPO2, A2C
import numpy as np
import gym
from stable_baselines.common.callbacks import CheckpointCallback
from utils import *

if __name__ == "__main__":
    num_envs = 8 # Must use the save number of envs as trained on but we create a single dummy env for testing.
    envs = SubprocVecEnv([make_env] * num_envs)    
    envs = VecFrameStack(envs, n_stack=4)

    model = PPO2.load("./training_checkpoints/your_model.zip")
    model.set_env(envs)
    obs = envs.reset()
    print(obs.shape)

    # Create one env for testing 
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()

    # model.predict(test_obs) would through an error
    # because the number of test env is different from the number of training env
    # so we need to complete the observation with zeroes
    zero_completed_obs = np.zeros((num_envs,) + envs.observation_space.shape)
    zero_completed_obs[0, :] = obs
    obs = zero_completed_obs

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render(mode="human")
        if dones.all() == True:
            break
            
        zero_completed_obs = np.zeros((num_envs,) + envs.observation_space.shape)
        zero_completed_obs[0, :] = obs
        obs = zero_completed_obs
        

