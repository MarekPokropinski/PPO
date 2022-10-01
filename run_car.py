from os import truncate
import gym
from gym.wrappers import RecordVideo
import numpy as np
import imageio

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from PPO import PPO
    from utils import LuminanceWrapper, StackObservation, ScaleObservation, normalize_obs, BoundAction

    def make_env():
        env = gym.make('CarRacing-v2', render_mode="human")
        env = normalize_obs(env)
        env = BoundAction(env, low=0, high=1)
        return env

    single_env = make_env()
    env = gym.vector.SyncVectorEnv([lambda: single_env])
    ppo = PPO(env.observation_space, env.action_space)

    ppo.load('models/car')

    # while True:

    terminated = False
    truncated = False
    obs, _ = env.reset()
    step = 0
    score = 0
    while not terminated and not truncated:
        value, mu, sigma = ppo.model(obs)
        action, _, _ = ppo.act(obs)
        # print(action, mu, sigma)
        obs_, reward, terminated, truncated, _ = env.step(action.numpy())

        single_env.render()

        score += reward
        obs = obs_
        step += 1
    print(score)
