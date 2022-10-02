import gym

from FlapPyBird_Env.flappy_env import FlappyEnv
from utils import LuminanceWrapper, StackObservation, ScaleObservation, normalize_obs


def make_env():
    env = FlappyEnv()
    env = LuminanceWrapper(env, normalized=True)
    env = ScaleObservation(env, (64, 64))
    env = StackObservation(env, 1)
    return env


if __name__ == '__main__':
    from PPO import PPO
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio

    single_env = make_env()
    env = gym.vector.SyncVectorEnv([lambda: single_env])

    ppo = PPO(env.observation_space, env.action_space)

    ppo.load('models~/flappy')

    while True:
        done = False
        obs = env.reset()
        step = 0
        score = 0

        while not done:
            action, _, _ = ppo.act(obs)
            obs_, [reward], [done], _, _ = env.step(action)

            single_env.render()
            score += reward

            if done:
                print(score)
                score = 0
            obs = obs_
            step += 1

