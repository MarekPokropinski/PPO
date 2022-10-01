import gym

from FlapPyBird_Env.flappy_env import FlappyEnv
from utils import LuminanceWrapper, StackObservation, ScaleObservation


def make_env():
    env = FlappyEnv()
    env = LuminanceWrapper(env, normalized=True)
    env = ScaleObservation(env, (64, 64))
    env = StackObservation(env, 1)
    return env

if __name__ == '__main__':
    from PPO import PPO

    env = gym.vector.SyncVectorEnv([make_env for _ in range(24)])
    ppo = PPO(env.observation_space, env.action_space, entropy_coeff=0.01)
    ppo.load('models/flappy')
    ppo.learn(env, 20000, steps_per_ep=64, mb_size=256, epochs_per_ep=4, lr=lambda x: x * 3e-4, clip_range=0.2, model_name='models/flappy', start_from_ep=8721)

