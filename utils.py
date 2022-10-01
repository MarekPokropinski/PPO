from gym import Env, Wrapper, ObservationWrapper, spaces, ActionWrapper
import gym
import numpy as np
from collections import deque
import cv2

class LuminanceWrapper(ObservationWrapper):
    def __init__(self, env, normalize=False, normalized=False):
        super().__init__(env)
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        
        shape = self.observation_space.shape
        self.dtype = self.observation_space.dtype if normalize==False else 'float32'
        self.observation_space = spaces.Box(low, high, shape=(shape[:-1]), dtype=self.dtype)
        self.normalize = normalize
        self.normalized = normalized

    def observation(self, observation):
        f1, f2, f3 = 0.299, 0.587, 0.114
        scale = 255 if not self.normalized else 1
        
        r = observation[..., 0]/scale
        g = observation[..., 1]/scale
        b = observation[..., 2]/scale

        if self.normalize or self.normalized:
            return np.sqrt(f1*r**2+f2*g**2+f3*b**2, dtype='float32')
        else:
            return (np.sqrt(f1*r**2+f2*g**2+f3*b**2)*255).astype('int8')

class StackObservation(ObservationWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        shape = self.observation_space.shape
        dtype = self.observation_space.dtype
        if len(shape)==2:
            self.observation_space = spaces.Box(low, high, shape=(*shape, k), dtype=dtype)
        else:
            self.observation_space = spaces.Box(low, high, shape=(*shape[:-1], shape[-1]*k), dtype=dtype)
        self.k = k
        self.observations = deque(maxlen=k)

    def reset(self, **kwargs):
        self.observations.clear()
        return super().reset(**kwargs)

    def observation(self, observation):
        self.observations.append(observation)
        # initially fill observations with 1st frame
        while len(self.observations)!=self.k:
            self.observations.append(observation)
        
        if len(observation.shape)==2:
            stacked = np.stack(self.observations, axis=-1)
        else:
            stacked = np.concatenate(self.observations, axis=-1)
        assert stacked.dtype == np.float32, f'{stacked.dtype}'
        return stacked

class ScaleObservation(ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)
        self.size = size
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        shape = self.observation_space.shape
        dtype = self.observation_space.dtype
        self.observation_space = spaces.Box(low, high, shape=size, dtype=dtype)

    def observation(self, observation):
        return cv2.resize(observation, self.size, interpolation=cv2.INTER_AREA)
        # return cv2.resize(observation, self.size, interpolation=cv2.INTER_LINEAR)

class BoundAction(ActionWrapper):
    def __init__(self, env:gym.Env, low=0, high=1):
        super().__init__(env)
        self.old_low = env.action_space.low
        self.old_high = env.action_space.high
        shape = env.action_space.shape
        self.action_space = spaces.Box(low=low, high=high, shape=shape)

    def action(self, action):
        l, h = self.action_space.low, self.action_space.high
        L, H = self.old_low, self.old_high

        return (action-l)/(h-l)*(H-L)+L


class PrintAction(ActionWrapper):
    def __init__(self, env:gym.Env):
        super().__init__(env)
        

    def action(self, action):
        print(action)

        return action



class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(low, high, shape=(np.prod(shape),))

    def observation(self, observation):
        return np.reshape(observation, (-1,))

class PartialObservation(ObservationWrapper):
    def __init__(self, env, idx):
        super().__init__(env)
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(low, high, shape=(len(idx),))
        self.idx=idx

    def observation(self, observation):
        return observation[self.idx]

RAM_PLAYER_1_POS = 60
RAM_BALL_Y_POS = 54
BOUNCE_COUNT = 17
RAM_BALL_X_POS = 49

class reward_wrapper(gym.Env):
    def __init__(self, env, bounce_coeff=0.005):
        super().__init__()
        self.env = env
        self.prev_state = None
        self.obs = None
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Discrete(4)
        self.bounce_coeff = bounce_coeff

    def reward(self, rew):
        state = self.obs
        prev_state = self.prev_state

        diff = 0
        if state[RAM_BALL_X_POS] > 155:
            diff = state[BOUNCE_COUNT] - prev_state[BOUNCE_COUNT]

        return diff*self.bounce_coeff + rew

    def reset(self):
        self.obs = self.env.reset()
        self.prev_state = self.obs
        return self.obs

    def step(self, action):
        self.prev_state = self.obs
        self.obs, reward, done, _ = self.env.step(action)
        return self.obs, self.reward(reward), done, {}

    def render(self, mode='human'):
        return self.env.render(mode)

class normalize_obs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.mean(self.observation_space.low)
        high = np.mean(self.observation_space.high)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(low, high, shape=shape, dtype='float32')

    @staticmethod
    def convert_obs(obs):
        return ((obs-127.5)/127.5).astype('float32')

    def observation(self, obs):
        return normalize_obs.convert_obs(obs)