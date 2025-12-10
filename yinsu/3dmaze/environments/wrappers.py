import gym
from gym import spaces
import numpy as np
from collections import deque
import cv2

cv2.setNumThreads(0)

# ===================================================================
#           OBSERVATION WRAPPERS (Compatible with gym 0.23.1)
# ===================================================================

class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB observations to grayscale."""
    def __init__(self, env):
        super().__init__(env)
        original_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(original_shape[0], original_shape[1], 1), 
            dtype=np.uint8
        )

    def observation(self, obs):
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray_obs, axis=-1)


class ResizeObservation(gym.ObservationWrapper):
    """Resize observations to a specified shape."""
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = tuple(shape)
        original_space = self.observation_space
        new_channel_dim = original_space.shape[2]
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=self.shape + (new_channel_dim,), 
            dtype=np.uint8
        )

    def observation(self, obs):
        resized_obs = cv2.resize(
            obs, 
            (self.shape[1], self.shape[0]), 
            interpolation=cv2.INTER_AREA
        )
        if resized_obs.ndim == 2:
            resized_obs = np.expand_dims(resized_obs, axis=-1)
        return resized_obs


class FrameStack(gym.ObservationWrapper):
    """
    Stack the last k frames together.
    Compatible with gym 0.23.1 (old API: reset returns obs, step returns 4 values).
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        original_shape = self.observation_space.shape
        new_shape = (original_shape[0], original_shape[1], original_shape[2] * k)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=new_shape, 
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        # gym 0.23.1: reset() returns just obs (not a tuple)
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation()

    def step(self, action):
        # gym 0.23.1: step() returns (obs, reward, done, info) - 4 values
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)