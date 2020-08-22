import gym
import numpy as np
import torch


class SortedARIState(gym.Wrapper):
    """
    Description:
        Return info dict as state, sorted according to keys -> idx (available as attribute)
    
    Notes:
        - N/A
    """
    def __init__(self, env):
        super(SortedARIState, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0,
            255,  # max value
            shape=(len(self.env.labels()),),
            dtype=np.uint8)

        self.label_to_idx = {k: i for i, k in enumerate(self.env.labels().keys())}

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # reset the env and get the current labeled RAM
        return np.array(
            list(map(lambda t: t[1],
                sorted(self.env.labels().items(), key=lambda t: self.label_to_idx[t[0]]))))

    def step(self, action):
        # we don't need the obs here, just the labels in info
        _, reward, done, info = self.env.step(action)
        # grab the labeled RAM out of info and put as next_state
        next_state = np.array(
            list(map(lambda t: t[1],
                sorted(info['labels'].items(), key=lambda t: self.label_to_idx[t[0]]))))

        return next_state, reward, done, info
