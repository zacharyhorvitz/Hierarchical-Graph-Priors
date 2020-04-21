import pytorch_lightning as pl
import torch 
import random
from collections import namedtuple

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer(torch.utils.data.IterableDataset):
    def __init__(self, replay_buffer, max_size, sample_size):
        super(ReplayBuffer).__init__()
        self.max_size = max_size
        self.sample_size = sample_size 
        self.buffer = replay_buffer

    def __iter__(self):
        minibatch = random.sample(self.buffer, self.sample_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], next_states[i], dones[i]
