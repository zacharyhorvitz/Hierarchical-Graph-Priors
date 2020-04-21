import torch
import numpy as np
import pytorch_lightning as pl
import random
from collections import deque 

from model import DQN_CNN_model, DQN_MLP_model
from replay_buffer import ReplayBuffer

class DQN_agent(pl.LightningModule):
    """Docstring for DQN agent """

    def __init__(self,
                 state_space,
                 action_space,
                 num_actions,
                 target_moving_average,
                 gamma,
                 replay_buffer_size,
                 epsilon_decay,
                 epsilon_decay_end,
                 warmup_period,
                 double_DQN,
                 model_type="mlp",
                 num_frames=None):
        """Defining DQN agent
        """
        super(DQN_agent, self).__init__()
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        if model_type == "mlp":
            self.online = DQN_MLP_model(state_space, action_space,
                                        num_actions)
            self.target = DQN_MLP_model(state_space, action_space,
                                        num_actions)
        elif model_type == "cnn":
            assert num_frames
            self.num_frames = num_frames
            self.online = DQN_CNN_model(state_space,
                                        action_space,
                                        num_actions,
                                        num_frames=num_frames)
            self.target = DQN_CNN_model(state_space,
                                        action_space,
                                        num_actions,
                                        num_frames=num_frames)
        else:
            raise NotImplementedError(model_type)

        self.online = self.online
        self.target = self.target

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.gamma = gamma
        self.target_moving_average = target_moving_average
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_end = epsilon_decay_end
        self.warmup_period = warmup_period
        self.device = device

        self.model_type = model_type
        self.double_DQN = double_DQN

    def loss_func(self, minibatch, writer=None, writer_step=None):
        # Make tensors
        state_tensor = torch.Tensor(minibatch.state).permute(0, 3, 1,
                                                             2)
        next_state_tensor = torch.Tensor(minibatch.next_state).permute(
            0, 3, 1, 2).to(self.device)
        action_tensor = torch.Tensor(minibatch.action)
        reward_tensor = torch.Tensor(minibatch.reward)
        done_tensor = torch.Tensor(minibatch.done)

        # Get q value predictions
        q_pred_batch = self.online(state_tensor).gather(
            dim=1, index=action_tensor.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_DQN:
                selected_actions = self.online.argmax_over_actions(
                    next_state_tensor)
                q_target = self.target(next_state_tensor).gather(
                    dim=1,
                    index=selected_actions.long().unsqueeze(1)).squeeze(1)
            else:
                q_target = self.target.max_over_actions(
                    next_state_tensor.detach()).values

        q_label_batch = reward_tensor + (self.gamma) * (1 -
                                                        done_tensor) * q_target
        q_label_batch = q_label_batch.detach()

        # Logging
        if writer:
            writer.add_scalar('training/batch_q_label', q_label_batch.mean(),
                              writer_step)
            writer.add_scalar('training/batch_q_pred', q_pred_batch.mean(),
                              writer_step)
            writer.add_scalar('training/batch_reward', reward_tensor.mean(),
                              writer_step)
        return torch.nn.functional.mse_loss(q_label_batch, q_pred_batch)

    def sync_networks(self):
        sync_networks(self.target, self.online, self.target_moving_average)

    def set_epsilon(self, global_steps, writer=None):
        if global_steps < self.warmup_period:
            self.online.epsilon = 1
            self.target.epsilon = 1
        else:
            self.online.epsilon = max(
                self.epsilon_decay_end,
                1 - (global_steps - self.warmup_period) / self.epsilon_decay)
            self.target.epsilon = max(
                self.epsilon_decay_end,
                1 - (global_steps - self.warmup_period) / self.epsilon_decay)
        if writer:
            writer.add_scalar('training/epsilon', self.online.epsilon,
                              global_steps)
