# utils/replay_buffer.py
import torch
import numpy as np


class RolloutStorage:
    """Storage buffer for on-policy algorithms like PPO."""

    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        self.num_steps = num_steps
        self.num_processes = num_processes
        h, w, c = obs_shape

        self.obs = torch.zeros(num_steps + 1, num_processes, c, h, w, dtype=torch.uint8)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.recurrent_cell_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.recurrent_cell_states = self.recurrent_cell_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        return self

    def insert(self, obs, recurrent_states, actions, action_log_probs, value_preds, rewards, masks):
        hidden, cell = recurrent_states
        # Store obs in (C, H, W) format directly
        self.obs[self.step + 1].copy_(obs.permute(0, 3, 1, 2))
        self.recurrent_hidden_states[self.step + 1].copy_(hidden)
        self.recurrent_cell_states[self.step + 1].copy_(cell)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.recurrent_cell_states[0].copy_(self.recurrent_cell_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        if use_gae:
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def get_batch_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = torch.randperm(batch_size)

        flat_obs = self.obs[:-1].view(-1, *self.obs.size()[2:])
        flat_hidden = self.recurrent_hidden_states[:-1].view(-1, self.recurrent_hidden_states.size(-1))
        flat_cell = self.recurrent_cell_states[:-1].view(-1, self.recurrent_cell_states.size(-1))
        flat_actions = self.actions.view(-1, self.actions.size(-1))
        flat_log_probs = self.action_log_probs.view(-1, 1)
        flat_advantages = advantages.view(-1, 1)
        flat_returns = self.returns[:-1].view(-1, 1)

        for i in range(num_mini_batch):
            indices = sampler[i * mini_batch_size: (i + 1) * mini_batch_size]
            yield (
                flat_obs[indices], (flat_hidden[indices], flat_cell[indices]),
                flat_actions[indices], flat_log_probs[indices],
                flat_returns[indices], flat_advantages[indices]
            )


class ReplayBuffer:
    """Replay buffer for off-policy algorithms like DQN."""

    def __init__(self, buffer_size, obs_shape, device):
        self.buffer_size = buffer_size
        self.device = device
        self.obs = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.ByteTensor(self.obs[indices]).to(self.device),
            torch.LongTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.ByteTensor(self.next_obs[indices]).to(self.device),
            torch.FloatTensor(1.0 - self.dones[indices]).to(self.device)
        )

    def __len__(self):
        return self.size