# models/agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticRNN(nn.Module):
    """PPO Model: Actor-Critic with a recurrent (LSTM) core."""

    def __init__(self, input_shape, num_actions, recurrent_hidden_dim=256):
        super(ActorCriticRNN, self).__init__()
        # PyTorch convolutions expect (N, C, H, W).
        # Our wrapper gives (H, W, C*k), so we need to know the channels.
        h, w, c = input_shape

        self.cnn_base = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn_base(dummy_input).shape[1]

        self.lstm = nn.LSTMCell(cnn_out_dim, recurrent_hidden_dim)

        # Actor head
        self.actor_head = nn.Linear(recurrent_hidden_dim, num_actions)

        # Critic head
        self.critic_head = nn.Linear(recurrent_hidden_dim, 1)

    def forward(self, obs, hidden_state, cell_state):
        # Input obs shape: (N, H, W, C). Permute to (N, C, H, W) for PyTorch CNNs.
        if obs.ndim == 4:
            obs = obs.permute(0, 3, 1, 2)

        cnn_features = self.cnn_base(obs.float() / 255.0)  # Normalize pixels
        hidden_state, cell_state = self.lstm(cnn_features, (hidden_state, cell_state))

        # Actor output
        action_logits = self.actor_head(hidden_state)
        action_dist = Categorical(logits=action_logits)

        # Critic output
        state_value = self.critic_head(hidden_state)

        return action_dist, state_value, (hidden_state, cell_state)



