### 4. `models/dqn_agent.py`
# models/dqn_agent.py
import torch
import torch.nn as nn


class DQNAgent(nn.Module):
    """DQN Model: Can be configured as a standard or a Dueling DQN."""

    def __init__(self, input_shape, num_actions, use_dueling=False):
        super(DQNAgent, self).__init__()
        self.use_dueling = use_dueling
        h, w, c = input_shape

        self.cnn_base = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn_base(dummy_input).shape[1]

        if self.use_dueling:
            # Dueling Architecture: Two separate streams
            self.value_stream = nn.Sequential(
                nn.Linear(cnn_out_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1)  # Outputs a single value for the state V(s)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(cnn_out_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)  # Outputs an advantage for each action A(s,a)
            )
        else:
            # Standard DQN Head
            self.head = nn.Sequential(
                nn.Linear(cnn_out_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )

    def forward(self, obs):
        # Input obs shape: (N, H, W, C). Permute to (N, C, H, W) for PyTorch CNNs.
        if obs.ndim == 4:
            obs = obs.permute(0, 3, 1, 2)

        features = self.cnn_base(obs.float() / 255.0)  # Normalize pixels

        if self.use_dueling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            # Combine streams to get final Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A))
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.head(features)

        return q_values