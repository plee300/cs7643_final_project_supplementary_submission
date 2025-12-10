# algorithms/dqn.py
import torch
import torch.optim as optim
import torch.nn.functional as F


class DQN:
    """Deep Q-Network algorithm logic with Double DQN support."""

    def __init__(self, policy_net, target_net, config):
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()

        dqn_config = config['dqn']
        self.gamma = dqn_config['gamma']
        self.use_double_dqn = dqn_config.get('double', False)
        self.optimizer = optim.Adam(policy_net.parameters(), lr=dqn_config['learning_rate'], eps=1e-5)

    def update(self, batch):
        obs_batch, actions_batch, rewards_batch, next_obs_batch, done_masks_batch = batch

        # Get Q-values for the actions that were actually taken
        current_q_values = self.policy_net(obs_batch).gather(1, actions_batch)

        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Use policy_net to select action, target_net to evaluate it
                next_actions = self.policy_net(next_obs_batch).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_obs_batch).gather(1, next_actions)
            else:
                # Standard DQN: Use target_net for both selection and evaluation
                next_q_values = self.target_net(next_obs_batch).max(1)[0].unsqueeze(1)

            # Compute the target Q-value using the Bellman equation
            target_q_values = rewards_batch + self.gamma * next_q_values * done_masks_batch

        # Use Huber loss for more stability
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())