# algorithms/ppo.py
import torch
import torch.optim as optim


class PPO:
    """Proximal Policy Optimization algorithm logic."""

    def __init__(self, agent, config):
        self.agent = agent
        ppo_config = config['ppo']
        self.clip_epsilon = ppo_config['clip_epsilon']
        self.ppo_epochs = ppo_config['ppo_epochs']
        self.num_mini_batches = ppo_config['num_mini_batches']
        self.value_loss_coef = ppo_config['value_loss_coef']
        self.entropy_coef = ppo_config['entropy_coef']
        self.max_grad_norm = ppo_config['max_grad_norm']
        self.optimizer = optim.Adam(agent.parameters(), lr=ppo_config['learning_rate'], eps=1e-5)

    def update(self, rollouts):
        # Advantage is computed from the critic's value predictions
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # Normalize advantages to stabilize training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for _ in range(self.ppo_epochs):
            batch_generator = rollouts.get_batch_generator(advantages, self.num_mini_batches)
            for sample in batch_generator:
                obs_batch, recurrent_states_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch = sample

                action_dist, state_values, _ = self.agent(obs_batch, *recurrent_states_batch)

                new_log_probs = action_dist.log_prob(actions_batch.squeeze(-1))
                entropy = action_dist.entropy().mean()

                # Policy Loss (Clipped Surrogate Objective)
                ratio = torch.exp(new_log_probs - old_log_probs_batch.squeeze(-1))
                surr1 = ratio * advantages_batch.squeeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_batch.squeeze(
                    -1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (MSE)
                value_loss = 0.5 * (returns_batch.squeeze(-1) - state_values.squeeze(-1)).pow(2).mean()

                # Total Loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()

        num_updates = self.ppo_epochs * self.num_mini_batches
        return total_value_loss / num_updates, total_policy_loss / num_updates, total_entropy_loss / num_updates