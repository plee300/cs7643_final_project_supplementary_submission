import os
import time
import yaml
import datetime
import torch
import numpy as np
import random
from collections import deque

import gym
import memory_maze

from models.agent import ActorCriticRNN
from models.dqn_agent import DQNAgent
from algorithms.ppo import PPO
from algorithms.dqn import DQN
from utils.replay_buffer import RolloutStorage, ReplayBuffer
from utils.logger import get_logger
from environments.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

try:
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
except ImportError:
    print("Please install stable-baselines3: pip install stable-baselines3")
    exit()

from environments.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

def create_env(env_id, config, is_vec_env=False, n_envs=1):
    def env_creator():
        env = gym.make(env_id)
        
        # No compatibility wrappers needed with gym 0.23.1!
        wrapper_config = config.get('env', {}).get('wrappers', {})
        if wrapper_config.get('grayscale', {}).get('enabled', False):
            env = GrayScaleObservation(env)
        if wrapper_config.get('resize', {}).get('enabled', False):
            shape = wrapper_config['resize']['shape']
            env = ResizeObservation(env, shape=tuple(shape))
        if wrapper_config.get('frame_stack', {}).get('enabled', False):
            k = wrapper_config['frame_stack']['k']
            env = FrameStack(env, k)
        return env

    if is_vec_env:
        return make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs={}, wrapper_class=lambda e: apply_wrappers(e, config))
    else:
        return env_creator()

def apply_wrappers(env, config):
    """Helper function to apply wrappers in the correct order for VecEnv."""
    wrapper_config = config.get('env', {}).get('wrappers', {})
    
    wrapper_config = config.get('env', {}).get('wrappers', {})
    if wrapper_config.get('grayscale', {}).get('enabled', False):
        env = GrayScaleObservation(env)
    if wrapper_config.get('resize', {}).get('enabled', False):
        shape = wrapper_config['resize']['shape']
        env = ResizeObservation(env, shape=tuple(shape))
    if wrapper_config.get('frame_stack', {}).get('enabled', False):
        k = wrapper_config['frame_stack']['k']
        env = FrameStack(env, k)
    return env

def evaluate_and_render_one_episode(config, agent, device):
    """Evaluate the agent and render one episode for visualization."""
    print("\n--- Starting Live Visualization ---")
    env = create_env(config['env']['name'], config)
    obs = env.reset()  # gym 0.23.1 returns just obs, not tuple
    done = False
    total_reward = 0
    
    is_recurrent = isinstance(agent, ActorCriticRNN)
    if is_recurrent:
        hidden_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)
        cell_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)

    while not done:
        env.render()
        time.sleep(0.02)
        
        with torch.no_grad():
            obs_tensor = torch.ByteTensor(obs).unsqueeze(0).to(device)
            if is_recurrent:
                action_dist, _, (hidden_state, cell_state) = agent(obs_tensor, hidden_state, cell_state)
                action = action_dist.mode()
            else:
                q_values = agent(obs_tensor)
                action = q_values.max(1)[1]
        
        obs, reward, done, info = env.step(action.cpu().item())
        total_reward += reward
    
    env.close()
    print(f"--- Live Visualization Finished | Total Reward: {total_reward:.2f} ---\n")

def train_ppo(config, device, experiment_dir):
    """Train using PPO algorithm."""
    print("Starting PPO Training...")
    log_path = os.path.join(experiment_dir, "tensorboard")
    model_save_path = os.path.join(experiment_dir, "models")
    os.makedirs(model_save_path, exist_ok=True)
    logger = get_logger(logger_type=config['training']['logger_type'], log_dir=log_path)
    logger.log_hyperparams(config)

    ppo_config = config['ppo']
    envs = create_env(config['env']['name'], config, is_vec_env=True, n_envs=ppo_config['num_processes'])
    
    obs_space_shape = (
        config['env']['wrappers']['resize']['shape'][0],
        config['env']['wrappers']['resize']['shape'][1],
        config['env']['wrappers']['frame_stack']['k']
    )

    agent = ActorCriticRNN(obs_space_shape, envs.action_space.n, config['model']['recurrent_hidden_dim']).to(device)
    ppo_algo = PPO(agent, config)
    rollouts = RolloutStorage(
        ppo_config['num_steps_per_update'], 
        ppo_config['num_processes'],
        obs_space_shape, 
        envs.action_space, 
        config['model']['recurrent_hidden_dim']
    ).to(device)
    
    start_time = time.time()
    num_updates = int(config['training']['num_env_steps']) // ppo_config['num_steps_per_update'] // ppo_config['num_processes']
    episode_rewards = deque(maxlen=20)
    last_eval_step = 0

    obs = envs.reset()
    rollouts.obs[0].copy_(torch.from_numpy(obs).permute(0, 3, 1, 2))

    for update_idx in range(num_updates):
        for step in range(ppo_config['num_steps_per_update']):
            with torch.no_grad():
                recurrent_states = (rollouts.recurrent_hidden_states[step], rollouts.recurrent_cell_states[step])
                action_dist, value, new_recurrent_states = agent(torch.from_numpy(obs).to(device), *recurrent_states)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            obs, reward, done, infos = envs.step(action.cpu().numpy())
            
            reward_tensor = torch.from_numpy(reward).float().unsqueeze(1).to(device)
            masks = torch.FloatTensor(1.0 - done).unsqueeze(1).to(device)
            
            rollouts.insert(
                torch.from_numpy(obs), 
                new_recurrent_states, 
                action.unsqueeze(1), 
                log_prob.unsqueeze(1), 
                value, 
                reward_tensor, 
                masks
            )
            
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

        with torch.no_grad():
            final_recurrent_states = (rollouts.recurrent_hidden_states[-1], rollouts.recurrent_cell_states[-1])
            _, next_value, _ = agent(torch.from_numpy(obs).to(device), *final_recurrent_states)

        rollouts.compute_returns(next_value, ppo_config['use_gae'], ppo_config['gamma'], ppo_config['gae_lambda'])
        
        value_loss, policy_loss, entropy_loss = ppo_algo.update(rollouts)
        rollouts.after_update()
        
        total_steps = (update_idx + 1) * ppo_config['num_steps_per_update'] * ppo_config['num_processes']

        if config['live_visualization']['enabled']:
            if (total_steps - last_eval_step) >= config['live_visualization']['eval_interval_steps']:
                evaluate_and_render_one_episode(config, agent, device)
                last_eval_step = total_steps
        
        if (update_idx + 1) % (config['training']['save_interval_steps'] // (ppo_config['num_steps_per_update'] * ppo_config['num_processes'])) == 0:
            save_file = os.path.join(model_save_path, f"agent_step_{total_steps}.pth")
            torch.save(agent.state_dict(), save_file)
            print(f"Model saved: {save_file}")

        if len(episode_rewards) > 0 and (update_idx + 1) % 10 == 0:
            mean_reward = np.mean(episode_rewards)
            logger.log_scalar("rollout/mean_reward", mean_reward, total_steps)
            logger.log_scalar("losses/value_loss", value_loss, total_steps)
            logger.log_scalar("losses/policy_loss", policy_loss, total_steps)
            logger.log_scalar("losses/entropy", entropy_loss, total_steps)
            fps = int(total_steps / (time.time() - start_time))
            logger.log_scalar("performance/fps", fps, total_steps)
            print(f"Update {update_idx+1}/{num_updates} | Steps: {total_steps} | Mean Reward: {mean_reward:.2f} | FPS: {fps}")
    
    logger.close()
    envs.close()
    print("PPO Training Complete!")

def train_dqn(config, device, experiment_dir):
    """Train using DQN algorithm."""
    print("Starting DQN Training...")
    log_path = os.path.join(experiment_dir, "tensorboard")
    model_save_path = os.path.join(experiment_dir, "models")
    os.makedirs(model_save_path, exist_ok=True)
    logger = get_logger(logger_type=config['training']['logger_type'], log_dir=log_path)
    logger.log_hyperparams(config)

    dqn_config = config['dqn']
    env = create_env(config['env']['name'], config)
    
    obs_space_shape = (
        config['env']['wrappers']['resize']['shape'][0],
        config['env']['wrappers']['resize']['shape'][1],
        config['env']['wrappers']['frame_stack']['k']
    )
    num_actions = env.action_space.n
    
    use_dueling = dqn_config.get('dueling', False)
    print(f"Using Dueling Architecture: {use_dueling}")
    
    policy_net = DQNAgent(obs_space_shape, num_actions, use_dueling=use_dueling).to(device)
    target_net = DQNAgent(obs_space_shape, num_actions, use_dueling=use_dueling).to(device)
    
    dqn_algo = DQN(policy_net, target_net, config)
    replay_buffer = ReplayBuffer(dqn_config['buffer_size'], obs_space_shape, device)

    obs = env.reset()  # gym 0.23.1 returns just obs
    last_eval_step = 0
    episode_rewards = deque(maxlen=20)
    episode_reward = 0

    for step in range(1, config['training']['num_env_steps'] + 1):
        epsilon = calculate_epsilon(step, config)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.ByteTensor(obs).unsqueeze(0).to(device)
                q_values = policy_net(obs_tensor)
                action = q_values.max(1)[1].item()
        
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        
        if done:
            episode_rewards.append(episode_reward)
            obs = env.reset()  # gym 0.23.1 returns just obs
            episode_reward = 0

        if config['live_visualization']['enabled']:
            if (step - last_eval_step) >= config['live_visualization']['eval_interval_steps']:
                evaluate_and_render_one_episode(config, policy_net, device)
                last_eval_step = step

        if step > dqn_config['learning_starts'] and step % dqn_config['train_frequency'] == 0:
            batch = replay_buffer.sample(dqn_config['batch_size'])
            loss = dqn_algo.update(batch)
            if step % 1000 == 0:
                logger.log_scalar("losses/dqn_loss", loss, step)

        if step % dqn_config['target_update_frequency'] == 0:
            dqn_algo.update_target_network()
            
        if step % config['training']['log_interval_steps'] == 0 and len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            logger.log_scalar("rollout/mean_reward", mean_reward, step)
            logger.log_scalar("exploration/epsilon", epsilon, step)
            print(f"Step {step}/{config['training']['num_env_steps']} | Mean Reward: {mean_reward:.2f} | Epsilon: {epsilon:.3f}")
            
        if step % config['training']['save_interval_steps'] == 0:
            save_file = os.path.join(model_save_path, f"agent_step_{step}.pth")
            torch.save(policy_net.state_dict(), save_file)
            print(f"Model saved: {save_file}")
    
    logger.close()
    env.close()
    print("DQN Training Complete!")

def calculate_epsilon(step, config):
    """Calculate epsilon for epsilon-greedy exploration."""
    dqn_config = config['dqn']
    start = dqn_config['epsilon_start']
    end = dqn_config['epsilon_end']
    decay_steps = dqn_config['epsilon_decay_steps']
    fraction = min(1.0, step / decay_steps)
    return start + fraction * (end - start)

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = config.get('training', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and config['training']['device'] == 'auto' else "cpu")
    print(f"Using device: {device}")
    
    algo_name = config['algorithm_to_run'].upper()
    env_name = config['env']['name']
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{algo_name}_{env_name}_{timestamp}"
    
    output_dir = config['training']['output_dir']
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Results will be saved to: {experiment_dir}")
    print(f"{'='*60}\n")

    if algo_name == "PPO":
        train_ppo(config, device, experiment_dir)
    elif algo_name == "DQN":
        train_dqn(config, device, experiment_dir)
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm_to_run']}")
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()