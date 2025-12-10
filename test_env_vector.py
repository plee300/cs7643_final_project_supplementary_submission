import argparse
from collections import deque, namedtuple
import os
import random
os.environ['MUJOCO_GL'] = 'glfw'

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import json
import importlib
from tqdm import tqdm

INPUT_CHANNELS = 3
HEIGHT, WIDTH = 64, 64
N_ACTIONS = 6

# Hyperparameters (set by config in main())
BUFFER_CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10

# Named tuple for storing transitions in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))            

# Preprocessing: Convert numpy array frame to PyTorch tensor and reshape
def get_screen(observations : np.ndarray, device):
    screen = torch.from_numpy(observations).to(device)

    #reshape from (N, H, W, C) to (N, C, H, W) like torch expects
    screen = screen.permute((0, 3, 1, 2))
    
    #return conversion from uint8 to float32 in range [0.0, 1.0]
    return screen.float() / 255.0

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def load_dqn_class_from_config(config):

    module_name = config.get('module')
    class_name = config.get('class')
    if not module_name or not class_name:
        raise ValueError('config.json must specify "module" and "class"')

    module = importlib.import_module(module_name)
    dqn_class = getattr(module, class_name)
    init_kwargs = config.get('init_kwargs', {})
    return dqn_class, init_kwargs


def select_action(states:torch.Tensor, step:int, policy_net:nn.Module, env:gym.vector.AsyncVectorEnv, device:torch.device):
    sample = torch.rand(states.size(0), device=device)
    # Decay epsilon from EPS_START to EPS_END over EPS_DECAY steps
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * step / EPS_DECAY)
    
    exploit_mask = sample > eps_threshold
    explore_mask = ~exploit_mask
    
    actions = torch.zeros(states.size(0), dtype=torch.long, device=device)
    
    if exploit_mask.any():
        with torch.no_grad():
            # Select actions with the highest predicted Q-values for exploit_mask
            exploit_actions = policy_net(states[exploit_mask]).argmax(dim=1, keepdim=True)

            actions[exploit_mask] = exploit_actions.squeeze(-1)
    
    if explore_mask.any():
        # select random actions for exploring
        explore_actions = env.action_space.sample()
        explore_actions = torch.from_numpy(explore_actions).to(device)
        actions[explore_mask] = explore_actions[explore_mask]
                          
    return actions

def optimize_model(memory, policy_net, target_net, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    
    # Sample batch
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Convert batch of transitions to tensors
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)
    non_final_next_states = torch.stack([s.squeeze(0) for s in batch.next_state if s is not None])
    state_batch = torch.stack(batch.state).squeeze(1)
    action_batch = torch.cat(batch.action).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)

    # 1. Compute Q(s_t, a) - The Q-value for the action taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 2. Compute V(s_{t+1}) = max_a Q_{target}(s_{t+1}, a) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
    # 3. Compute the Expected Q values (Target): r + gamma * V(s_{t+1})
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 4. Compute Loss (Huber loss is common for DQN)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 5. Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping (optional but recommended for stability)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) 
    optimizer.step()

def train_dqn_agent(envs:gym.vector.AsyncVectorEnv, num_episodes:int, target_update, policy_net, target_net, optimizer, memory:ReplayBuffer, device):
    for i_episode in range(0, num_episodes, envs.num_envs):
        
        # Now that env is vectorized, observation should be (N, H, W, C)
        observations, infos = envs.reset()
        states = get_screen(observations, device) # shape is now (N, C, H, W)
        
        total_rewards = torch.zeros(envs.num_envs, device=device)
        
        # 1000 steps per episode per
        # https://github.com/jurgisp/memory-maze/tree/main#:~:text=because%20there%20are-,1000%20steps,-(actions)%20in%20a
        NUM_STEPS = 1000 
        
        step_iterator = tqdm(range(NUM_STEPS), desc=f"Episode {i_episode+1}/{num_episodes}")
        for t in step_iterator:
            # Select and execute action
            actions = select_action(states, t, policy_net, envs, device)
            
            #move the action to cpu and convert to numpy before passing to envs.step()
            observations, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
            
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            total_rewards += rewards
            
            next_states = get_screen(observations, device)
            
            # Store the transitions in memory as individual steps for each env
            for i in range(envs.num_envs):
                state = states[i].unsqueeze(0)
                action = actions[i].unsqueeze(0)
                next_state = next_states[i].unsqueeze(0)
                reward = rewards[i].unsqueeze(0)
                memory.push(state, action, next_state, reward)

            # Move to the next state
            states = next_states

            # Perform one optimization step
            optimize_model(memory, policy_net, target_net, optimizer, device)

            step_iterator.set_postfix({'Total Rewards': int(torch.sum(total_rewards))})

        # Update the target network every TARGET_UPDATE episodes
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DQN agent for Memory Maze environment")
    parser.add_argument('--config', type=str, default=None, help='path to json config')
    parser.add_argument('--output_dir', type=str, default='models/', help='directory to save trained models')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")
    
    #load json config
    with open(args.config, 'r') as f:
        config : dict = json.load(f)
        
    #vectorized environment
    NUM_ENVS = 4
    env = gym.make_vec(config.get('env_name', 'memory_maze:MemoryMaze-9x9-v0'), num_envs=NUM_ENVS, vectorization_mode='async')
    
    num_episodes = config.get('num_episodes', 5)
    LEARNING_RATE = config.get('learning_rate', 1e-4)
    BUFFER_CAPACITY = config.get('buffer_capacity', BUFFER_CAPACITY)
    EPS_START = config.get('eps_start', EPS_START)
    EPS_END = config.get('eps_end', EPS_END)
    EPS_DECAY = config.get('eps_decay', EPS_DECAY)
    BATCH_SIZE = config.get('batch_size', BATCH_SIZE)
    GAMMA = config.get('gamma', GAMMA)
    TARGET_UPDATE = config.get('target_update', TARGET_UPDATE)
    
    # Instantiate policy and target networks via config
    dqn_class, dqn_init_kwargs = load_dqn_class_from_config(config)
    
    # load weights if they exist
    weight_path = config.get('pretrained_weights_path')
    if weight_path and os.path.isfile(weight_path):
        print(f"Loading pretrained weights from {weight_path}")
        with open(weight_path, 'rb') as f:
            pretrained_state_dict = torch.load(f, map_location=device)
            dqn_init_kwargs['pretrained_weights'] = pretrained_state_dict
    
    policy_net = dqn_class(config=config, **dqn_init_kwargs).to(device)
    target_net = dqn_class(config=config, **dqn_init_kwargs).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target network is used for prediction, not training

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayBuffer(BUFFER_CAPACITY)
    
    train_dqn_agent(env, num_episodes, TARGET_UPDATE, policy_net,
                    target_net, optimizer, memory, device)
    
    #save the trained model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{config.get('class', 'dqn_model')}.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")
    
    env.close()