import argparse
from collections import deque, namedtuple
import os, sys
import random

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import json
import importlib
from tqdm import tqdm

# if 'MUJOCO_GL' not in os.environ:
#     if "linux" in sys.platform:
#         os.environ['MUJOCO_GL'] = 'osmesa' # Software rendering to avoid rendering interference with pygame
#     else:
#         os.environ['MUJOCO_GL'] = 'glfw'  # Windowed rendering
os.environ['MUJOCO_GL'] = 'glfw'  # Windowed rendering

INPUT_CHANNELS = 3
HEIGHT, WIDTH = 64, 64
N_ACTIONS = 6

# 1000 steps per episode per
# https://github.com/jurgisp/memory-maze/tree/main#:~:text=because%20there%20are-,1000%20steps,-(actions)%20in%20a
NUM_STEPS = 1000 
MODULE = 'memory_maze:MemoryMaze-9x9-v0'
NUM_ENVS = 1

# Named tuple for storing transitions in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))            

# Preprocessing: Convert numpy array frame to PyTorch tensor and reshape
def get_screen(observations : np.ndarray, device):
    screen = torch.from_numpy(observations).to(device)

    #reshape from (N, H, W, C) to (N, C, H, W) like torch expects
    screen = screen.permute((0, 3, 1, 2))
    
    #return conversion from uint8 to float32 in range [0.0, 1.0]
    return screen.float() / 255.0

def load_dqn_class_from_config(config):

    module_name = config.get('module')
    class_name = config.get('class')
    if not module_name or not class_name:
        raise ValueError('config.json must specify "module" and "class"')

    module = importlib.import_module(module_name)
    dqn_class = getattr(module, class_name)
    init_kwargs = config.get('init_kwargs', {})
    return dqn_class, init_kwargs


def select_action(states:torch.Tensor, policy_net:nn.Module, ):    
    
    actions = policy_net(states).argmax(dim=1, keepdim=True).squeeze(-1)
                          
    return actions

def evaluate_dqn_agent(envs:gym.vector.AsyncVectorEnv, num_steps: int, policy_net, device, output_dir):        
    # Now that env is vectorized, observation should be (N, H, W, C)
    observations, infos = envs.reset()
    states = get_screen(observations, device) # shape is now (N, C, H, W)
    
    total_rewards = torch.zeros(envs.num_envs, device=device)
    
    
    step_iterator = tqdm(range(num_steps), desc=f"Evaluating model.")
    for t in step_iterator:
        # Select and execute action
        actions = select_action(states, policy_net)
        
        #move the action to cpu and convert to numpy before passing to envs.step()
        observations, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
        
        # Create reward and next_state tensors
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        total_rewards += rewards
        
        next_states = get_screen(observations, device)
        
        if output_dir is not None:
            #save image of first environment
            os.makedirs(output_dir, exist_ok=True)
            img = T.ToPILImage()(next_states[0].cpu())
            img.save(os.path.join(output_dir, f"step_{t:04d}.png"))

        # Move to the next state
        states = next_states
    
    print(f"Evaluation complete. Total rewards per environment: {total_rewards.cpu().numpy()}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DQN agent for Memory Maze environment")
    parser.add_argument('--config', type=str, default=None, help='path to json config')
    parser.add_argument('--output_dir', type=str, default=None, help='directory to save evaluation images')
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
    print("Creating vectorized environment...")
    NUM_ENVS = config.get('num_environments', NUM_ENVS)
    class_name = config.get('class')

    env = gym.make_vec(config.get('env_name', 'memory_maze:MemoryMaze-9x9-v0'), num_envs=1, vectorization_mode='sync')
    
    NUM_STEPS = config.get('num_steps', NUM_STEPS)
    
    print("Loading dqn")
    dqn_class, dqn_init_kwargs = load_dqn_class_from_config(config)
    
    if class_name == "DRQN":
        dqn_init_kwargs['lstm_hidden_size'] = config.get('lstm_hidden_size')
        dqn_init_kwargs['lstm_num_layers'] = config.get('lstm_num_layers')
        dqn_init_kwargs['dropout'] = config.get('dropout')
        dqn_init_kwargs['device'] = device

    # load weights if they exist
    weight_path = config.get('pretrained_weights_path')
    if weight_path and os.path.isfile(weight_path):
        print(f"Loading pretrained weights from {weight_path}")
        with open(weight_path, 'rb') as f:
            pretrained_state_dict = torch.load(f, map_location=device)
            dqn_init_kwargs['pretrained_weights'] = pretrained_state_dict
    
    policy_net = dqn_class(config=config, **dqn_init_kwargs).to(device)
    policy_net.eval()
    
    output_dir = args.output_dir
    
    print("Beginning Evaluation...")
    evaluate_dqn_agent(env, NUM_STEPS, policy_net, device, output_dir)
    
    env.close()