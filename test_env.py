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
from PIL import Image

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glfw'  # Windowed rendering

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
CRITERION_FUNCTION = nn.SmoothL1Loss()
NUM_ENVS = 4
NUM_EPS = 5
SAVE_IMAGES = False
SAVE_DIR = './results/images/'
# 1000 steps per episode per
# https://github.com/jurgisp/memory-maze/tree/main#:~:text=because%20there%20are-,1000%20steps,-(actions)%20in%20a
NUM_STEPS = 1000 
MODULE = 'memory_maze:MemoryMaze-9x9-v0'

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


def select_action(states:torch.Tensor, eps_threshold:float, policy_net:nn.Module, env:gym.vector.AsyncVectorEnv, device:torch.device):
    sample = torch.rand(states.size(0), device=device)
    
    # Decay epsilon from EPS_START to EPS_END over EPS_DECAY steps    
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

def optimize_model(memory, policy_net, optimizer, criterion, device):
    # taken from https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Normally, we'd want to mask out non-final states, but in there is no
    # terminal state in memory maze, so all states are non-final.
    next_states = torch.cat(batch.next_state, dim=0).to(device) #(N, C, H, W)
    state_batch = torch.cat(batch.state, dim=0).to(device) #(N, C, H, W)
    action_batch = torch.stack(batch.action, dim=0).to(device) #(N, 1)
    reward_batch = torch.stack(batch.reward, dim=0).to(device) #(N, 1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    result = policy_net(state_batch)
    state_action_values = result.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    with torch.no_grad():
        next_state_values = target_net(next_states).max(1).values.unsqueeze(-1) # (N, 1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train_dqn_agent(envs:gym.vector.AsyncVectorEnv, num_episodes:int, num_steps: int, policy_net, optimizer, criterion, memory:ReplayBuffer, class_name:str, save:str, device, eval=False):
    reward_array = np.zeros((num_episodes//NUM_ENVS + 1,NUM_ENVS)) # vector to save training rewards over time
    eval_array = []
    for i_episode in range(1, num_episodes+1, envs.num_envs):
        
        # Now that env is vectorized, observation should be (N, H, W, C)
        observations, infos = envs.reset()
        policy_net.reset_memory()

        states = get_screen(observations, device) # shape is now (N, C, H, W)
        
        total_rewards = torch.zeros(envs.num_envs, device=device)
        
        
        step_iterator = tqdm(range(num_steps), desc=f"Episode {i_episode}/{num_episodes}")
        for t in step_iterator:
            if eval:
                eps_threshold = 0
            else:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * i_episode / EPS_DECAY)
            # Select and execute action
            actions = select_action(states, eps_threshold, policy_net, envs, device)
            
            #move the action to cpu and convert to numpy before passing to envs.step()
            observations, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
            
            if SAVE_IMAGES:
                img = observations[0]
                img = Image.fromarray(img)
                img.save(os.path.join(SAVE_DIR, f"step_{t:04d}.png"))
            
            # Create reward and next_state tensors
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            total_rewards += rewards
            
            next_states = get_screen(observations, device)
            
            # Store the transitions in memory unbatched
            # these are computed as batch size NUM_ENVS,
            # but we will collect to BATCH_SIZE for gradient descent
            for i in range(envs.num_envs):
                state = states[i].unsqueeze(0)
                action = actions[i].unsqueeze(0)
                next_state = next_states[i].unsqueeze(0)
                reward = rewards[i].unsqueeze(0)
                memory.push(state, action, next_state, reward)

            # Move to the next state
            states = next_states

            # Perform one optimization step
            optimize_model(memory, policy_net, optimizer, criterion, device)

            step_iterator.set_postfix({'Total Rewards': int(torch.sum(total_rewards))})
        reward_array[i_episode//NUM_ENVS,:] = total_rewards.cpu().numpy() # add rewards for all num_envs to array for visualization in report
        
        if eval:
            return reward_array[i_episode//NUM_ENVS,:]

        if int(i_episode/envs.num_envs) %5 == 0: #evaluate and save results so far every 5 batches
            print("Evaluating:")
            eval_rewards = train_dqn_agent(envs, 1, num_steps, policy_net, optimizer, criterion, memory, class_name, save, device, eval=True)
            eval_array.append((i_episode, np.average(eval_rewards)))
            try:
                np.save(f"results/{class_name}_eval_data.npy", np.array(eval_array))
            except Exception as e:
                print("could not save to numpy array")
                print(e)

        # Update the target network after every batch of episodes
        target_net.load_state_dict(policy_net.state_dict())
    try:
        if save == 'y':
            np.save(f"results/{class_name}_train_data.npy", reward_array)
        else:
            np.save(f"results/{class_name}_eval_data.npy", reward_array)
        print("Rewards saved to results folder.")

    except Exception as e:
        print("could not save to numpy array")
        print(e)

if __name__ == "__main__":
    
    #prevents colab from hanging
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="DQN agent for Memory Maze environment")
    parser.add_argument('--config', type=str, default=None, help='path to json config')
    parser.add_argument('--output_dir', type=str, default='models/', help='directory to save trained models')
    parser.add_argument('--num_envs', type=int, default=None, help='number of parallel environments')
    parser.add_argument('--sync', action='store_true', help='use synchronous vectorized environment')
    parser.add_argument('--load_weights', action='store_true', help='load weights from previously trained model')
    parser.add_argument('--eval_mode', action='store_true', help='run in evaluation mode')
    parser.add_argument('--eval_dir', type=str, default='./results/images', help='directory to save evaluation results')
    args = parser.parse_args()
    
    SAVE_DIR = args.eval_dir
    
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
    if args.num_envs is not None:
        NUM_ENVS = args.num_envs
    else:
        NUM_ENVS = config.get('num_environments', NUM_ENVS)      
    if args.sync:
        vec_mode = 'sync'
        print("Using synchronous vectorized environment.")
    else:
        vec_mode = 'async'
        print("Using asynchronous vectorized environment.")
    env = gym.make_vec(config.get('env_name', 'memory_maze:MemoryMaze-9x9-v0'), num_envs=NUM_ENVS, vectorization_mode=vec_mode)
    
    class_name = config.get('class', "")
    NUM_EPS = config.get('num_episodes', NUM_EPS)
    NUM_STEPS = config.get('num_steps', NUM_STEPS)
    LEARNING_RATE = config.get('learning_rate', 1e-4)
    BUFFER_CAPACITY = config.get('buffer_capacity', BUFFER_CAPACITY)
    EPS_START = config.get('eps_start', EPS_START)
    EPS_END = config.get('eps_end', EPS_END)
    EPS_DECAY = config.get('eps_decay', EPS_DECAY)
    BATCH_SIZE = config.get('batch_size', BATCH_SIZE)
    GAMMA = config.get('gamma', GAMMA)
    CRITERION_FUNCTION = config.get('criterion', CRITERION_FUNCTION)
    
    if args.eval_mode:
        NUM_EPS = 1  # only run one episode for evaluation
        EPS_END = 0.0
        EPS_START = 0.0
        SAVE_IMAGES = True

    if CRITERION_FUNCTION == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else: # default function
        criterion = nn.SmoothL1Loss() 
    
    # Instantiate policy and target networks via config
    
    print("Loading dqn")
    dqn_class, dqn_init_kwargs = load_dqn_class_from_config(config)
    
    if class_name == "DRQN":
        dqn_init_kwargs['lstm_hidden_size'] = config.get('lstm_hidden_size')
        dqn_init_kwargs['lstm_num_layers'] = config.get('lstm_num_layers')
        dqn_init_kwargs['dropout'] = config.get('dropout')
        dqn_init_kwargs['device'] = device
    
    # load weights if they exist
    weight_path = config.get('pretrained_weights_path')
    if weight_path and os.path.isfile(weight_path) and args.load_weights:
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
    
    if args.eval_mode:
        save = 'n'
        print("Beginning Evaluation")
        num_eps_for_mode = NUM_ENVS
    else:
        save = 'y'
        print("Beginning Training")
        num_eps_for_mode = NUM_EPS
    try:
        train_dqn_agent(env, num_eps_for_mode, NUM_STEPS, policy_net, optimizer, criterion, memory, class_name, save, device)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        save = input("Save the trained model? (y/n): ").strip().lower()
    finally:
        if save == 'y':
            #save the trained model
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f"{config.get('class', 'dqn_model')}.pth")
            torch.save(policy_net.state_dict(), model_path)
            print(f"Trained model saved to {model_path}")
    
        env.close()