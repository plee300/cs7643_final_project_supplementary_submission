# # # # main_evaluate.py
# # # import os
# # # import yaml
# # # import argparse
# # # import torch
# # # import gym
# # # import memory_maze

# # # from models.agent import ActorCriticRNN
# # # from models.dqn_agent import DQNAgent
# # # from environments.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

# # # def find_latest_model(model_dir):
# # #     files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
# # #     if not files:
# # #         raise ValueError(f"No model checkpoints found in {model_dir}")
# # #     steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
# # #     latest_step = max(steps)
# # #     latest_model_file = f"agent_step_{latest_step}.pth"
# # #     return os.path.join(model_dir, latest_model_file)

# # # def create_eval_env(env_id, config, video_folder=None):
# # #     env = gym.make(env_id)
# # #     if video_folder:
# # #         print(f"Recording videos to: {video_folder}")
# # #         os.makedirs(video_folder, exist_ok=True)
# # #         env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)

# # #     wrapper_config = config.get('env', {}).get('wrappers', {})
# # #     if wrapper_config.get('grayscale', {}).get('enabled', False):
# # #         env = GrayScaleObservation(env)
# # #     if wrapper_config.get('resize', {}).get('enabled', False):
# # #         shape = wrapper_config['resize']['shape']
# # #         env = ResizeObservation(env, shape=tuple(shape))
# # #     if wrapper_config.get('frame_stack', {}).get('enabled', False):
# # #         k = wrapper_config['frame_stack']['k']
# # #         env = FrameStack(env, k)
# # #     return env

# # # def main():
# # #     parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
# # #     parser.add_argument("--exp-dir", type=str, required=True, help="Path to the experiment directory")
# # #     args = parser.parse_args()

# # #     with open('configs/config.yaml', 'r') as f:
# # #         config = yaml.safe_load(f)

# # #     model_dir = os.path.join(args.exp_dir, "models")
# # #     model_path = find_latest_model(model_dir)
# # #     print(f"Loading latest model: {model_path}")

# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# # #     algo_name = os.path.basename(args.exp_dir).split('_')[0].upper()
# # #     env_name = config['env']['name']
    
# # #     obs_space_shape = (
# # #         config['env']['wrappers']['resize']['shape'][0],
# # #         config['env']['wrappers']['resize']['shape'][1],
# # #         config['env']['wrappers']['frame_stack']['k']
# # #     )
# # #     num_actions = gym.make(env_name).action_space.n

# # #     if algo_name == "PPO":
# # #         agent = ActorCriticRNN(obs_space_shape, num_actions, config['model']['recurrent_hidden_dim'])
# # #     elif algo_name == "DQN":
# # #         use_dueling = config['dqn'].get('dueling', False)
# # #         agent = DQNAgent(obs_space_shape, num_actions, use_dueling=use_dueling)
# # #     else:
# # #         raise ValueError(f"Unknown algorithm '{algo_name}' inferred from directory name.")

# # #     agent.load_state_dict(torch.load(model_path, map_location=device))
# # #     agent.to(device)
# # #     agent.eval()

# # #     video_folder = None
# # #     if config['evaluation']['record_video']:
# # #         video_folder = os.path.join(args.exp_dir, config['evaluation']['video_folder'])
    
# # #     env = create_eval_env(env_name, config, video_folder)

# # #     total_rewards = []
# # #     for i in range(config['evaluation']['num_episodes']):
# # #         obs = env.reset()
# # #         done = False
# # #         episode_reward = 0
        
# # #         if algo_name == "PPO":
# # #             hidden_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)
# # #             cell_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)

# # #         while not done:
# # #             with torch.no_grad():
# # #                 obs_tensor = torch.ByteTensor(obs).unsqueeze(0).to(device)
# # #                 if algo_name == "PPO":
# # #                     action_dist, _, (hidden_state, cell_state) = agent(obs_tensor, hidden_state, cell_state)
# # #                     action = action_dist.mode()
# # #                 else: # DQN
# # #                     q_values = agent(obs_tensor)
# # #                     action = q_values.max(1)[1]

# # #             obs, reward, done, info = env.step(action.cpu().item())
# # #             episode_reward += reward
        
# # #         total_rewards.append(episode_reward)
# # #         print(f"Episode {i+1}/{config['evaluation']['num_episodes']}: Total Reward: {episode_reward:.2f}")

# # #     env.close()
# # #     print("\n--- Evaluation Summary ---")
# # #     print(f"Average reward over {len(total_rewards)} episodes: {sum(total_rewards)/len(total_rewards):.2f}")
# # #     if video_folder:
# # #         print(f"Videos saved in: {video_folder}")

# # # if __name__ == '__main__':
# # #     main()

# # # main_evaluate.py
# # import os
# # import yaml
# # import argparse
# # import torch
# # import gym
# # import memory_maze

# # from models.agent import ActorCriticRNN
# # from models.dqn_agent import DQNAgent
# # from environments.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

# # def find_latest_model(model_dir):
# #     files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
# #     if not files:
# #         raise ValueError(f"No model checkpoints found in {model_dir}")
# #     steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
# #     latest_step = max(steps)
# #     latest_model_file = f"agent_step_{latest_step}.pth"
# #     return os.path.join(model_dir, latest_model_file)
# # from environments.wrappers import ResizeObservation, GrayScaleObservation, FrameStack, GymResetCompatibilityWrapper

# # def create_eval_env(env_id, config, is_vec_env=False, n_envs=1):
# #     def env_creator():
# #         env = gym.make(env_id)
        
# #         # ===================================================================
# #         #           APPLY THE COMPATIBILITY WRAPPER FIRST
# #         # ===================================================================
# #         env = GymResetCompatibilityWrapper(env)
# #         # ===================================================================

# #         wrapper_config = config.get('env', {}).get('wrappers', {})
# #         if wrapper_config.get('grayscale', {}).get('enabled', False):
# #             env = GrayScaleObservation(env)
# #         if wrapper_config.get('resize', {}).get('enabled', False):
# #             shape = wrapper_config['resize']['shape']
# #             env = ResizeObservation(env, shape=tuple(shape))
# #         if wrapper_config.get('frame_stack', {}).get('enabled', False):
# #             k = wrapper_config['frame_stack']['k']
# #             env = FrameStack(env, k)
# #         return env

# #     if is_vec_env:
# #         return make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_fn=env_creator)
# #     else:
# #         return env_creator()
# # def main():
# #     parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
# #     parser.add_argument("--exp-dir", type=str, required=True, help="Path to the experiment directory")
# #     args = parser.parse_args()

# #     with open('configs/config.yaml', 'r') as f:
# #         config = yaml.safe_load(f)

# #     model_dir = os.path.join(args.exp_dir, "models")
# #     model_path = find_latest_model(model_dir)
# #     print(f"Loading latest model: {model_path}")

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# #     algo_name = os.path.basename(args.exp_dir).split('_')[0].upper()
# #     env_name = config['env']['name']
    
# #     obs_space_shape = (
# #         config['env']['wrappers']['resize']['shape'][0],
# #         config['env']['wrappers']['resize']['shape'][1],
# #         config['env']['wrappers']['frame_stack']['k']
# #     )
# #     num_actions = gym.make(env_name).action_space.n

# #     if algo_name == "PPO":
# #         agent = ActorCriticRNN(obs_space_shape, num_actions, config['model']['recurrent_hidden_dim'])
# #     elif algo_name == "DQN":
# #         use_dueling = config['dqn'].get('dueling', False)
# #         agent = DQNAgent(obs_space_shape, num_actions, use_dueling=use_dueling)
# #     else:
# #         raise ValueError(f"Unknown algorithm '{algo_name}' inferred from directory name.")

# #     agent.load_state_dict(torch.load(model_path, map_location=device))
# #     agent.to(device)
# #     agent.eval()

# #     video_folder = None
# #     if config['evaluation']['record_video']:
# #         video_folder = os.path.join(args.exp_dir, config['evaluation']['video_folder'])
    
# #     env = create_eval_env(env_name, config, video_folder)

# #     total_rewards = []
# #     for i in range(config['evaluation']['num_episodes']):
# #         obs, _ = env.reset() # CORRECTED: Unpack tuple
# #         done = False
# #         episode_reward = 0
        
# #         if algo_name == "PPO":
# #             hidden_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)
# #             cell_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)

# #         while not done:
# #             with torch.no_grad():
# #                 obs_tensor = torch.ByteTensor(obs).unsqueeze(0).to(device)
# #                 if algo_name == "PPO":
# #                     action_dist, _, (hidden_state, cell_state) = agent(obs_tensor, hidden_state, cell_state)
# #                     action = action_dist.mode()
# #                 else: # DQN
# #                     q_values = agent(obs_tensor)
# #                     action = q_values.max(1)[1]

# #             obs, reward, done, info = env.step(action.cpu().item())
# #             episode_reward += reward
        
# #         total_rewards.append(episode_reward)
# #         print(f"Episode {i+1}/{config['evaluation']['num_episodes']}: Total Reward: {episode_reward:.2f}")

# #     env.close()
# #     print("\n--- Evaluation Summary ---")
# #     print(f"Average reward over {len(total_rewards)} episodes: {sum(total_rewards)/len(total_rewards):.2f}")
# #     if video_folder:
# #         print(f"Videos saved in: {video_folder}")

# # if __name__ == '__main__':
# #     main()
# # main_evaluate.py
# import os
# import yaml
# import argparse
# import torch
# import gym
# import memory_maze

# from models.agent import ActorCriticRNN
# from models.dqn_agent import DQNAgent
# from environments.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

# def find_latest_model(model_dir):
#     """Find the most recent model checkpoint in the directory."""
#     files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
#     if not files:
#         raise ValueError(f"No model checkpoints found in {model_dir}")
#     steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
#     latest_step = max(steps)
#     latest_model_file = f"agent_step_{latest_step}.pth"
#     return os.path.join(model_dir, latest_model_file)

# def create_eval_env(env_id, config, video_folder=None):
#     """
#     Create an evaluation environment with wrappers.
#     No compatibility wrappers needed with gym 0.23.1.
#     """
#     env = gym.make(env_id)
    
#     # Optional: Record videos
#     if video_folder:
#         print(f"Recording videos to: {video_folder}")
#         os.makedirs(video_folder, exist_ok=True)
#         env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
    
#     # Apply observation wrappers
#     wrapper_config = config.get('env', {}).get('wrappers', {})
#     if wrapper_config.get('grayscale', {}).get('enabled', False):
#         env = GrayScaleObservation(env)
#     if wrapper_config.get('resize', {}).get('enabled', False):
#         shape = wrapper_config['resize']['shape']
#         env = ResizeObservation(env, shape=tuple(shape))
#     if wrapper_config.get('frame_stack', {}).get('enabled', False):
#         k = wrapper_config['frame_stack']['k']
#         env = FrameStack(env, k)
    
#     return env

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
#     parser.add_argument("--exp-dir", type=str, required=True, 
#                         help="Path to the experiment directory")
#     parser.add_argument("--render", action="store_true", 
#                         help="Render the environment during evaluation")
#     args = parser.parse_args()

#     # Load configuration
#     with open('configs/config.yaml', 'r') as f:
#         config = yaml.safe_load(f)

#     # Find the latest model checkpoint
#     model_dir = os.path.join(args.exp_dir, "models")
#     model_path = find_latest_model(model_dir)
#     print(f"Loading latest model: {model_path}")

#     # Setup device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Determine algorithm from directory name
#     algo_name = os.path.basename(args.exp_dir).split('_')[0].upper()
#     env_name = config['env']['name']
    
#     # Get observation space shape
#     obs_space_shape = (
#         config['env']['wrappers']['resize']['shape'][0],
#         config['env']['wrappers']['resize']['shape'][1],
#         config['env']['wrappers']['frame_stack']['k']
#     )
    
#     # Get number of actions
#     temp_env = gym.make(env_name)
#     num_actions = temp_env.action_space.n
#     temp_env.close()

#     # Create the agent
#     if algo_name == "PPO":
#         agent = ActorCriticRNN(
#             obs_space_shape, 
#             num_actions, 
#             config['model']['recurrent_hidden_dim']
#         )
#     elif algo_name == "DQN":
#         use_dueling = config['dqn'].get('dueling', False)
#         agent = DQNAgent(obs_space_shape, num_actions, use_dueling=use_dueling)
#     else:
#         raise ValueError(f"Unknown algorithm '{algo_name}' inferred from directory name.")

#     # Load the trained weights
#     agent.load_state_dict(torch.load(model_path, map_location=device))
#     agent.to(device)
#     agent.eval()
#     print(f"Loaded {algo_name} agent")

#     # Setup video recording if enabled
#     video_folder = None
#     if config['evaluation']['record_video']:
#         video_folder = os.path.join(args.exp_dir, config['evaluation']['video_folder'])
    
#     # Create evaluation environment
#     env = create_eval_env(env_name, config, video_folder)

#     # Run evaluation episodes
#     total_rewards = []
#     episode_lengths = []
#     num_episodes = config['evaluation']['num_episodes']
    
#     print(f"\nEvaluating for {num_episodes} episodes...")
#     print("=" * 60)
    
#     for i in range(num_episodes):
#         obs = env.reset()  # gym 0.23.1 returns just obs
#         done = False
#         episode_reward = 0
#         episode_length = 0
        
#         # Initialize recurrent states for PPO
#         if algo_name == "PPO":
#             hidden_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)
#             cell_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)

#         while not done:
#             if args.render:
#                 env.render()
            
#             with torch.no_grad():
#                 obs_tensor = torch.ByteTensor(obs).unsqueeze(0).to(device)
                
#                 if algo_name == "PPO":
#                     action_dist, _, (hidden_state, cell_state) = agent(
#                         obs_tensor, hidden_state, cell_state
#                     )
#                     action = action_dist.mode()
#                 else:  # DQN
#                     q_values = agent(obs_tensor)
#                     action = q_values.max(1)[1]

#             obs, reward, done, info = env.step(action.cpu().item())
#             episode_reward += reward
#             episode_length += 1
        
#         total_rewards.append(episode_reward)
#         episode_lengths.append(episode_length)
#         print(f"Episode {i+1}/{num_episodes}: Reward: {episode_reward:.2f}, Length: {episode_length}")

#     env.close()
    
#     # Print evaluation summary
#     print("\n" + "=" * 60)
#     print("--- Evaluation Summary ---")
#     print("=" * 60)
#     print(f"Number of episodes: {len(total_rewards)}")
#     print(f"Mean reward: {sum(total_rewards)/len(total_rewards):.2f}")
#     print(f"Std reward: {torch.tensor(total_rewards).std().item():.2f}")
#     print(f"Min reward: {min(total_rewards):.2f}")
#     print(f"Max reward: {max(total_rewards):.2f}")
#     print(f"Mean episode length: {sum(episode_lengths)/len(episode_lengths):.1f}")
    
#     if video_folder:
#         print(f"\nVideos saved in: {video_folder}")
    
#     print("=" * 60)

# if __name__ == '__main__':
#     main()
# main_evaluate.py
import os
import yaml
import argparse
import torch
import gym
import memory_maze

from models.agent import ActorCriticRNN
from models.dqn_agent import DQNAgent
from environments.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

def find_latest_model(model_dir):
    """Find the most recent model checkpoint in the directory."""
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not files:
        raise ValueError(f"No model checkpoints found in {model_dir}")
    steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
    latest_step = max(steps)
    latest_model_file = f"agent_step_{latest_step}.pth"
    return os.path.join(model_dir, latest_model_file)

def create_eval_env(env_id, config, video_folder=None):
    """
    Create an evaluation environment with wrappers.
    No compatibility wrappers needed with gym 0.23.1.
    """
    env = gym.make(env_id)
    
    # Apply observation wrappers BEFORE video recording
    # This ensures we record the processed observations
    wrapper_config = config.get('env', {}).get('wrappers', {})
    if wrapper_config.get('grayscale', {}).get('enabled', False):
        env = GrayScaleObservation(env)
    if wrapper_config.get('resize', {}).get('enabled', False):
        shape = wrapper_config['resize']['shape']
        env = ResizeObservation(env, shape=tuple(shape))
    if wrapper_config.get('frame_stack', {}).get('enabled', False):
        k = wrapper_config['frame_stack']['k']
        env = FrameStack(env, k)
    
    # Apply video recording AFTER other wrappers
    if video_folder:
        print(f"Recording videos to: {video_folder}")
        os.makedirs(video_folder, exist_ok=True)
        try:
            # Try to use RecordVideo wrapper
            env = gym.wrappers.RecordVideo(
                env, 
                video_folder, 
                episode_trigger=lambda episode_id: True,  # Record all episodes
                name_prefix="eval"
            )
        except Exception as e:
            print(f"Warning: Could not initialize video recording: {e}")
            print("Continuing without video recording...")
    
    return env

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
    parser.add_argument("--exp-dir", type=str, required=True, 
                        help="Path to the experiment directory")
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment during evaluation")
    args = parser.parse_args()

    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Find the latest model checkpoint
    model_dir = os.path.join(args.exp_dir, "models")
    model_path = find_latest_model(model_dir)
    print(f"Loading latest model: {model_path}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine algorithm from directory name
    algo_name = os.path.basename(args.exp_dir).split('_')[0].upper()
    env_name = config['env']['name']
    
    # Get observation space shape
    obs_space_shape = (
        config['env']['wrappers']['resize']['shape'][0],
        config['env']['wrappers']['resize']['shape'][1],
        config['env']['wrappers']['frame_stack']['k']
    )
    
    # Get number of actions
    temp_env = gym.make(env_name)
    num_actions = temp_env.action_space.n
    temp_env.close()

    # Create the agent
    if algo_name == "PPO":
        agent = ActorCriticRNN(
            obs_space_shape, 
            num_actions, 
            config['model']['recurrent_hidden_dim']
        )
    elif algo_name == "DQN":
        use_dueling = config['dqn'].get('dueling', False)
        agent = DQNAgent(obs_space_shape, num_actions, use_dueling=use_dueling)
    else:
        raise ValueError(f"Unknown algorithm '{algo_name}' inferred from directory name.")

    # Load the trained weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    agent.eval()
    print(f"Loaded {algo_name} agent")

    # Setup video recording if enabled
    video_folder = None
    if config['evaluation']['record_video']:
        video_folder = os.path.join(args.exp_dir, config['evaluation']['video_folder'])
    
    # Create evaluation environment
    env = create_eval_env(env_name, config, video_folder)

    # Run evaluation episodes
    total_rewards = []
    episode_lengths = []
    num_episodes = config['evaluation']['num_episodes']
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 60)
    
    for i in range(num_episodes):
        obs = env.reset()  # gym 0.23.1 returns just obs
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Initialize recurrent states for PPO
        if algo_name == "PPO":
            hidden_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)
            cell_state = torch.zeros(1, config['model']['recurrent_hidden_dim']).to(device)

        while not done:
            if args.render:
                env.render()
            
            with torch.no_grad():
                obs_tensor = torch.ByteTensor(obs).unsqueeze(0).to(device)
                
                if algo_name == "PPO":
                    action_dist, _, (hidden_state, cell_state) = agent(
                        obs_tensor, hidden_state, cell_state
                    )
                    action = action_dist.mode()
                else:  # DQN
                    q_values = agent(obs_tensor)
                    action = q_values.max(1)[1]

            obs, reward, done, info = env.step(action.cpu().item())
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {i+1}/{num_episodes}: Reward: {episode_reward:.2f}, Length: {episode_length}")

    env.close()
    
    # Print evaluation summary
    print("\n" + "=" * 60)
    print("--- Evaluation Summary ---")
    print("=" * 60)
    print(f"Number of episodes: {len(total_rewards)}")
    print(f"Mean reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Std reward: {torch.tensor(total_rewards).std().item():.2f}")
    print(f"Min reward: {min(total_rewards):.2f}")
    print(f"Max reward: {max(total_rewards):.2f}")
    print(f"Mean episode length: {sum(episode_lengths)/len(episode_lengths):.1f}")
    
    if video_folder:
        print(f"\nVideos saved in: {video_folder}")
    
    print("=" * 60)

if __name__ == '__main__':
    main()