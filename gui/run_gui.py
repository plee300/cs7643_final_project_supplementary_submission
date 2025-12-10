import os, sys

import argparse
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from PIL import Image

from recording import SaveNpzWrapper

if 'MUJOCO_GL' not in os.environ:
    if "linux" in sys.platform:
        os.environ['MUJOCO_GL'] = 'osmesa' # Software rendering to avoid rendering interference with pygame
    else:
        os.environ['MUJOCO_GL'] = 'glfw'  # Windowed rendering

PANEL_LEFT = 250
PANEL_RIGHT = 250
FOCUS_HACK = False
RECORD_DIR = './log'
K_NONE = tuple()


def get_keymap(env):
    return {
        tuple(): 0,
        (pygame.K_UP, ): 1,
        (pygame.K_LEFT, ): 2,
        (pygame.K_RIGHT, ): 3,
        (pygame.K_UP, pygame.K_LEFT): 4,
        (pygame.K_UP, pygame.K_RIGHT): 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='memory_maze:MemoryMaze-9x9-v0')
    parser.add_argument('--size', type=int, nargs=2, default=(600, 600))
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--random', type=float, default=0.0)
    parser.add_argument('--noreset', action='store_true')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--nonoop', action='store_true', help='Pause instead of noop')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record_mp4', action='store_true')
    parser.add_argument('--record_gif', action='store_true')
    args = parser.parse_args()
    render_size = args.size
    window_size = (render_size[0] + PANEL_LEFT + PANEL_RIGHT, render_size[1])

    print(f'Creating environment: {args.env}')
    env = gym.make(args.env, disable_env_checker=True)

    if isinstance(env.observation_space, spaces.Dict):
        print('Observation space:')
        for k, v in env.observation_space.spaces.items():  # type: ignore
            print(f'{k:>25}: {v}')
    else:
        print(f'Observation space:  {env.observation_space}')
    print(f'Action space:  {env.action_space}')

    if args.record:
        env = SaveNpzWrapper(
            env,
            RECORD_DIR,
            video_format='mp4' if args.record_mp4 else 'gif' if args.record_gif else None,
            video_fps=args.fps * 2)

    keymap = get_keymap(env)

    steps = 0
    return_ = 0.0
    episode = 0
    # Handle both Gym and Gymnasium reset signatures
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out

    pygame.init()
    start_fullscreen = args.fullscreen or FOCUS_HACK
    screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN if start_fullscreen else 0)
    if FOCUS_HACK and not args.fullscreen:
        # Hack: for some reason app window doesn't get focus when launching, so
        # we launch it as full screen and then exit full screen.
        pygame.display.toggle_fullscreen()
    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont('Mono', 16)
    fontsmall = pygame.freetype.SysFont('Mono', 12)

    running = True
    paused = False
    speedup = False

    while running:

        # Rendering

        screen.fill((64, 64, 64))

        # Render image observation
        if isinstance(obs, dict):
            assert 'image' in obs, 'Expecting dictionary observation with obs["image"]'
            image = obs['image']  # type: ignore
        else:
            #assert isinstance(obs, np.ndarray) and len(obs.shape) == 3, 'Expecting image observation'
            image = obs
        # Robustly handle different image formats/shapes:
        # - grayscale (H,W)
        # - channels-last (H,W,C)
        # - channels-first (C,H,W)
        img = np.array(image)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            # If channels-first (C,H,W) where C==1 or 3
            if img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
                img = img.transpose(1, 2, 0)
            # If channels-last already (H,W,C) and C==1, convert to 3 channels
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
        else:
            raise ValueError(f'Unsupported image ndim: {img.ndim}')

        pil = Image.fromarray(img)
        pil = pil.resize(render_size, resample=Image.NEAREST)
        img = np.array(pil)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        surface = pygame.surfarray.make_surface(img.transpose((1, 0, 2)))
        screen.blit(surface, (PANEL_LEFT, 0))

        # Render statistics
        lines = obs_to_text(obs, env, steps, return_)
        y = 5
        for line in lines:
            text_surface, rect = font.render(line, (255, 255, 255))
            screen.blit(text_surface, (16, y))
            y += font.get_sized_height() + 2  # type: ignore

        # Render keymap help
        lines = keymap_to_text(keymap)
        y = 5
        for line in lines:
            text_surface, rect = fontsmall.render(line, (255, 255, 255))
            screen.blit(text_surface, (render_size[0] + PANEL_LEFT + 16, y))
            y += fontsmall.get_sized_height() + 2  # type: ignore

        pygame.display.flip()
        clock.tick(args.fps if not speedup else 0)

        # Keyboard input

        pygame.event.pump()
        keys_down = defaultdict(bool)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Close
                running = False
            if event.type == pygame.KEYDOWN:
                keys_down[event.key] = True
        keys_hold = pygame.key.get_pressed()

        # Action keys
        action = keymap[K_NONE]  # noop, if no keys pressed
        for keys, act in keymap.items():
            if all(keys_hold[key] or keys_down[key] for key in keys):
                # The last keymap entry which has all keys pressed wins
                action = act

        # Special keys
        force_reset = False
        speedup = False
        if keys_down[pygame.K_ESCAPE]:  # Quit
            running = False
        if keys_down[pygame.K_SPACE]:  # Pause
            paused = not paused
        else:
            if action != keymap[K_NONE]:
                paused = False  # unpause on action press
        if keys_down[pygame.K_BACKSPACE]:  # Force reset
            force_reset = True
        if keys_hold[pygame.K_TAB]:
            speedup = True

        if paused:
            continue
        if action == keymap[K_NONE] and args.nonoop and not force_reset:
            continue

        # Environment step

        if args.random:
            if np.random.random() < args.random:
                action = env.action_space.sample()

        # Support both Gym (obs, reward, done, info) and Gymnasium
        # (obs, reward, terminated, truncated, info) signatures.
        t = env.step(action)  # type: ignore
        if len(t) == 4:
            obs, reward, done, info = t
        else:
            obs, reward, terminated, truncated, info = t
            done = bool(terminated or truncated)
        # print({k: v for k, v in obs.items() if k != 'image'})
        steps += 1
        return_ += float(reward)

        # Episode end

        if reward:
            print(f'reward: {reward}')
        if done or force_reset:
            print(f'Episode done - length: {steps}  return: {return_}')
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out
            steps = 0
            return_ = 0.0
            episode += 1
            if done and args.record:
                # If recording, require relaunch for next episode
                running = False

    pygame.quit()


def obs_to_text(obs, env, steps, return_):
    kvs = []
    kvs.append(('## Stats ##', ''))
    kvs.append(('', ''))
    kvs.append(('step', steps))
    kvs.append(('return', return_))
    lines = [f'{k:<15} {v:>5}' for k, v in kvs]
    return lines


def keymap_to_text(keymap, verbose=False):
    kvs = []
    kvs.append(('## Commands ##', ''))
    kvs.append(('', ''))

    # mapped actions
    kvs.append(('forward', 'up arrow'))
    kvs.append(('left', 'left arrow'))
    kvs.append(('right', 'right arrow'))

    # special actions
    kvs.append(('', ''))
    kvs.append(('reset', 'backspace'))
    kvs.append(('pause', 'space'))
    kvs.append(('speed up', 'tab'))
    kvs.append(('quit', 'esc'))

    lines = [f'{k:<15} {v}' for k, v in kvs]
    return lines


if __name__ == '__main__':
    main()
