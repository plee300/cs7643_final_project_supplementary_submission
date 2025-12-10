import numpy as np
import random
from image_generation import render_first_person, generate_maze
from math import sin, cos, pi

# MAZE_WIDTH = 11
# MAZE_HEIGHT = 11

# Facing directions
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

DIRECTION_VECTORS = {
    NORTH: (-1, 0),
    EAST:  (0, 1),
    SOUTH: (1, 0),
    WEST:  (0, -1),
}

FACING_TO_ANGLE = {
    EAST: 0,
    NORTH: -pi/2,
    SOUTH:  pi/2,
    WEST: pi
}

class MazeGame:
    """
    Simple maze game environment.
    Agent receives first-person images and can turn/move.
    """

    def __init__(self, maze_width=11, maze_height=11):
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.maze = generate_maze(self.maze_width, self.maze_height)

        self.agent_pos = None
        self.agent_facing = EAST
        self.goal_pos = None
        self.agent_angle = FACING_TO_ANGLE[self.agent_facing]

    # Configuration
    def set_init_position(self, position):
        r, c = position
        assert self.maze[r][c] == 0, "Initial position must be free."
        self.agent_pos = (r, c)

    def set_goal_position(self, position):
        r, c = position
        assert self.maze[r][c] == 0, "Goal position must be free."
        self.goal_pos = (r, c)

    # Environment API
    # def reset(self):
    #     """Reset to initial position."""
    #     self.agent_pos = self.init_position
    #     self.agent_facing = EAST
    #     self.agent_angle = FACING_TO_ANGLE[self.agent_facing]

    def is_winning(self):
        return self.agent_pos == self.goal_pos

    def get_observation(self):
        """Return the first-person rendered image."""
        return render_first_person(self.maze, self.agent_pos[1]+0.5, self.agent_pos[0]+0.5, self.agent_angle)

    # Movement logic
    def can_move_forward(self):
        dr, dc = DIRECTION_VECTORS[self.agent_facing]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.maze_height and 0 <= nc < self.maze_width:
            return self.maze[nr][nc] == 0
        return False

    def step(self, action):
        """
        Action encoding:
            0: move forward
            1: turn left
            2: turn right
        Returns:
            obs, reward, done, info
        """
        reward = -0.01  # small step penalty

        # Turn left
        if action == 1:
            self.agent_facing = (self.agent_facing - 1) % 4

        # Turn right
        elif action == 2:
            self.agent_facing = (self.agent_facing + 1) % 4

        # Move forward
        elif action == 0:
            if self.can_move_forward():
                dr, dc = DIRECTION_VECTORS[self.agent_facing]
                r, c = self.agent_pos
                self.agent_pos = (r + dr, c + dc)

        # Check if win
        done = False
        if self.is_winning():
            reward = 1.0
            done = True

        obs = self.get_observation()

        return obs, reward, done, {}


class Robot:
    """
    Robot that interacts with MazeGame.
    The policy will later be replaced by a neural network / RL.
    """

    def __init__(self, init_position, goal_position):
        self.init_position = init_position
        self.position = init_position
        self.goal_position = goal_position
        self.facing = EAST
        self.memory = []  # store past observations/actions

    def choose_action(self, obs):
        """
        Placeholder for CNN+RL.
        For now: random legal action.
        """
        return random.choice([0, 1, 2])  # forward, left, right

    def update_memory(self, obs, action, reward):
        self.memory.append((obs, action, reward))


if __name__ == "__main__":
    maze_width = 5
    maze_height = 5
    env = MazeGame(maze_width, maze_height)
    print(env.maze)

    # choose random free init + goal
    free_cells = [(r, c) for r in range(maze_height)
                  for c in range(maze_width)
                  if env.maze[r][c] == 0]

    init = random.choice(free_cells)
    goal = random.choice([x for x in free_cells if x != init])

    env.set_init_position(init)
    env.set_goal_position(goal)

    robot = Robot(init, goal)

    obs = env.get_observation()

    # print(env.agent_pos)
    # print(env.agent_facing)
    # obs.show()


    for step in range(200):
        action = robot.choose_action(obs)
        # print(action)
        next_obs, reward, done, info = env.step(action)
        robot.update_memory(obs, action, reward)
        obs = next_obs
        # print(env.agent_pos)
        # print(env.agent_facing)

        if done:
            print("Robot reached goal!")
            break