import numpy as np
import random
from math import sin, cos, pi
from PIL import Image
import os
import pickle


# maze generation (recursive backtracking)

def generate_maze(width, height):
    maze = np.ones((height, width), dtype=np.int32)  # 1 = wall, 0 = free

    def carve(x, y):
        maze[y, x] = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                if maze[ny, nx] == 1:
                    maze[y + dy, x + dx] = 0
                    carve(nx, ny)

    carve(1, 1)
    return maze


# first-person ray-cast render

def render_first_person(maze, px, py, angle,
                        fov=100, width=200, height=150, max_depth=100):

    img = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = height, width
    fov_rad = np.deg2rad(fov)

    sky_color = np.array([20, 40, 80], dtype=np.uint8)
    floor_color = np.array([50, 50, 50], dtype=np.uint8)

    for col in range(w):
        # Ray direction
        ray_angle = angle - fov_rad/2 + (col / w) * fov_rad
        sin_a, cos_a = sin(ray_angle), cos(ray_angle)

        dist = 0
        hit = False

        # Ray marching
        while dist < max_depth:
            dist += 0.02
            rx = px + cos_a * dist
            ry = py + sin_a * dist

            if not (0 <= int(rx) < maze.shape[1] and
                    0 <= int(ry) < maze.shape[0]):
                break

            if maze[int(ry), int(rx)] == 1:
                hit = True
                break

        # Sky + floor
        img[:h//2, col] = sky_color
        img[h//2:, col] = floor_color

        # Walls
        if hit:
            wall_height = min(h, int(70 / (dist + 0.0001)))
            top = max(0, h//2 - wall_height//2)
            bottom = min(h, h//2 + wall_height//2)

            shade = int(200 / (1 + dist * 0.3))
            shade = max(0, min(shade, 255))
            wall_color = np.array([shade]*3, dtype=np.uint8)

            img[top:bottom, col] = wall_color

    return Image.fromarray(img)


# explore the entire movement logic

DIRECTIONS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0)
}

LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}

FACING_TO_ANGLE = {
    "E": 0,
    "N": -pi/2,
    "S":  pi/2,
    "W": pi
}


def can_move(maze, x, y, direction):
    dx, dy = DIRECTIONS[direction]
    nx, ny = x + dx, y + dy

    if 0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0]:
        return maze[ny, nx] == 0
    return False


# data collection and splitting

def save_sample(folder, index, sample):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"sample_{index}.pkl"), "wb") as f:
        pickle.dump(sample, f)


def collect_and_split_online(layout, train_ratio=0.8, val_ratio=0.1):
    idx = 0
    for r in range(layout.shape[0]):
        for c in range(layout.shape[1]):
            if layout[r, c] != 0:
                continue

            for facing in ["N", "S", "E", "W"]:

                px = c + 0.5
                py = r + 0.5
                angle = FACING_TO_ANGLE[facing]

                img = render_first_person(layout, px, py, angle)
                filename = f"sample_{idx}.png"
                img_path = os.path.join(f"dataset/images", filename)
                img.save(img_path)

                # Compute movement tags
                front_ok = int(can_move(layout, c, r, facing))
                left_ok  = int(can_move(layout, c, r, LEFT_OF[facing]))
                right_ok = int(can_move(layout, c, r, RIGHT_OF[facing]))
                tags = [front_ok, left_ok, right_ok]

                sample = {
                    "image": np.array(img),  # store NumPy array
                    "tags": tags,
                    "pos": (r, c),
                    "facing": facing
                }

                # Random split selection
                p = random.random()
                if p < train_ratio:
                    split = "train"
                elif p < train_ratio + val_ratio:
                    split = "val"
                else:
                    split = "test"

                save_sample(f"dataset/{split}", idx, sample)
                idx += 1


# test code
# To generate files

# maze = np.array([
#     [1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1],
#     [1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1]
# ])
random.seed(1)
maze = generate_maze(11, 11)
# print(maze)
collect_and_split_online(maze)

# px = 1.5
# py = 1.5
# angle = FACING_TO_ANGLE["S"]
# img = render_first_person(maze, px, py, angle)
# img.show()