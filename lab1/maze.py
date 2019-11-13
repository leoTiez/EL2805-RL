#!/usr/bin/python3
import numpy as np


class Maze:

    MOVE_DICT = {
        'up': (1, 0),
        'down': (-1, 0),
        'right': (0, 1),
        'left': (0, -1),
        'stay': (0, 0)
    }

    MOVE_ARRAY = ['up', 'down', 'right', 'left', 'stay']

    def __init__(self, size=(7, 8), init_pos_a=(0, 0), init_pos_b=(6,5), wall_mask=None):
        self.maze_size = size
        self.pos_a = init_pos_a
        self.pos_b = init_pos_b
        self.maze = self._create_maze(wall_mask)

    def _create_maze(self, wall_mask):
        if wall_mask is None:
            wall_mask = self._create_default_wall_mask()
        maze = np.zeros(self.maze_size)
        maze[wall_mask] = np.NINF
        maze[self.pos_a] = 1
        maze[self.pos_b] = 2

        return maze

    def _create_default_wall_mask(self):
        wall_mask = np.full(self.maze_size, False, dtype='bool')
        # Vertical
        wall_mask[0:4, 2] = True
        wall_mask[1:4, 5] = True
        wall_mask[5:, 4] = True
        # Horizontal
        wall_mask[2, 5:] = True
        wall_mask[5, 1:7] = True

        return wall_mask


if __name__ == '__main__':
    maze = Maze()

