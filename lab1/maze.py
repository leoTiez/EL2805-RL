#!/usr/bin/python3
import numpy as np


class Maze:

    STATE_DICT = {
        'loosing': -1,
        'running': 0,
        'winning': 1
    }

    MOVE_DICT = {
        'up': np.asarray([-1, 0]),
        'down': np.asarray([1, 0]),
        'right': np.asarray([0, 1]),
        'left': np.asarray([0, -1]),
        'stay': np.asarray([0, 0])
    }

    MOVE_ARRAY = ['up', 'down', 'right', 'left', 'stay']

    A = 1
    B = 2

    def __init__(self, size=(7, 8), init_pos_a=[0, 0], init_pos_b=[6, 5], goal_state=[6, 5], wall_mask=None):
        self.maze_size = size
        self.pos_a = np.asarray(init_pos_a)
        self.pos_b = np.asarray(init_pos_b)
        self.goal_state = np.asarray(goal_state)
        self.wall_mask = wall_mask
        self.maze = self._create_maze()

    def next_state(self, move):
        # Calculate move of a
        next_a = self.pos_a + Maze.MOVE_DICT[move]

        (out_of_left_bound,
         out_of_right_bound,
         out_of_upper_bound,
         out_of_lower_bound) = self._is_in_bounds(next_a)
        is_wall = np.isneginf(self.maze[next_a[0], next_a[1]])

        if out_of_left_bound or out_of_right_bound or out_of_lower_bound or out_of_upper_bound or is_wall:
            next_a = self.pos_a

        # Calculate move of b
        move_b = Maze.MOVE_DICT[np.random.choice(Maze.MOVE_ARRAY)]
        next_b = self.pos_b + move_b

        (out_of_left_bound,
         out_of_right_bound,
         out_of_upper_bound,
         out_of_lower_bound) = self._is_in_bounds(next_b)
        if out_of_left_bound or out_of_right_bound or out_of_lower_bound or out_of_upper_bound:
            next_b = self.pos_b

        # Set move of b
        while np.isneginf(self.maze[next_b[0], next_b[1]]):
            next_b += move_b
            (out_of_left_bound,
             out_of_right_bound,
             out_of_upper_bound,
             out_of_lower_bound) = self._is_in_bounds(next_b)
            if out_of_left_bound or out_of_right_bound or out_of_lower_bound or out_of_upper_bound:
                next_b = self.pos_b
                break

        # Reset previous position
        self.maze[self.pos_b[0], self.pos_b[1]] = 0
        # Set new position via collapsing maze
        self.maze[next_b[0], next_b[1]] = Maze.B

        if np.all(next_b == next_a):
            return Maze.STATE_DICT['loosing']

        # Set new positions
        self.maze[self.pos_a[0], self.pos_a[1]] = 0
        self.maze[next_a[0], next_a[1]] = Maze.A
        self.pos_a = next_a
        self.pos_b = next_b

        if np.all(next_a == self.goal_state):
            return Maze.STATE_DICT['winning']
        else:
            return Maze.STATE_DICT['running']

    def reward(self, state):
        return 1 if np.all(state == self.goal_state) else 0

    def _is_in_bounds(self, state_indices):
        out_of_left_bound = state_indices[1] < 0
        out_of_right_bound = state_indices[1] >= self.maze_size[1]
        out_of_upper_bound = state_indices[0] < 0
        out_of_lower_bound = state_indices[0] >= self.maze_size[0]
        return  out_of_left_bound, out_of_right_bound, out_of_upper_bound, out_of_lower_bound

    def _create_maze(self):
        if self.wall_mask is None:
            self.wall_mask = self._create_default_wall_mask()
        maze = np.zeros(self.maze_size)
        maze[self.wall_mask] = np.NINF
        maze[self.pos_a[0], self.pos_a[1]] = Maze.A
        maze[self.pos_b[0], self.pos_b[1]] = Maze.B

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
    np.random.seed(1)
    maze = Maze()
    maze.next_state('down')


