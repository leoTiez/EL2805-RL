#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt


class Maze:

    STATE_DICT = {
        'losing': -1,
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

    def __init__(
            self,
            size=(7, 8),
            init_pos_a=[0, 0],
            init_pos_b=[6, 5],
            goal_state=[6, 5],
            time_horizon=20,
            wall_mask=None
    ):
        self.init_pos_a = np.asarray(init_pos_a)
        self.init_pos_b = np.asarray(init_pos_b)
        self.maze_size = size
        self.pos_a = np.asarray(init_pos_a)
        self.pos_b = np.asarray(init_pos_b)
        self.goal_state = np.asarray(goal_state)
        self.wall_mask = wall_mask
        self.time_horizon = time_horizon
        self.maze, self.policy = self._create_maze()
        
    def reset(self):
        self.pos_a = self.init_pos_a
        self.pos_b = self.init_pos_b

        self.maze, _ = self._create_maze()
        
    def _next_state(self, state, maze, is_a=False):
        if is_a and np.all(state == self.goal_state):
            return [state]

        next_moves = []

        for move in self.MOVE_DICT.values():
            next_state = state + move
            if self._is_in_bounds_all(next_state):
                if is_a and not np.isneginf(maze[tuple(next_state)]):
                    next_moves.append(next_state)
                else:
                    next_moves.append(next_state)
        return next_moves

    def _next_move_a(self, state_a, maze):
        return self._next_state(state_a, maze, is_a=True)

    def _next_move_b(self, state_b, maze):
        return self._next_state(state_b, maze, is_a=False)

    def next_state(self, move):
        """Assumed that `move` is a valid move for player a"""
        # Calculate move of a
        next_a = self.pos_a + move


        # Calculate move of b
        next_moves_b = self._next_move_b(self.pos_b, self.maze)
        pick_move = np.random.randint(0, len(next_moves_b))
        next_b = next_moves_b[pick_move]

        # Reset previous position
        self.maze[self.pos_b[0], self.pos_b[1]] = 0
        # Set new position via collapsing maze
        self.maze[next_b[0], next_b[1]] = Maze.B

        # Set new positions
        self.maze[self.pos_a[0], self.pos_a[1]] = 0
        self.maze[next_a[0], next_a[1]] = Maze.A
        self.pos_a = next_a
        self.pos_b = next_b

        is_at_same_pos = np.all(next_a == next_b)
        is_at_goal = np.all(next_a == self.goal_state)
        is_prev_stay = np.all(move == np.asarray([0, 0]))

        if (not is_at_goal and is_at_same_pos) or \
                (is_at_goal and is_at_same_pos and not is_prev_stay):
            return Maze.STATE_DICT['losing']
        elif np.all(next_a == self.goal_state):
            return Maze.STATE_DICT['winning']
        else:
            return Maze.STATE_DICT['running']

    def run(self):
        game_result = -1
        for i in range(self.time_horizon):
            game_result = self.next_state(
                self.policy[self.pos_a[0], self.pos_a[1],
                            self.pos_b[0], self.pos_b[1]]
            )
        return game_result

    def set_policy(self, policy):
        self.policy = policy

    def reward(self, state):
        return 1 if np.all(state == self.goal_state) else 0

    def plot_state(self):
        plt.matshow(self.maze, cmap=plt.cm.cividis)
        plt.grid()
        plt.show()

    def _is_in_bounds_all(self, state_indices):
        return not any(self._is_in_bounds(state_indices))

    def _is_in_bounds(self, state_indices):
        out_of_left_bound = state_indices[1] < 0
        out_of_right_bound = state_indices[1] >= self.maze_size[1]
        out_of_upper_bound = state_indices[0] < 0
        out_of_lower_bound = state_indices[0] >= self.maze_size[0]
        return  out_of_left_bound, out_of_right_bound, out_of_upper_bound, \
                out_of_lower_bound

    def learn_optimal_policy(self):
        u = np.zeros(self.policy.shape)
        pi = np.zeros(self.policy.shape + (2,), dtype='int64')

        u[self.goal_state, :, :] = self.reward(self.goal_state)
        for t in range(self.time_horizon - 1, 0, -1):
            # check if deep copy
            u_t = np.copy(u)
            for state_ind, _ in np.ndenumerate(u):

                next_rewards = []
                next_moves = []
                next_moves_a = self._next_move_a(state_ind[0:2], self.maze)
                for sa in next_moves_a:
                    next_moves_b = self._next_move_b(state_ind[2:4], self.maze)
                    p = 1.0 / len(next_moves_b)
                    for sb in next_moves_b:
                        next_rewards.append(
                            p * u_t[sa[0], sa[1], sb[0], sb[1]]
                        )
                        next_moves.append(sa)
                u[state_ind] = np.max(next_rewards)
                if t == 1:
                    pi[state_ind] = next_moves[np.argmax(next_rewards)] - np.asarray(state_ind[0:2])

        return pi

    def _create_maze(self):
        if self.wall_mask is None:
            self.wall_mask = self._create_default_wall_mask()
        maze = np.zeros(self.maze_size)
        # Create 4 dim policy matrix to determine move in every state A and B is in
        policy = np.random.choice(Maze.MOVE_ARRAY, (
            self.maze_size[0],
            self.maze_size[1],
            self.maze_size[0],
            self.maze_size[1]
        ))

        maze[self.wall_mask] = np.NINF
        maze[self.pos_a[0], self.pos_a[1]] = Maze.A
        maze[self.pos_b[0], self.pos_b[1]] = Maze.B

        return maze, policy

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
    # np.random.seed(1)
    maze = Maze()
    maze.set_policy(maze.learn_optimal_policy())
    trials = 10000

    wins = 0
    draws = 0
    losses = 0

    for i in range(trials):
        res = maze.run()
        if res == 1:
            wins += 1
        elif res == 0:
            draws += 1
        elif res == -1:
            losses += 1
        maze.reset()
    print('wins %d draws %d losses %d' % (wins, draws, losses))
    print('wins %f draws %f losses %f' % (wins / trials, draws / trials,
                                          losses / trials))


