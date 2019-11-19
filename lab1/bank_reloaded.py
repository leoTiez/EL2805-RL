#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt


class GridTown:
    A = 1
    B = 2
    P = 3

    MOVE_DICT_A = {
        'up': np.asarray([-1, 0]),
        'down': np.asarray([1, 0]),
        'right': np.asarray([0, 1]),
        'left': np.asarray([0, -1]),
        'stay': np.asarray([0, 0]),
        'exit': None
    }

    MOVE_DICT_B = {
        'up': np.asarray([-1, 0]),
        'down': np.asarray([1, 0]),
        'right': np.asarray([0, 1]),
        'left': np.asarray([0, -1])
    }

    STATE_DICT = {
        'running': 0,
        'exited': -1
    }

    def __init__(self, size=(4, 4), init_pos_a=(0, 0), init_pos_b=(1, 1), init_pos_p=(3, 3),
                 policy=None):
        self.init_pos_a = init_pos_a
        self.init_pos_b = init_pos_b
        self.init_pos_p = init_pos_p
        self.size = size
        self.grid = None

        self.pos_a = init_pos_a
        self.pos_b = init_pos_b
        self.pos_p = init_pos_p
        self.cumulative_reward = 0
        self.policy = policy
        self.reset()

    def reset(self):
        self.grid = self._create_init_grid(self.size)
        self.pos_a = self.init_pos_a
        self.pos_b = self.init_pos_b
        self.pos_p = self.init_pos_p
        self.cumulative_reward = 0

    def reward(self, pos_a,  pos_p):
        if np.all(pos_a == pos_p):
            return -10
        elif np.all(pos_a == self.pos_b):
            return 1
        else:
            return 0

    def _is_in_bounds_all(self, state_indices):
        return not any(self._is_in_bounds(state_indices))

    def _is_in_bounds(self, state_indices):
        grid_size = self.grid.shape
        out_of_left_bound = state_indices[1] < 0
        out_of_right_bound = state_indices[1] >= grid_size[1]
        out_of_upper_bound = state_indices[0] < 0
        out_of_lower_bound = state_indices[0] >= grid_size[0]
        return out_of_left_bound, out_of_right_bound, out_of_upper_bound, out_of_lower_bound

    def _next_state_action(self, state, is_a=False):
        next_states = []
        next_actions = []

        if is_a:
            move_dict = GridTown.MOVE_DICT_A
        else:
            move_dict = GridTown.MOVE_DICT_B

        for move in move_dict.values():
            if move is None:
                next_states.append(None)
                next_actions.append(None)
            else:
                next_state = state + move
                if self._is_in_bounds_all(next_state):
                    next_states.append(next_state)
                    next_actions.append(move)
        return next_states, next_actions

    def run(self, plot_state=True):
        if self.policy is None:
            raise ValueError('Policy is none and should be set before')

        while True:
            game_result = self.next_state(plot_state=plot_state)
            if game_result == GridTown.STATE_DICT['exited']:
                break
        return game_result

    def next_state(self,  plot_state=False):
        if self.policy is None:
            raise ValueError('Policy is none and should be set before')

        # Calculate move of a
        _, next_actions = self._next_state_action(self.pos_a, is_a=True)
        move = self.policy.get_move(self.pos_a, self.pos_p, next_actions)

        if move is None:
            return GridTown.STATE_DICT['exited'], 0
        else:
            next_a = self.pos_a + move

        # Calculate move of police (p)
        next_states_p, _ = self._next_state_action(self.pos_p, is_a=False)
        pick_move = np.random.randint(0, len(next_states_p))
        next_p = next_states_p[pick_move]

        # Reset previous position
        self.grid[self.pos_p[0], self.pos_p[1]] = 0
        self.grid[self.pos_a[0], self.pos_a[1]] = 0

        # Set new positions
        self.grid[next_p[0], next_p[1]] = GridTown.P
        self.grid[next_a[0], next_a[1]] = GridTown.A
        self.grid[self.init_pos_b[0], self.init_pos_b[1]] = GridTown.B

        self.pos_a = next_a
        self.pos_p = next_p

        current_reward = self.reward(self.pos_a, self.pos_p)
        self.cumulative_reward += current_reward

        if plot_state:
            self.plot_state()

        return GridTown.STATE_DICT['running'], current_reward

    def _create_init_grid(self, size=(4, 4)):
        grid = np.zeros(size)
        grid[self.init_pos_a] = GridTown.A
        grid[self.init_pos_b] = GridTown.B
        grid[self.init_pos_p] = GridTown.P

        return grid


    def plot_state(self):
        plt.matshow(self.grid, cmap=plt.cm.cividis)
        plt.grid()
        plt.show()


class Policy:
    def get_move(self, pos_a, pos_b, actions):
        pass


class RandomPolicy(Policy):
    def get_move(self, pos_a, pos_p, actions):
        return actions[np.random.randint(0, len(actions))]

class QLearner(Policy):
    class QPolicy(Policy):
        def __init__(self, q):
            self.q = q

        def get_move(self, pos_a, pos_p, actions):
            np.self.q[pos_a[0], pos_a[1], pos_p[0], pos_p[1], ]

    def __init__(self, learning_rate=0.1, epsilon=0.1, episodes=100):
        self.q = None
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.episodes = episodes

    def export_q(self):
        return QLearner.QPolicy(self.q)

    def get_move(self, pos_a, pos_b, actions):
        """epsilon-greedy policy"""
        pass

    def learn(self, enviro):
        for episode in range(self.episodes):
            while True:
                old_pos_a, old_pos_p = enviro.pos_a, enviro.pos_p
                game_result, reward = self.enviro.next_state(plot_state=False)
                self.q[old_pos_a, old_pos_p, act] += self.learning_rate * (
                    reward + np.max(self.q[enviro.pos_a, enviro.pos_p, :] -
                                    self.q[old_pos_a, old_pos_p, :]))
                if game_result == GridTown.STATE_DICT['exited']:
                    break
            enviro.reset()

if __name__ == '__main__':
    policy = RandomPolicy()
    enviro = GridTown(policy=policy)
    enviro.run(plot_state=True)


