#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt


def state_to_idx(list):
    idx = ([])
    for value in list:
        idx += tuple(value)
    return tuple(idx)


class LegalMovesMixin:
    MOVE_DICT_A = {
        'up': np.asarray([-1, 0]),
        'down': np.asarray([1, 0]),
        'right': np.asarray([0, 1]),
        'left': np.asarray([0, -1]),
        'stay': np.asarray([0, 0])
    }

    MOVE_DICT_B = {
        'up': np.asarray([-1, 0]),
        'down': np.asarray([1, 0]),
        'right': np.asarray([0, 1]),
        'left': np.asarray([0, -1])
    }


class GridTown(LegalMovesMixin):
    A = 1
    B = 2
    P = 3


    STATE_DICT = {
        'running': 0,
        'exited': -1
    }

    def __init__(self, size=(4, 4), init_pos_a=(0, 0), init_pos_b=(1, 1), init_pos_p=(3, 3)):
        self.init_pos_a = init_pos_a
        self.init_pos_b = init_pos_b
        self.init_pos_p = init_pos_p
        self.size = size
        self.grid = None

        self.pos_a = init_pos_a
        self.pos_b = init_pos_b
        self.pos_p = init_pos_p
        self.cumulative_reward = 0
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

    def next_state_action(self, state, is_a=False):
        next_states = []
        next_actions = []

        if is_a:
            move_dict = GridTown.MOVE_DICT_A
        else:
            move_dict = GridTown.MOVE_DICT_B

        for move in move_dict:
            if move_dict[move] is None:
                next_states.append(move_dict[move])
                next_actions.append(move)
            else:
                next_state = state + move_dict[move]
                if self._is_in_bounds_all(next_state):
                    next_states.append(next_state)
                    next_actions.append(move)
        return next_states, next_actions

    def next_state(self, move,  plot_state=False):
        next_a = self.pos_a + GridTown.MOVE_DICT_A[move]

        # Calculate move of police (p)
        next_states_p, _ = self.next_state_action(self.pos_p, is_a=False)
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
    def get_move(self, pos_a, pos_p, actions):
        pass


class RandomPolicy(Policy):
    def get_move(self, pos_a, pos_p, actions):
        return actions[np.random.randint(0, len(actions))]


class QLearner(Policy, LegalMovesMixin):
    def __init__(self, enviro, learning_rate=0.1, discount_factor=0.8):
        self.enviro = enviro
        self.q = np.zeros(
            state_to_idx([
                enviro.size, enviro.size, [len(QLearner.MOVE_DICT_A.keys())]
            ])
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.move_str_to_idx = {k: v for v, k in enumerate(QLearner.MOVE_DICT_A)}

    def get_move(self, pos_a, pos_p, actions):
        moves = []
        for a in actions:
            idx = self.move_str_to_idx[a]
            moves.append(self.q[pos_a[0], pos_a[1], pos_p[0], pos_p[1], idx])
            return actions[np.argmax(moves)]

    def get_epsilon_move(self, pos_a, pos_p, actions):
        """epsilon-greedy policy"""
        rand_val = np.random.uniform()
        if rand_val <= self.epsilon:
            return actions[np.random.randint(0, len(actions))]
        else:
            return self.get_move(pos_a, pos_p, actions)

    def get_random_move(self, actions):
        return actions[np.random.randint(0, len(actions))]

    def learn(self, max_iterations=int(1e7), record_initial_q=False, use_learning_rate=False, plotting_freq=int(1e3)):
        max_iterations = int(max_iterations)
        plotting_freq = int(plotting_freq)

        initial_q = []
        initial_idx = state_to_idx([self.enviro.init_pos_a,
                                    self.enviro.init_pos_p])

        num_q_update = np.zeros(self.q.shape)

        self.enviro.reset()

        for time_step in range(max_iterations):
            # record the Q-function for the initial state of A
            if record_initial_q and time_step % plotting_freq == 0:
                print(time_step)
                initial_q.append(np.copy(self.q[initial_idx]))

            # get the move based on epsilon-greedy policy
            states, moves = self.enviro.next_state_action(
                self.enviro.pos_a, is_a=True)
            move = self.get_random_move(moves)

            # save the old state s
            old_pos_a, old_pos_p = self.enviro.pos_a, self.enviro.pos_p

            # observe a reward from applying `move`
            game_result, reward = self.enviro.next_state(move, plot_state=False)

            # make the update
            move_idx = self.move_str_to_idx[move]
            cur_idx = state_to_idx([self.enviro.pos_a, self.enviro.pos_p])
            old_idx = state_to_idx([old_pos_a, old_pos_p])

            update = reward + self.discount_factor * np.max(self.q[cur_idx]) - self.q[old_idx][move_idx]
            num_q_update[old_idx][move_idx] += 1
            if not use_learning_rate:
                step_size = 1 / (num_q_update[old_idx][move_idx] ** (2/3.))
            else:
                step_size = self.learning_rate
            self.q[old_idx][move_idx] += step_size * update

        if record_initial_q:
            return initial_q


def run(grid, policy):
    while True:
        _, actions = grid.next_state_action(grid.pos_a, is_a=True)
        game_result, reward = grid.next_state(
            policy.get_move(grid.pos_a, grid.pos_p, actions)
        )
        print('State: %s, reward: %d, total reward: %d' %
            (game_result, reward, grid.cumulative_reward))

        if game_result == GridTown.STATE_DICT['exited']:
            break

def plot_initial_q(initial_q):
    initial_q = np.asarray(initial_q)
    plt.plot(initial_q)
    plt.legend(LegalMovesMixin.MOVE_DICT_A.keys())
    plt.show()

if __name__ == '__main__':
    grid = GridTown()
    policy = QLearner(grid)
    initial_q = policy.learn(
        max_iterations=1e7,
        record_initial_q=True,
        use_learning_rate=False,
        plotting_freq=1e3
    )

    plot_initial_q(initial_q)

    run(grid, policy)

