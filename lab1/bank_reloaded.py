#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from os import makedirs


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

    def __init__(self, size=(4, 4), init_pos_a=(0, 0), init_pos_b=(1, 1),
                 init_pos_p=(3, 3), save_name=None):
        self.init_pos_a = init_pos_a
        self.init_pos_b = init_pos_b
        self.init_pos_p = init_pos_p
        self.size = size
        self.grid = None

        self.pos_a = init_pos_a
        self.pos_b = init_pos_b
        self.pos_p = init_pos_p
        self.cumulative_reward = 0
        self.save_name = save_name
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
            next_state = state + move_dict[move]
            if self._is_in_bounds_all(next_state):
                next_states.append(next_state)
                next_actions.append(move)
        return next_states, next_actions

    def next_state(self, move,  plot_state=False, move_num=1):
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

        if plot_state and self.save_name:
            self.plot_state(self.save_name + '/' + str(move_num) + '.png')
        elif plot_state:
            self.plot_state()

        return GridTown.STATE_DICT['running'], current_reward

    def _create_init_grid(self, size=(4, 4)):
        grid = np.zeros(size)
        grid[self.init_pos_a] = GridTown.A
        grid[self.init_pos_b] = GridTown.B
        grid[self.init_pos_p] = GridTown.P

        return grid

    def plot_state(self, save_name=None):
        plt.matshow(self.grid, cmap=plt.cm.cividis)
        plt.grid()
        if save_name is not None:
            makedirs(self.save_name, exist_ok=True)
            plt.savefig(save_name)
        else:
            plt.show()


class Policy:
    def get_move(self, pos_a, pos_p, actions):
        pass


class RandomPolicy(Policy):
    def get_move(self, pos_a, pos_p, actions):
        return actions[np.random.randint(0, len(actions))]


class QLearner(Policy, LegalMovesMixin):
    def __init__(self, enviro, learning_rate=0.1, discount_factor=0.8,
                 epsilon=0.1):
        self.enviro = enviro
        self.q = np.zeros(
            state_to_idx([
                enviro.size, enviro.size, [len(QLearner.MOVE_DICT_A.keys())]
            ])
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.move_str_to_idx = {k: v for v, k in enumerate(QLearner.MOVE_DICT_A)}

    def get_move(self, pos_a, pos_p, actions):
        moves = []
        for a in actions:
            idx = self.move_str_to_idx[a]
            moves.append(self.q[pos_a[0], pos_a[1], pos_p[0], pos_p[1], idx])
        return actions[np.argmax(moves)]

    def get_random_move(self, actions):
        return actions[np.random.randint(0, len(actions))]

    def get_epsilon_move(self, pos_a, pos_p, actions):
        """epsilon-greedy policy"""
        rand_val = np.random.uniform()
        if rand_val <= self.epsilon:
            return actions[np.random.randint(0, len(actions))]
        else:
            return self.get_move(pos_a, pos_p, actions)

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
            _, reward = self.enviro.next_state(move, plot_state=False,
                                               move_num=time_step)

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


class SARSALearner(QLearner):
    def __init__(self, enviro, learning_rate=0.1, discount_factor=0.8,
                 epsilon=0.1):
        super(SARSALearner, self).__init__(enviro, learning_rate,
                                           discount_factor, epsilon)

    def learn(self, max_iterations=int(1e7), record_initial_q=False,
              use_learning_rate=False, plotting_freq=int(1e3),
              use_adapting_epsilon=False):
        max_iterations = int(max_iterations)
        plotting_freq = int(plotting_freq)

        initial_q = []
        initial_idx = state_to_idx([self.enviro.init_pos_a,
                                    self.enviro.init_pos_p])

        num_q_update = np.zeros(self.q.shape)

        self.enviro.reset()

        _, moves = self.enviro.next_state_action(self.enviro.pos_a, is_a=True)

        move = self.get_epsilon_move(self.enviro.pos_a, self.enviro.pos_p, moves)

        for time_step in range(max_iterations):
            # record the Q-function for the initial state of A
            if record_initial_q and time_step % plotting_freq == 0:
                print(time_step)
                initial_q.append(np.copy(self.q[initial_idx]))

            # save the old state s
            old_pos_a, old_pos_p = self.enviro.pos_a, self.enviro.pos_p
            old_idx = state_to_idx([old_pos_a, old_pos_p])
            move_idx = self.move_str_to_idx[move]

            # observe a reward from applying `move`
            game_result, reward = self.enviro.next_state(move,
                                                         plot_state=False,
                                                         move_num=time_step)

            if use_adapting_epsilon:
                self.epsilon = 1.0 / (num_q_update[old_idx][move_idx] + 1)

            # get the move based on epsilon-greedy policy
            _, moves = self.enviro.next_state_action(
                self.enviro.pos_a, is_a=True)

            move_prime = self.get_epsilon_move(self.enviro.pos_a,
                                               self.enviro.pos_p, moves)

            # make the update
            move_prime_idx = self.move_str_to_idx[move_prime]
            cur_idx = state_to_idx([self.enviro.pos_a, self.enviro.pos_p])

            update = reward + self.discount_factor * self.q[cur_idx][move_prime_idx] - \
                     self.q[old_idx][move_idx]
            num_q_update[old_idx][move_idx] += 1

            if not use_learning_rate:
                step_size = 1 / (num_q_update[old_idx][move_idx] ** (2 / 3.))
            else:
                step_size = self.learning_rate

            self.q[old_idx][move_idx] += step_size * update
            move = move_prime

        if record_initial_q:
            return initial_q


def run(grid, policy, max_iters=None, plot_state=False):
    grid.reset()
    iters = 0
    while True:
        if max_iters is not None and iters > max_iters:
            return
        _, actions = grid.next_state_action(grid.pos_a, is_a=True)
        game_result, reward = grid.next_state(
            policy.get_move(grid.pos_a, grid.pos_p, actions),
            move_num=iters, plot_state=plot_state
        )
        print('State: %s, reward: %d, total reward: %d' %
            (game_result, reward, grid.cumulative_reward))

        if game_result == GridTown.STATE_DICT['exited']:
            break
        iters += 1


def plot_initial_q(initial_q, save_name=None):
    initial_q = np.asarray(initial_q)
    plt.figure()
    plt.plot(initial_q)
    plt.legend(LegalMovesMixin.MOVE_DICT_A.keys())
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def main_sarsa(use_adaptive_epsilon=True):
    if not use_adaptive_epsilon:
        for epsilon in [0.1, 0.3, 0.7]:
            grid = GridTown()
            policy = SARSALearner(grid, epsilon=epsilon)
            initial_q = policy.learn(
                max_iterations=int(1e7),
                record_initial_q=True,
                use_learning_rate=True,
                plotting_freq=int(1e4),
                use_adapting_epsilon=False
            )

            grid.save_name = 'sarsa_epsilon_%.2f' % epsilon

            run(grid, policy, max_iters=20, plot_state=True)
            print('Cumulative reward for epsilon = %f is %d' %
                  (epsilon, grid.cumulative_reward))

            plot_initial_q(initial_q, grid.save_name +
                           '/initial_q_sarsa_epsilon.png')

    else:
        grid = GridTown()
        policy = SARSALearner(grid)
        initial_q = policy.learn(
            max_iterations=int(1e7),
            record_initial_q=True,
            use_learning_rate=True,
            plotting_freq=int(1e4),
            use_adapting_epsilon=True
        )

        grid.save_name = 'sarsa_adaptive_epsilon2'

        run(grid, policy, max_iters=20, plot_state=True)
        print('Cumulative reward for epsilon = 1/t is %d' %
              grid.cumulative_reward)
        plot_initial_q(initial_q, grid.save_name + '/initial_q_sarsa_adaptive.png')


def main_q():
    grid = GridTown()
    policy = QLearner(grid)
    initial_q = policy.learn(
        max_iterations=int(1e7),
        record_initial_q=True,
        use_learning_rate=False,
        plotting_freq=int(1e4)
    )

    grid.save_name = 'q'

    run(grid, policy, max_iters=20, plot_state=True)
    plot_initial_q(initial_q, grid.save_name + '/initial_q_qlearning.png')


if __name__ == '__main__':
    # main_q()
    main_sarsa()

