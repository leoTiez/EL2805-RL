#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt


class Maze:
    STATE_DICT = {
        'losing': -1,
        'running': 0,
        'winning': 1
    }

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

    A = 1
    B = 2

    def __init__(
            self,
            size=(7, 8),
            init_pos_a=[0, 0],
            init_pos_b=[6, 5],
            goal_state=[6, 5],
            wall_mask=None,
            verbosity=1
    ):
        self.init_pos_a = np.asarray(init_pos_a)
        self.init_pos_b = np.asarray(init_pos_b)
        self.maze_size = size
        self.pos_a = np.asarray(init_pos_a)
        self.pos_b = np.asarray(init_pos_b)
        self.goal_state = np.asarray(goal_state)
        self.wall_mask = wall_mask
        self.policy = None
        self.verbosity = verbosity
        self.maze = self._create_maze()

    def reset(self):
        self.pos_a = self.init_pos_a
        self.pos_b = self.init_pos_b

        self.maze = self._create_maze()

    def _next_state(self, state, maze, is_a=False):
        if is_a and np.all(state == self.goal_state):
            return [state]

        next_states = []

        if is_a:
            move_dict = Maze.MOVE_DICT_A
        else:
            move_dict = Maze.MOVE_DICT_B

        for move in move_dict.values():
            next_state = state + move
            if self._is_in_bounds_all(next_state):
                if is_a:
                    if not np.isneginf(maze[tuple(next_state)]):
                        next_states.append(next_state)
                else:
                    next_states.append(next_state)
        return next_states

    def _next_move_a(self, state_a, maze):
        return self._next_state(state_a, maze, is_a=True)

    def _next_move_b(self, state_b, maze):
        return self._next_state(state_b, maze, is_a=False)

    def next_state(self, move, plot_state=True):
        """Assumed that `move` is a valid move for player a"""
        # Calculate move of a
        next_a = self.pos_a + move

        # Calculate move of b
        next_moves_b = self._next_move_b(self.pos_b, self.maze)
        pick_move = np.random.randint(0, len(next_moves_b))
        next_b = next_moves_b[pick_move]

        # Reset previous position
        self.maze[self.pos_b[0], self.pos_b[1]] = 0
        self.maze[self.pos_a[0], self.pos_a[1]] = 0
        self.maze[self.wall_mask] = np.NINF
        # Set new position via collapsing maze
        self.maze[next_b[0], next_b[1]] = Maze.B

        # Set new positions
        self.maze[next_a[0], next_a[1]] = Maze.A
        self.pos_a = next_a
        self.pos_b = next_b

        is_at_same_pos = np.all(next_a == next_b)
        is_at_goal = np.all(next_a == self.goal_state)
        is_prev_stay = np.all(move == np.asarray([0, 0]))

        if plot_state:
            self.plot_state()

        if (not is_at_goal and is_at_same_pos) or \
                (is_at_goal and is_at_same_pos and not is_prev_stay):
            return Maze.STATE_DICT['losing']
        elif np.all(next_a == self.goal_state):
            return Maze.STATE_DICT['winning']
        else:
            return Maze.STATE_DICT['running']

    def set_policy(self, policy):
        self.policy = policy

    def reward(self, state_a, state_b):
        if state_b is None:
            return Maze.STATE_DICT['winning']
        has_eaten = np.all(state_a == state_b)
        in_goal = np.all(state_a == self.goal_state)
        if has_eaten and not in_goal:
            return Maze.STATE_DICT['losing']
        elif in_goal:
            return Maze.STATE_DICT['winning']
        else:
            return Maze.STATE_DICT['running']

    def plot_state(self):
        plt.matshow(self.maze, cmap=plt.cm.cividis)
        plt.grid()
        plt.show()

    @staticmethod
    def _stopping_criterion(value_diff, precision, surviving_p):
        return value_diff > precision * (1 - surviving_p) / surviving_p

    def _compute_value(self, state_ind, value, surviving_p=1):
        next_rewards = []
        next_moves_a = self._next_move_a(state_ind[0:2], self.maze)
        next_moves_b = self._next_move_b(state_ind[2:4], self.maze)
        p = 1.0 / len(next_moves_b)
        for sa in next_moves_a:
            summed_reward_over_states = 0
            for sb in next_moves_b:
                summed_reward_over_states += p * value[sa[0], sa[1], sb[0], sb[1]]

            summed_reward_over_states *= surviving_p
            summed_reward_over_states += self.reward(state_ind[:2], state_ind[2:])
            next_rewards.append(summed_reward_over_states)

        new_value = np.max(next_rewards)
        return new_value, next_moves_a[np.argmax(next_rewards)] - np.asarray(state_ind[0:2])

    def _is_in_bounds_all(self, state_indices):
        return not any(self._is_in_bounds(state_indices))

    def _is_in_bounds(self, state_indices):
        out_of_left_bound = state_indices[1] < 0
        out_of_right_bound = state_indices[1] >= self.maze_size[1]
        out_of_upper_bound = state_indices[0] < 0
        out_of_lower_bound = state_indices[0] >= self.maze_size[0]
        return out_of_left_bound, out_of_right_bound, out_of_upper_bound, out_of_lower_bound

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


class MazeFiniteHorizon(Maze):
    def __init__(
            self,
            size=(7, 8),
            init_pos_a=[0, 0],
            init_pos_b=[6, 5],
            goal_state=[6, 5],
            time_horizon=20,
            wall_mask=None
    ):
        self.time_horizon = time_horizon
        super(MazeFiniteHorizon, self).__init__(
            size=size,
            init_pos_a=init_pos_a,
            init_pos_b=init_pos_b,
            goal_state=goal_state,
            wall_mask=wall_mask
        )

    def learn(self):
        u = np.zeros(self.maze_size + self.maze_size)
        pi = np.zeros(self.maze_size + self.maze_size + (2,), dtype='int64')

        u[self.goal_state[0], self.goal_state[1], :, :] = self.reward(self.goal_state, None)

        for t in range(self.time_horizon - 1, 0, -1):
            u_t = np.copy(u)
            for state_ind in np.ndindex(u.shape):
                u_t[state_ind], best_move_a = self._compute_value(state_ind, u)
                if t == 1:
                    pi[state_ind] = best_move_a
            u = u_t

        return pi

    def run(self, plot_state=True):
        if self.policy is None:
            raise ValueError('Policy is none and should be set before')

        game_result = -1
        for i in range(self.time_horizon):
            game_result = self.next_state(
                self.policy[self.pos_a[0], self.pos_a[1],
                            self.pos_b[0], self.pos_b[1]],
                plot_state=plot_state
            )
            if game_result == Maze.STATE_DICT['losing'] or game_result == Maze.STATE_DICT['winning']:
                break
        return game_result


class MazeInfiniteHorizon(Maze):
    def learn(self, surviving_p, precision=0.1):
        value = np.zeros(self.maze_size + self.maze_size)
        pi = np.zeros(self.maze_size + self.maze_size + (2,), dtype='int64')

        value_difference = np.Inf
        while Maze._stopping_criterion(value_difference, precision, surviving_p):
            value_temp = np.copy(value)
            for state_ind in np.ndindex(value.shape):
                value_temp[state_ind], pi[state_ind] = self._compute_value(state_ind, value, surviving_p=surviving_p)
            value_difference = np.linalg.norm(value - value_temp)
            if self.verbosity > 0:
                print(value_difference)
            value = value_temp

        return pi

    def run(self, plot_state=True):
        if self.policy is None:
            raise ValueError('Policy is none and should be set before')

        while True:
            game_result = self.next_state(
                self.policy[self.pos_a[0], self.pos_a[1],
                            self.pos_b[0], self.pos_b[1]],
                plot_state=plot_state
            )
            if game_result == Maze.STATE_DICT['losing'] or game_result == Maze.STATE_DICT['winning']:
                break
        return game_result


def main_comparison():
    maze_finite = MazeFiniteHorizon(time_horizon=20)
    maze_infinte = MazeInfiniteHorizon()
    print('Start Finite Policy Learning')
    finite_policy = maze_finite.learn()
    print('Start Infinite Policy Learning\n')
    infinite_policy = maze_infinte.learn(29/30., precision=1)
    print(infinite_policy.size - np.count_nonzero(finite_policy == infinite_policy))
    assert np.all(finite_policy == infinite_policy)


def main(is_finite=True, is_plotting=True):
    if is_finite:
        maze = MazeFiniteHorizon(time_horizon=20)
        policy = maze.learn()
    else:
        maze = MazeInfiniteHorizon()
        policy = maze.learn(29/30., precision=1)

    maze.set_policy(policy)
    trials = 10000

    wins = 0
    draws = 0
    losses = 0

    for i in range(trials):
        res = maze.run(plot_state=is_plotting)
        if res == Maze.STATE_DICT['winning']:
            wins += 1
        elif res == Maze.STATE_DICT['running']:
            draws += 1
        elif res == Maze.STATE_DICT['losing']:
            losses += 1
        maze.reset()
    print('wins %d draws %d losses %d' % (wins, draws, losses))
    print('wins %f draws %f losses %f' % (wins / trials, draws / trials,
                                          losses / trials))


def test_maze_finite(is_plotting=True):
    maze_size = (2, 2)
    wall_mask = np.zeros(maze_size, dtype='bool')
    init_pos_a = [0, 0]
    init_pos_b = [1, 1]
    goal_state = [1, 1]
    time_horizon = 4
    maze = MazeFiniteHorizon(
        size=maze_size,
        init_pos_a=init_pos_a,
        init_pos_b=init_pos_b,
        goal_state=goal_state,
        wall_mask=wall_mask,
        time_horizon=time_horizon
    )
    maze.set_policy(maze.learn())

    trials = 10000
    wins = 0
    draws = 0
    losses = 0

    for i in range(trials):
        res = maze.run(plot_state=is_plotting)
        if res == Maze.STATE_DICT['winning']:
            wins += 1
        elif res == Maze.STATE_DICT['running']:
            draws += 1
        elif res == Maze.STATE_DICT['losing']:
            losses += 1
        maze.reset()

    print('wins %d draws %d losses %d' % (wins, draws, losses))
    print('wins %f draws %f losses %f' % (wins / trials, draws / trials,
                                          losses / trials))


if __name__ == '__main__':
    # test_maze_finite(is_plotting=False)
    main_comparison()
    main(is_finite=True, is_plotting=False)
    main(is_finite=False, is_plotting=False)

