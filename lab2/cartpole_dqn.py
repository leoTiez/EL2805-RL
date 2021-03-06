import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers

EPISODES = 2000 #Maximum number of episodes

#DQN Agent for the Cartpole
#Q function approximation with NN, experience replay, and target network
class DQNAgent:
    #Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_size, action_size, target_update_frequency=1, arch=[16], discount_factor=0.95, learning_rate=0.0005, mem_size=5000):
        self.check_solve = True	#If True, stop if you satisfy solution confition
        self.render = False        #If you want to see Cartpole learning, then change to True

        #Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

       # Modify here

        #Set hyper parameters for the DQN. Do not adjust those labeled as Fixed.
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 0.02 #Fixed
        self.batch_size = 32 #Fixed
        self.memory_size = mem_size
        self.train_start = 1000 #Fixed
        self.target_update_frequency = target_update_frequency

        #Number of test states for Q value plots
        self.test_state_no = 10000

        #Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        self.arch = arch
        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        #Initialize target network
        self.update_target_model()

    #Approximate Q function using Neural Network
    #State is the input and the Q Values are the output.
###############################################################################
###############################################################################
        #Edit the Neural Network model here
        #Tip: Consult https://keras.io/getting-started/sequential-model-guide/
    def build_model(self):
        model = Sequential()
        for num_units in self.arch:
            model.add(Dense(num_units, input_dim=self.state_size, activation='relu',
                            kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
###############################################################################
###############################################################################

    #After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #Get action from model using epsilon-greedy policy
    def get_action(self, state):
###############################################################################
###############################################################################
        #Insert your e-greedy policy code here
        #Tip 1: Use the random package to generate a random action.
        #Tip 2: Use keras.model.predict() to compute Q-values from the state.
        action = random.randrange(self.action_size)
        if random.random() < self.epsilon:
            return action
        else:
            Q_actions = self.model.predict(state)
            return np.argmax(Q_actions, axis=1)[0]

    ###############################################################################
###############################################################################
    #Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #Add sample to the end of the list

    #Sample <s,a,r,s'> from replay memory
    def train_model(self):
        if len(self.memory) < self.train_start: #Do not train if not enough memory
            return
        batch_size = min(self.batch_size, len(self.memory)) #Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.memory, batch_size) #Uniformly sample the memory buffer
        #Preallocate network and target network input matrices.
        update_input = np.zeros((batch_size, self.state_size)) #batch_size by state_size two-dimensional array (not matrix!)
        update_target = np.zeros((batch_size, self.state_size)) #Same as above, but used for the target network
        action, reward, done = [], [], [] #Empty arrays that will grow dynamically

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0] #Allocate s(i) to the network input array from iteration i in the batch
            action.append(mini_batch[i][1]) #Store a(i)
            reward.append(mini_batch[i][2]) #Store r(i)
            update_target[i] = mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
            done.append(mini_batch[i][4])  #Store done(i)

        target = self.model.predict(update_input) #Generate target values for training the inner loop network using the network model
        target_val = self.target_model.predict(update_target) #Generate the target values for training the outer loop target network

        #Q Learning: get maximum Q value at s' from target network
###############################################################################
###############################################################################
        #Insert your Q-learning code here
        #Tip 1: Observe that the Q-values are stored in the variable target
        #Tip 2: What is the Q-value of the action taken at the last state of the episode?
        for i in range(self.batch_size): #For every batch
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = self.discount_factor * np.max(target_val[i]) + reward[i]
###############################################################################
###############################################################################

        #Train the inner loop network
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        return
    #Plots the score per episode as well as the maximum q value per episode, averaged over precollected states.
    def plot_data(self, episodes, scores, max_q_mean, dir_name, arch=None):

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if arch is None:
            subname = ''
        else:
            subname = str(arch)
        pylab.figure(0)
        pylab.plot(episodes, max_q_mean, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Average Q Value")
        pylab.savefig("%s/qvalues-%s.png" % (dir_name, subname))

        pylab.figure(1)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Score")
        pylab.savefig("%s/scores-%s.png" % (dir_name,  subname))

def plot_data_multiple(episodes, scores, max_q_mean, solved_times, dir_name, names):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    pylab.figure(0)
    pylab.clf()
    for i in range(len(episodes)):
        pylab.plot(episodes[i], max_q_mean[i], label=names[i])
    actually_solved_times = [time for time in solved_times if time != -1]
    solved_values = [max_q_mean[i][solved_times[i]] for i in range(len(solved_times)) if solved_times[i] != -1]
    pylab.plot(actually_solved_times, solved_values, 'kx', label='solved')
    pylab.xlabel("Episodes")
    pylab.ylabel("Average Q Value")
    pylab.legend(names)
    pylab.savefig("%s/qvalues.png" % dir_name)

    pylab.figure(1)
    pylab.clf()
    for i in range(len(episodes)):
        pylab.plot(episodes[i], scores[i])
    pylab.xlabel("Episodes")
    pylab.ylabel("Score")
    pylab.legend()
    pylab.savefig("%s/scores.png" % dir_name)

###############################################################################
###############################################################################

def simulate(agent, times):
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory='sims/trained', force=True)

    for run in range(times):
        state = env.reset()
        state_size = env.observation_space.shape[0]
        state = np.reshape(state, [1, state_size])  # Reshape state so that to a 1 by state_size two-dimensional array ie. [x_1,x_2] to [[x_1,x_2]]

        done = False
        while not done:
            #env.render()  # Show cartpole animation

            # Get action for the current state and go one step in environment
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            state = np.reshape(state, [1, state_size])  # Reshape next_state similarly to state
    env.close()


def train(arch, discount_factor=0.95, learning_rate=0.0005, mem_size=5000, update_freq=1):
    solved = -1
    #For CartPole-v0, maximum episode length is 200
    env = gym.make('CartPole-v0') #Generate Cartpole-v0 environment object from the gym library
    #env = wrappers.Monitor(env, directory='sims/training', force=True)

    #Get state and action sizes from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    #Create agent, see the DQNAgent __init__ method for details
    agent = DQNAgent(state_size, action_size, arch=arch, discount_factor=discount_factor, learning_rate=learning_rate, mem_size=mem_size, target_update_frequency=update_freq)

    #Collect test states for plotting Q values using uniform random policy
    test_states = np.zeros((agent.test_state_no, state_size))
    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))

    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action = random.randrange(action_size)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state

    scores, episodes = [], [] #Create dynamically growing score and episode counters
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset() #Initialize/reset the environment
        state = np.reshape(state, [1, state_size]) #Reshape state so that to a 1 by state_size two-dimensional array ie. [x_1,x_2] to [[x_1,x_2]]
        #Compute Q values for plotting
        tmp = agent.model.predict(test_states)
        max_q[e][:] = np.max(tmp, axis=1)
        max_q_mean[e] = np.mean(max_q[e][:])

        while not done:
            if agent.render:
                env.render() #Show cartpole animation

            #Get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size]) #Reshape next_state similarly to state

            #Save sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            #Training step
            agent.train_model()
            score += reward #Store episodic reward
            state = next_state #Propagate state

            if done:
                #At the end of very episodesepisode, update the target network
                if e % agent.target_update_frequency == 0:
                    agent.update_target_model()
                #Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                if e % 100 == 0:
                    print("episode:", e, "  score:", score," q_value:", max_q_mean[e],"  memory length:",
                          len(agent.memory))

                # if the mean of scores of last 100 episodes is bigger than 195
                # stop training
                if agent.check_solve and solved == -1:
                    if np.mean(scores[-min(100, len(scores)):]) >= 195:
                        print("solved after", e-100, "episodes")
                        solved = e-100
                        # agent.plot_data(episodes,scores,max_q_mean[:e+1],'part-g2',arch)
                        env.close()
                        simulate(agent, 3)
                        return episodes, scores, max_q_mean, solved
    env.close()
    #agent.plot_data(episodes,scores,max_q_mean, '')
    return episodes, scores, max_q_mean, solved


if __name__ == '__main__':
    episodes = []
    scores = []
    max_q_means = []
    # num_unit_values = [16, 32, 64]
    # archs = [[8, 32], [16, 32, 32], [16, 32, 32, 32]]
    # archs = [[8], [16], [32], [64], [128]]
    # archs = [[8]]
    #names = list(map(str, archs))

    # discount factor tests
    # discount_factors = [0.7, 0.8, 0.9, 0.95, 0.99]
    # names = list(map(str, discount_factors))
    arch = [128]
    # learning_rates = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    # names = list(map(str, learning_rates))
    # mem_sizes = [500, 1000, 5000, 9000]
    # names = list(map(str, mem_sizes))

    # update_freqs = [1, 2, 4, 6, 10]
    # names = list(map(str, update_freqs))
    #
    # solved_times = []
    #
    # for update_freq in update_freqs:
    #     eps, score, qs, solved = train(arch, discount_factor=0.95, learning_rate=0.0005, mem_size=5000, update_freq=update_freq)
    #     episodes.append(eps)
    #     scores.append(score)
    #     max_q_means.append(qs[:len(score)])
    #     solved_times.append(solved)
    # plot_data_multiple(episodes, scores, max_q_means, solved_times, 'update-freq', names)

    train([128], discount_factor=0.95, learning_rate=0.0005, mem_size=5000, update_freq=1)