import gym
import numpy as np
import tensorflow as tf

class ExperienceReplay:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []

    def addEvent(self, state, prev_state, action, reward):
        evt = {
            'state': state,
            'prev_state': prev_state,
            'action': action,
            'reward': reward
        }

        self.memory.append(evt)

        mem_size = len(self.memory)

        if (mem_size > self.memory_size):
            self.memory.pop(0)

    def getSamples(self, sample_size):
        return np.random.choice(self.memory, sample_size)


class SingleLayerNetwork:
    def __init__(self):
        self.initial_learning_rate = 0.1
        self.regularization_param = 0.001

    def configure_network(self):
        #network structure
        self.state = tf.placeholder(shape=[1, 16], dtype=tf.float32)
        self.weights = tf.Variable(tf.truncated_normal([16, 4]))
        self.output = tf.matmul(self.state, self.weights)
        self.prediction = tf.argmax(self.output)

        #training parameters
        self.targetOutput = tf.placeholder(shape=[1, 4], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.targetOutput - self.output)) + self.regularization_param * tf.nn.l2_loss(self.weights)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, 100, 0.96)
        self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.updateModel = self.trainer.minimize(self.loss, global_step=self.global_step)

    def reset_network(self, session):
        tf.global_variables_initializer().run()

    def predict_action(self, input, session):
        action, q_values = session.run([self.prediction, self.output], feed_dict={self.state: input})

        return action, q_values

    def update_network(self, input, target, session):
        session.run([self.updateModel], feed_dict={self.state: input, self.targetOutput: target})


env = gym.make('FrozenLake-v0')

#general constants
number_of_random_episodes = 1000
number_of_episodes = 1000
moves_per_episode = 99
epsilon = 0.1
gamma = 0.98
sample_size = 100

#object for taking the most recent events
experienceReplay = ExperienceReplay(memory_size=100000)
network = SingleLayerNetwork()
network.configure_network()

#transform the state information into a tensor
def transform_state_to_tensor(state):
    return np.identity(16)[state:state+1]


#adjust the rewards slightly to try to nudge learning along
def shape_reward(reward, done):
    #goal should always be worth 1
    if done and reward == 1:
        return reward

    #penalize holes (not done by the environment)
    if done and reward == 0:
        return -1

    #otherwise, return whatever the reward was (probably zero)
    return reward

def run_random_episodes():
    for i in range(number_of_random_episodes):
        state = env.reset()

        for m in range(moves_per_episode):
            #take a random action
            action = env.action_space.sample()

            #update the environment
            next_state, reward, done, _ = env.step(action)

            #shape the reward
            reward = shape_reward(reward, done)

            #transform state, next state to tensors for use in experience replay
            t_state = transform_state_to_tensor(state)
            t_next_state = transform_state_to_tensor(next_state)

            #push this observation into the experience replay
            experienceReplay.addEvent(t_next_state, t_state, action, reward)
            state = next_state

            #if done, don't bother with more turns
            if done:
                break

#train the simple NN to play the game.
def train_network(session):

    #train from fresh state
    network.reset_network(session)

    for i in range(number_of_episodes):
        s = env.reset()
        state = transform_state_to_tensor(s)

        for m in range(moves_per_episode):
            #predict the next action
            action, _ = network.predict_action(state, session)

            #with probability epsilon, take a random action
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()

            #update the environment, get next observation
            ns, reward, done, _ = env.step(action[0])

            #turn the next state into a tensor
            next_state = transform_state_to_tensor(ns)

            #reshape the reward
            reward = shape_reward(reward, done)

            #add this move to our collection of experiences
            experienceReplay.addEvent(next_state, state, reward, action[0])

            #get a training sample
            batch = experienceReplay.getSamples(sample_size=sample_size)

            #update weights with the sample data
            for sample in batch:
                #get the sample states
                

            #if done, don't bother with more turns
            if done:
                break


#prime the experience replay with some random experiences
run_random_episodes()

with tf.Session() as session:
    train_network(session)

