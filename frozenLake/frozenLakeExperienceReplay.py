import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class ExperienceReplay:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []

    def addEvent(self, state, prev_state, action, reward, done):
        evt = {
            'state': state,
            'prev_state': prev_state,
            'action': action,
            'reward': reward,
            'done': done
        }

        self.memory.append(evt)

        mem_size = len(self.memory)

        if (mem_size > self.memory_size):
            self.memory.pop(0)

    def getSamples(self, sample_size):
        return np.random.choice(self.memory, sample_size)

    def clear(self):
        self.memory = []

class SingleLayerNetwork:
    def __init__(self):
        self.initial_learning_rate = 0.01
        self.regularization_param = 0.001

    def configure_network(self):
        #network structure
        self.state = tf.placeholder(shape=[1, 16], dtype=tf.float32)
        self.weights = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
        self.output = tf.tanh(tf.matmul(self.state, self.weights))
        self.prediction = tf.argmax(self.output, 1)

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
        _, l = session.run([self.updateModel, self.loss], feed_dict={self.state: input, self.targetOutput: target})

    def set_weights(self, new_weights):
        self.weights = tf.identity(new_weights)
        pass

class DeepQNetwork:
    def __init__(self):
        self.initial_learning_rate = 0.01
        self.regularization_param = 0.001
        self.hidden_layer_size = 16

    def configure_network(self):
        #network structure
        self.state = tf.placeholder(shape=[1, 16], dtype=tf.float32)

        self.layer_1_weights = tf.Variable(tf.random_uniform([16, self.hidden_layer_size], 0, 0.01))
        self.hidden_layer = tf.tanh(tf.matmul(self.state, self.layer_1_weights))

        self.layer_2_weights = tf.Variable(tf.random_uniform([self.hidden_layer_size, 4], 0, 0.01))
        self.output = tf.tanh(tf.matmul(self.hidden_layer, self.layer_2_weights))
        self.prediction = tf.argmax(self.output, 1)

        self.targetOutput = tf.placeholder(shape=[1, 4], dtype=tf.float32)
        self.loss = tf.reduce_sum(
            tf.square(self.targetOutput - self.output)) + self.regularization_param * (tf.nn.l2_loss(self.layer_1_weights) + tf.nn.l2_loss(self.layer_2_weights))
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
        _, l = session.run([self.updateModel, self.loss], feed_dict={self.state: input, self.targetOutput: target})

    def set_weights(self, new_weights):
        pass


#general constants
number_of_random_episodes = 10000
number_of_episodes = 100
moves_per_episode = 99
epsilon = 0.01
gamma = 0.99
sample_size = 500
eval_steps = 200

#object for taking the most recent events
experienceReplay = ExperienceReplay(memory_size=10000)
network = SingleLayerNetwork()
network.configure_network()

target_network = SingleLayerNetwork()
target_network.configure_network()

#Bellman Equation step
def bellmen_update(reward, gamma, maxQ, q_values, action, done):
    # if this is not the last step, use discounted reward
    if not done:
        q_values[0, action[0]] = reward + gamma * maxQ

    # otherwise, use the true reward
    if done:
        q_values[0, action[0]] = reward

    return q_values

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

    #give the Mr. Meseeks a living penalty - it wants to die with the
    #highest score.
    return -0.001

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
            experienceReplay.addEvent(t_next_state, t_state, action, reward, done)
            state = next_state

            #if done, don't bother with more turns
            if done:
                break

performance_history = []

#train the simple NN to play the game.
def train_network(session):

    #update the target network every C steps
    #variable name is reference to Googles Atari playing paper in Nature
    C = 2000

    #train from fresh state
    network.reset_network(session)

    #track training iteration so we know when to copy the weights
    #to the target network
    ti = 0

    for i in range(number_of_episodes):

        #keep track of how well the network is doing
        performance = evaluate_performance(session, False)

        s = env.reset()
        state = transform_state_to_tensor(s)

        #track its output to keep me from getting paranoid
        print "Episode: ", i, " current performance: ", performance

        #track performance history
        performance_history.append(performance)

        for m in range(moves_per_episode):
            #predict the next action
            action, allQ = network.predict_action(state, session)

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
            experienceReplay.addEvent(next_state, state, reward, action[0], done)

            # if done, don't bother with more turns
            if done:
                break

        #get a training sample
        batch = experienceReplay.getSamples(sample_size=sample_size)


        #update weights with the sample data
        for sample in batch:
            #get the Q values for the start state
            action, qValues = target_network.predict_action(sample['prev_state'], session)

            #set the action tensor to the sample action
            action[0] = sample['action']

            #get the best action for the target state
            _, targetQ = target_network.predict_action(sample['state'], session)
            maxQ = np.max(targetQ)

            #perform a bellmen update on the Q values
            qValues = bellmen_update(reward=sample['reward'], gamma=gamma, maxQ=maxQ, q_values=qValues, action=action, done=sample['done'])

            #call the train function
            network.update_network(sample['prev_state'], qValues, session)

            #update training iteration number
            ti += 1

            #when we hit C, we want to update the target network to match our current
            #approxmimation.
            if ti == C:
                print "Copying weights to target network ..."

                target_network.set_weights(network.weights)
                ti = 0

#run the agent without any random actions to see how well its learned the optimal policy
#due to stochasticity in the environment, we can never win 100%, but a result of around
#80% or so should be considered reasonable.
def evaluate_performance(session, render_environment):
    rewardList = []

    for i in range(eval_steps):
        rSum = 0.0
        done = False
        s = env.reset()
        state = transform_state_to_tensor(s)

        while not done:
            action, _ = network.predict_action(state, session)

            s1, r, done, _ = env.step(action[0])

            rSum += r
            next_state = transform_state_to_tensor(s1)
            state = next_state

            #render as needed
            if render_environment:
                env.render()

            rewardList.append(rSum)

    return (sum(rewardList) * 1.0 / eval_steps) * 100

def experience_replay_alg():
    # prime the experience replay with some random experiences
    experienceReplay.clear()

    print "Generating Experience Replay Data"
    run_random_episodes()

    with tf.Session() as session:
        train_network(session)
        performance = evaluate_performance(session, False)
        print "Percent of sucessful episodes: " + str(performance) + "%"
        return performance


env = gym.make('FrozenLake-v0')

#performance of the network is a bit unstable, running multiple times
#and averaging the result to get a good measure of the performance
num_iterations = 1
results = []

for i in range(num_iterations):
    p = experience_replay_alg()
    results.append(p)

total_score = sum(results)
avg_score = total_score / num_iterations

#print "Average Score: ", avg_score, "%"
plt.plot(performance_history)
plt.show()

