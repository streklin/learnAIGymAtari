import numpy as np
import gym

from sklearn.pipeline import  FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
# from sklearn.linear_model import SGDRegressor

# from sklearn.linear_model import SGDRegressor

import tensorflow as tf


class SGDRegressor:
    def __init__(self, D):
        lr = 0.1

        # create inputs, targets, params
        # matmul doesn't like when w is 1-D
        # so we make it 2-D and then flatten the prediction
        self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')


        # make prediction and cost
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)

        # ops we want to call later
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})


class FeatureTransformer():

    def __init__(self):
        observation_examples = np.random.random((2000, 8)) * 2 - 2
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf2", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.25, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.125, n_components=1000))
        ])

        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.scaler = scaler
        self.featurizer = featurizer
        self.dimensions = feature_examples.shape[1]

    def transform(self, observations):
        # scale and featurize a set of observations
        scaled_features = self.scaler.transform(observations)
        return self.featurizer.transform(scaled_features)

class Agent():
    def __init__(self, env, learning_rate):
        self.env = env
        self.feature_transformer = FeatureTransformer()
        self.learning_rate = learning_rate

        self.models = []

        # want one model for each action in environment
        # using a SGDRegresser as the models
        for i in range(env.action_space.n):
            m = SGDRegressor(self.feature_transformer.dimensions)
            self.models.append(m)

    def predict(self, s):
        # do a prediction for each action using the pre-created SGDRegressors
        X = self.feature_transformer.transform([s]) # presumably s is an observation from the gym environment
        return np.stack([m.predict(X) for m in self.models]).T

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])  # presumably s is an observation from the gym environment
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def calculate_shaped_reward(observation, max_shaped_reward = 100.0):
    x1 = observation[0]
    y1 = observation[1]
    v1 = observation[2]
    v2 = observation[3]
    a1 = observation[4]
    a2 = observation[5]
    distance = np.sqrt(x1 ** 2 + y1 ** 2 + v1 ** 2 + v2 ** 2 + a1 ** 2 + a2 ** 2)

    return 1.0 / (distance + 1) * max_shaped_reward


def play_one(env, agent, eps, gamma, render=False, update=True):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 400:

        action = agent.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward

        if update:
            # reward agent for moving towards the landing position
            reward += calculate_shaped_reward(observation)

            # update the model
            next = agent.predict(observation)
            G = reward + gamma * np.max(next)
            agent.update(prev_observation, action, G)

        if render:
            env.render()

        iters += 1

    return totalreward


env = gym.make('LunarLander-v2')
agent = Agent(env, 0.01)
num_games = 500

# train the agent
for i in range(num_games):
    eps = 1 * (0.99 ** i)
    reward = play_one(env, agent, eps, 0.98, False)
    print("Iteration: {}: {}".format(i, reward))

# evaluate trained performance
eval_games = 100
reward_history = []
for i in range(eval_games):
    r = play_one(env, agent, 0, 0.98, False, False)
    reward_history.append(r)

avg_return = np.mean(reward_history)

print("Average Return: {}".format(avg_return))

play_one(env, agent, 0.1, 0.99, True, False)
