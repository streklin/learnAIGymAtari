#included here for reference
import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

rList = []

def shape_reward(current_reward, current_state, done):
    if current_reward == 1:
        return current_reward

    if done and current_reward == 0:
        return -2.0

    return current_reward
    #return -0.001 + (0.0005 / (16.0 - current_state))

def train_network():
    # constants
    num_episodes = 1000
    epsilon = 0.5
    gamma = 0.98  # parameter for the bellmen equation

    tf.reset_default_graph()

    #configure single layer network
    inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
    W1 = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))

    Qout = tf.matmul(inputs1, W1)
    predict = tf.argmax(Qout, 1)

    nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout)) + 0.0001 * tf.nn.l2_loss(W1) # + 0.001 * tf.nn.l2_loss(W2)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.9, global_step, 100, 0.96)
    trainer = tf.train.AdamOptimizer(learning_rate)
    updateModel = trainer.minimize(loss, global_step=global_step)

    init = tf.initialize_all_variables()

    #run the game
    with tf.Session() as session:

        #initialize the network
        session.run(init)

        for i in range(num_episodes):
            state = env.reset()
            moves = 0

            while moves < 99:
                moves += 1

                #get the next predictions
                [action, q_values] = session.run([predict, Qout], feed_dict={inputs1: np.identity(16)[state:state + 1]})

                #with probability epsilon perform a random action to explore the state space
                if np.random.rand(1) < epsilon:
                    action[0] = env.action_space.sample()

                #update the environment
                next_state, reward, done, _ = env.step(action[0])

                #shape the reward
                reward = shape_reward(next_state, reward, done)

                #get the predicted q values
                q_prime = session.run([Qout], feed_dict={inputs1: np.identity(16)[next_state:next_state + 1]})

                #bellmen update
                maxQ = np.max(q_prime)


                #if this is not the last step, use discounted reward
                if not done:
                    q_values[0, action[0]] = reward + gamma * maxQ

                #otherwise, use the true reward
                if done:
                    q_values[0, action[0]] = reward

                #train the network
                session.run([updateModel], {inputs1:np.identity(16)[state:state+1],nextQ:q_values})

                state = next_state

                if done:
                    # Reduce chance of random action as we train the model.
                    break

            epsilon = 1. / ((i / 50) + 10)

        for i in range(100):
            # Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            # The Q-Network
            while not d:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a, q = session.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s + 1]})
                # print q

                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])

                rAll += r
                s = s1
                env.render()
                rList.append(rAll)


train_network()
print "Percent of succesful episodes: " + str((sum(rList) / 100) * 100) + "%"

