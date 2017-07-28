#frozen lake Q-Learning example from medium.com blog article https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
#included here for reference
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

num_rectifiers = 16

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([16,num_rectifiers], 0, 0.01))

W2 = tf.Variable(tf.truncated_normal([num_rectifiers, 4]))

hidden = tf.tanh(tf.matmul(inputs1, W1))
hidden = tf.nn.dropout(hidden, 0.5)

Qout = tf.matmul(hidden, W2)

#Qout = tf.matmul(inputs1,W1)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout)) + 0.0001 * tf.nn.l2_loss(W1) + 0.0001 * tf.nn.l2_loss(W2)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 500, 0.96)
trainer = tf.train.AdamOptimizer(learning_rate)
updateModel = trainer.minimize(loss, global_step=global_step)

init = tf.initialize_all_variables()

# Set learning parameters
y = 0.25
e = 0.1
#num_episodes = 10000
random_episodes = 1000
num_episodes = 5000
#create lists to contain total rewards and steps per episode
jList = []
rList = []

def shape_reward(current_reward, current_state, done):
    if current_reward == 1:
        print "Win"
        return current_reward

    if done and current_reward == 0:
        return -2.0

    return -0.001 + (0.0005 / (16.0 - current_state))

#train the network
with tf.Session() as sess:
    sess.run(init)

    for i in range(random_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-Network
        while j < 99:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s + 1]})

            #random actions only, let the system learn a bit
            a[0] = env.action_space.sample()
            # Get new state and reward from environment
            s1, r, d, _ = env.step(a[0])

            # add a "reward for failure"
            # if d and r == 0:
            # r = -1
            r = shape_reward(r, s1, d)

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1
            # Train our network using target and predicted Q values
            sess.run([updateModel], feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
            s = s1

            if d == True:
                # Reduce chance of random action as we train the model.
                break

    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])

            #add a "reward for failure"
            #if d and r == 0:
                #r = -1
            r = shape_reward(r, s1, d)

            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ

            #update the reward to be correctly calculated
            if not d:
                targetQ[0,a[0]] = r + y*maxQ1

            if d:
                targetQ[0,a[0]] = r

            #Train our network using target and predicted Q values
            sess.run([updateModel],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            s = s1

            if d == True:
                # Reduce chance of random action as we train the model.
                break

        e = 1. / ((i / 50) + 10)

    #run the trained network
    for i in range(100):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while not d:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a, q = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            #print q

            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])

            rAll += r
            s = s1

            env.render()
            jList.append(j)
            rList.append(rAll)

print "Percent of succesful episodes: " + str((sum(rList) / 100) * 100) + "%"

