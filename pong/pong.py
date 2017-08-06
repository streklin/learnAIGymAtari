import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
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


#Bellman Equation step
def bellmen_update(reward, gamma, maxQ, q_values, action, done):
    # if this is not the last step, use discounted reward
    if not done:
        q_values[0, action[0]] = reward + gamma * maxQ

    # otherwise, use the true reward
    if done:
        q_values[0, action[0]] = reward

    return q_values



def transform_to_greyscale(rgb_values):
    return np.dot(rgb_values[..., :3], [0.3, 0.6, 0.1])


def pong_covnet(image, qValues, mode):
    #reshape the image tensor to be compatible with tensor flow setup
    inputs_layer = tf.reshape(image, [-1, 210, 160, 1])

    #start with a convolution
    conv1 = tf.layers.conv2d(
        inputs=inputs_layer,
        filters=32,
        kernel_size=[10, 10],
        padding="same",
        activation=tf.nn.relu
    )

    #max pool (no overlap)
    max_pool = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    #now we should have a [-1, 105, 80, 32] tensor
    conv2 = tf.layers.conv2d(
        inputs=max_pool,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    max_pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[5, 5],
        strides=5
    )


    #should now have a [-1, 21, 16, 64] tensor
    #flatten the max_pool2 so we can feed it to a dense layer
    flat_pool = tf.reshape(max_pool2, [-1, 21 * 16 * 64])
    dense = tf.layer.dense(
        inputs=flat_pool,
        units=1024,
        activation=tf.nn.relu
    )

    #dropout (training only)
    dropout = tf.layer.dropout(
        inputs=dense,
        rate=0.5,
        training=(mode == learn.ModeKeys.TRAIN)
    )

    #prediction
    logits = tf.layers.dense(
        inputs=dense,
        units=6 #pong has six actions, apparently
    )

    loss = None
    train_op = None

    # Calculate loss for both TRAIN and EVAL modes
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(
            qValues,
            logits
        )

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="Adam")


    #generate the next action
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1
        )
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def play_pong():
    memory = ExperienceReplay(memory_size=100000)
    covnet = tf.contrib.learn.Estimator(model_fn=pong_covnet)

    

env = gym.make('Pong-v0')


play_pong()