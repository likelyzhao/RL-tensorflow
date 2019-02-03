#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("../common")
from common.netconstruct import weight_variable,bias_variable, conv2d,max_pool_2x2
import random
import numpy as np
from collections import deque
from DQN.config import cfg_DQN


def createNetwork(ACTIONS):
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


class DQN(object):
    def __init__(self, cfg, action_num):
        print(cfg)
        self.ACTIONS = action_num
        self.status, self.readout,self.h_fc1 = createNetwork(self.ACTIONS)
        actions = tf.placeholder("float", [None, self.ACTIONS])
        self.y = tf.placeholder("float", [None])
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, actions), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        self.model_saver = tf.train.Saver()
        self.replay_memory = deque()
        self.replay_memory_terminal = deque()
        # store the previous observations in replay memory

        self.t = 0
        self.saver = tf.train.Saver()

        # observer phase
        self.INITIAL_EPSILON = cfg.INITIAL_EPSILON
        self.epsilon = self.INITIAL_EPSILON
        self.FINAL_EPSILON = cfg.FINAL_EPSILON
        self.EXPLORE_NUM = cfg.EXPLORE_NUM
        self.OBSERVE_NUM = cfg.OBSERVE_NUM
        self.GAMMA = cfg.GAMMA

        # Q learning parameters
        self.memory_size = cfg.REPLAY_MEMORY_LENGTH
        self.sample_batch = cfg.TRAIN_BATCHSIZE

    def train(self, s_t, a_t, r_t, s_t1, terminal):
        if terminal:
            self.replay_memory_terminal.append((s_t, a_t, r_t, s_t1, terminal))
        else:
            self.replay_memory.append((s_t, a_t, r_t, s_t1, terminal))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.popleft()
        if len(self.replay_memory_terminal) > self.memory_size:
            self.replay_memory_terminal.popleft()

        if self.t > self.OBSERVE_NUM:
            # sample a minibatch to train on
            print("training ")
            minibatch = random.sample(self.replay_memory, self.sample_batch)
            for i in range(self.sample_batch):
                minibatch.append(random.choice(self.replay_memory_terminal))

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            # readout_j1_batch = self.readout.eval(feed_dict={s: s_j1_batch})
            readout_j1_batch = self.pridict(s_j1_batch)
            for i in range(0, len(minibatch)):
                terminal_t = minibatch[i][4]
                # if terminal, only equals reward
                if terminal_t:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + self.GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            self.train_step.run(feed_dict={
                self.y: y_batch,
                self.a: a_batch,
                self.s: s_j_batch}
            )

    def predict_epsion_greedy(self, status):
        import random
        self.t += 1
        # scale down epsilon
        if self.epsilon > self.FINAL_EPSILON and self.t > self.OBSERVE_NUM:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE_NUM
        if random.random() <= self.epsilon:
            print("----------Random Action----------")
            readout_t = self.readout.eval(feed_dict={self.status: [status]})[0]
            action_index = random.randrange(self.ACTIONS)
            return action_index, readout_t
        else:
            return self.predict(status)

    def predict(self, status):
        readout_t = self.readout.eval(feed_dict={self.status: [status]})[0]
        action_index = np.argmax(readout_t)
        return action_index, readout_t

    def load(self, tf_sess, model_path):
        # sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(tf_sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save(self, tf_sess, save_path, global_step):
        # sess.run(tf.initialize_all_variables())
        self.saver.save(tf_sess, save_path, global_step=global_step)


def main():
    model = DQN(cfg_DQN, 4)


if __name__ == "__main__":
    main()
