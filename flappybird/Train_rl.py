#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("../RL")
sys.path.append("game")
from RL.DQN.dqn import DQN
import wrapped_flappy_bird as game
import numpy as np

model_path = "saved_networks_v1.2/"

TRAIN = True
if TRAIN:
    # OBSERVE = 10000
    # EXPLORE = 300000
    OBSERVE = 300000
    EXPLORE = 100000
    FINAL_EPSILON = 0.0000001
    INITIAL_EPSILON = 0.1
    GAMMA = 0.8# decay rate of past observations
else:
    GAMMA = 0.99 # decay rate of past observations
    OBSERVE = 100000. # timesteps to observe before training
    EXPLORE = 2000000. # frames over which to anneal epsilon
    FINAL_EPSILON = 0.00000001 # final value of epsilon
    INITIAL_EPSILON = 0.0000001 # starting value of epsilon

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions
# # GAMMA = 0.99 # decay rate of past observations
# # OBSERVE = 100000. # timesteps to observe before training
# # EXPLORE = 2000000. # frames over which to anneal epsilon
# # FINAL_EPSILON = 0.0001 # final value of epsilon
# # INITIAL_EPSILON = 0.0001 # starting value of epsilon
# REPLAY_MEMORY = 50000 # number of previous transitions to remember
# BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def __mkdir(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)

def trainNetwork(sess,model):
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    __mkdir("logs_" + GAME)
    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    x_t = cv2.transpose(x_t)
    # cv2.namedWindow("Image")  # 创建一个窗口用来显示图片
    # cv2.imshow("Image", (x_t))  # 显示图片
    # cv2.waitKey(0)  # 等待输入,这里主要让图片持续显示。
    #ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t_init = np.stack((x_t, x_t, x_t, x_t), axis=2)

    if 1:
        model.load(sess, model_path)

    # start training
    epsilon = INITIAL_EPSILON
    t = 295000
    s_t = s_t_init
    while "flappy bird" != "angry bird":
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            action_index, q_values = model.predict_epsion_greedy(s_t, epsilon)
        else:
            action_index, q_values = model.predict(s_t)
            action_index = 0
        a_t[action_index] = 1
        # get
        x_t1_colored, r_t, terminal = game_state.frame_step_v2(a_t)
        # run the selected action and observe next state and reward
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("Image")  # 创建一个窗口用来显示图片
        # cv2.imshow("Image", cv2.transpose(x_t1))  # 显示图片
        # cv2.waitKey(0)  # 等待输入,这里主要让图片持续显示。
        #ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(cv2.transpose(x_t1), (80, 80, 1))
        # cv2.namedWindow("Image")  # 创建一个窗口用来显示图片
        # cv2.imshow("Image", x_t1)  # 显示图片
        # cv2.waitKey(0)  # 等待输入,这里主要让图片持续显示。
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        model.train(s_t, a_t, r_t, s_t1, terminal)

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # update the old values
        if terminal:
            x_t, r_0, terminal = game_state.frame_step(do_nothing)
            x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
            x_t = cv2.transpose(x_t)
            # cv2.namedWindow("Image")  # 创建一个窗口用来显示图片
            # cv2.imshow("Image", (x_t))  # 显示图片
            # cv2.waitKey(0)  # 等待输入,这里主要让图片持续显示。
            # ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
            s_t_init = np.stack((x_t, x_t, x_t, x_t), axis=2)
            s_t = s_t_init
        else:
            s_t = s_t1
        t += 1

        # save progress every 10000 iterations

        if t % 10000 == 0:
            model.save(sess,'saved_networks_v1.2/' + GAME + '-dqn',t)
            # ?save(sess, 'saved_networks_v1.2/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if TRAIN:
            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX %e" % np.max(q_values))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    model = DQN(ACTIONS)
    sess.run(tf.initialize_all_variables())
    trainNetwork(sess, model)

def main():
    playGame()

if __name__ == "__main__":
    main()
