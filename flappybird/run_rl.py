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


model_path = "saved_networks_target"

EPSION_MAX = 1000
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions
FRAME_PER_ACTION = 2

def __mkdir(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)

def runNetwork(sess,model):
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    __mkdir("logs_" + GAME)
    # printing
    a_file = open("logs_" + GAME + "/scores.txt", 'w')

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
    t =0
    s_t = s_t_init
    while "flappy bird" != "angry bird" and t < EPSION_MAX:
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            action_index, _ = model.predict(s_t)
        else:
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

        # update the old values
        if terminal:
            print("EPSILON", t, "/ REWARD", game_state.get_last_score())
            a_file.write(str(t) + "," + str(game_state.get_last_score())+'\n')
            x_t, r_0, terminal = game_state.frame_step(do_nothing)
            x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
            x_t = cv2.transpose(x_t)
            # cv2.namedWindow("Image")  # 创建一个窗口用来显示图片
            # cv2.imshow("Image", (x_t))  # 显示图片
            # cv2.waitKey(0)  # 等待输入,这里主要让图片持续显示。
            # ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
            s_t_init = np.stack((x_t, x_t, x_t, x_t), axis=2)
            s_t = s_t_init
            t += 1
        else:
            s_t = s_t1
        #t += 1

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
    runNetwork(sess, model)

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--conf", dest='conf')
    parser.parse_args()
    playGame()

if __name__ == "__main__":
    main()
