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
from config.config import cfg

action_num = cfg.SYSTEM.ACTION_NUM
DEBUG = False


def frame_process(frame_ori, extention=False):
    x_t = cv2.cvtColor(cv2.resize(frame_ori, (80, 80)), cv2.COLOR_BGR2GRAY)
    x_t = cv2.transpose(x_t)
    if extention:
        x_t = np.reshape(x_t, (80, 80, 1))
    #ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY) // option
    if DEBUG:
        cv2.namedWindow("Image")  # 创建一个窗口用来显示图片
        cv2.imshow("Image", (x_t))  # 显示图片
        cv2.waitKey(0)  # 等待输入,这里主要让图片持续显示。
    return x_t


def __mkdir(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)


def runNetwork(sess,model):
    # open up a game state to communicate with emulator
    game_state = game.GameState(cfg.SYSTEM.FLAPPY)
    __mkdir("logs_" + cfg.TEST.GAME_NAME)
    # printing
    a_file = open("logs_" + cfg.TEST.GAME_NAME + "/scores.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(action_num)
    do_nothing[0] = 1
    frame, r_0, terminal = game_state.frame_step(do_nothing)
    frame = frame_process(frame)
    s_t_init = np.stack((frame, frame, frame, frame), axis=2)

    if cfg.TEST.SAVE_PATH:
        model.load(sess, cfg.TEST.SAVE_PATH)
    # start training
    t = 0
    s_t = s_t_init
    score_total = 0
    while "flappy bird" != "angry bird" and t < cfg.TEST.EPSION_MAX:
        a_t = np.zeros([action_num])
        if t % cfg.TEST.FRAME_PER_ACTION == 0:
            action_index, _ = model.predict(s_t)
        else:
            action_index = 0
        a_t[action_index] = 1
        # get
        print("action = ",action_index)
        frame, r_t, terminal = game_state.frame_step_v2(a_t)
        frame = frame_process(frame, extention=True)
        s_t = np.append(frame, s_t[:, :, :3], axis=2)

        # update the old values
        if terminal:
            print("EPSILON", t, "/ REWARD", game_state.get_last_score())
            a_file.write(str(t) + "," + str(game_state.get_last_score())+'\n')
            score_total += game_state.get_last_score()
            x_t, r_0, terminal = game_state.frame_step(do_nothing)
            frame = frame_process(x_t)
            s_t_init = np.stack((frame, frame, frame, frame), axis=2)
            s_t = s_t_init
            t += 1

    print("avescore is ", str(score_total*1.0/t))
    a_file.write("avescore is " + str(score_total*1.0/t) + '\n')
    a_file.close()


def playGame():
    sess = tf.InteractiveSession()
    model = DQN(cfg.TEST.DQN,action_num)
    sess.run(tf.initialize_all_variables())
    runNetwork(sess, model)


def main():
    cfg.merge_from_file("test.yaml")
    cfg.freeze()
    playGame()


if __name__ == "__main__":
    main()
