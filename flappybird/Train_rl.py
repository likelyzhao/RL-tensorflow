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


def trainNetwork(sess, model):
    save_path = cfg.TRAIN.SAVE_PATH
    FRAME_PER_ACTION = cfg.TRAIN.FRAME_PER_ACTION

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    __mkdir(save_path)
    # printing
    a_file = open(save_path + "/readout.txt", 'w')
    h_file = open(save_path + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(action_num)
    do_nothing[0] = 1
    frame, r_0, terminal = game_state.frame_step(do_nothing)
    frame = frame_process(frame)
    s_t_init = np.stack((frame, frame, frame, frame), 2)

    if cfg.TRAIN.RESUME:
        model.load(sess, cfg.TRAIN.RESUME_PATH)

    # start training
    t = 0
    s_t = s_t_init
    while "flappy bird" != "angry bird":
        a_t = np.zeros([action_num])
        if t % FRAME_PER_ACTION == 0:
            action_index, q_values = model.predict_epsion_greedy(s_t)
        else:
            action_index, q_values = model.predict(s_t)
            action_index = 0
        a_t[action_index] = 1

        frame, r_t, terminal = game_state.frame_step_v2(a_t)
        # run the selected action and observe next state and reward
        frame = frame_process(frame, extention=True)
        s_t1 = np.append(frame, s_t[:, :, :3], axis=2)

        model.train(s_t, a_t, r_t, s_t1, terminal)

        # update the old values
        if terminal:
            x_t, r_0, terminal = game_state.frame_step(do_nothing)
            frame = frame_process(x_t)
            s_t = np.stack((frame, frame, frame, frame), axis=2)
        else:
            s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if (t - cfg.TRAIN.DQN.OBSERVE_NUM) > 0 and t % 10000 == 0:
            model.save(sess, save_path, t)
            # ?save(sess, 'saved_networks_v1.2/' + GAME + '-dqn', global_step = t)

        state = "train"
        print("TIMESTEP", t, "/ STATE", state, \
                "/ ACTION", action_index, "/ REWARD", r_t, \
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
    model = DQN(cfg.TRAIN.DQN, action_num)
    sess.run(tf.initialize_all_variables())
    trainNetwork(sess, model)


def main():
    cfg.merge_from_file("train.yaml")
    cfg.freeze()
    print(cfg)
    playGame()


if __name__ == "__main__":
    main()
