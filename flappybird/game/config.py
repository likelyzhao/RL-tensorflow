from yacs.config import CfgNode as CN

_FLAPPYBIRD = CN()
# FPS for the game
_FLAPPYBIRD.FPS = 3000
# game screen width
_FLAPPYBIRD.SCREENWIDTH = 288
# game screen height
_FLAPPYBIRD.SCREENHEIGHT = 512
# Number of agent observe epsilon reduce from init to final
_FLAPPYBIRD.OBSERVE_NUM = 300000
# Reward decay for Next Step
_FLAPPYBIRD.GAMMA = 0.8
# Number of previous transitions to remember
_FLAPPYBIRD.REPLAY_MEMORY_LENGTH = 50000
# BatchSize for Q learning
_FLAPPYBIRD.TRAIN_BATCHSIZE = 32



cfg_FLAPPY = _FLAPPYBIRD