from yacs.config import CfgNode as CN
from RL.DQN.config import cfg_DQN
from game.config import cfg_FLAPPY

_C = CN()

_C.SYSTEM = CN()
# Number of actions in game
_C.SYSTEM.ACTION_NUM = 4
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
# game flappy bird config
_C.SYSTEM.FLAPPY = cfg_FLAPPY

_C.TRAIN = CN()
# the configs of DQN model
_C.TRAIN.DQN = cfg_DQN
# Reward decay for Next Step
_C.TRAIN.GAMMA = 0.8
# Path for model save
_C.TRAIN.SAVE_PATH = "saved_networks_target"
# Training from somewhere Default is False
_C.TRAIN.RESUME = False
# Path for Resume Training Model
_C.TRAIN.RESUME_PATH = ""
# action frequency
_C.TRAIN.FRAME_PER_ACTION = 1

_C.TEST = CN()
# the configs of DQN model
_C.TEST.DQN = cfg_DQN
# Reward decay for Next Step
_C.TEST.EPSION_MAX = 1000
# Path for model save
_C.TEST.SAVE_PATH = "saved_networks_target"
# Testing game name
_C.TEST.GAME_NAME = "bird"
# action frequency
_C.TEST.FRAME_PER_ACTION = 4

# Exporting as cfg is a nice convention
cfg = _C