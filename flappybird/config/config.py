from yacs.config import CfgNode as CN
from RL.DQN.config import cfg_DQN

_C = CN()

_C.SYSTEM = CN()
# Number of actions in game
_C.SYSTEM.ACTION_NUM = 4
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4

_C.TRAIN = CN()
# Init epsilon value in e-greedy action selection
_C.TRAIN.DQN = cfg_DQN
# Reward decay for Next Step
_C.TRAIN.GAMMA = 0.8
# Path for model save
_C.TRAIN.SAVE_PATH = "Train"
# Training from somewhere Default is False
_C.TRAIN.RESUME = False
# Path for Resume Training Model
_C.TRAIN.RESUME_PATH = ""
# action frequency
_C.TRAIN.FRAME_PER_ACTION = 4

# Exporting as cfg is a nice convention
cfg = _C