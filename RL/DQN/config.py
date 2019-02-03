from yacs.config import CfgNode as CN

_DQN = CN()
# Init epsilon value in e-greedy action selection
_DQN.INITIAL_EPSILON = 0.1
# Final epsilon value in e-greedy action selection in Training stage
_DQN.FINAL_EPSILON = 1e-6
# Number of agent exploring epsilon set with init value
_DQN.EXPLORE_NUM = 100000
# Number of agent observe epsilon reduce from init to final
_DQN.OBSERVE_NUM = 300000
# Reward decay for Next Step
_DQN.GAMMA = 0.8
# Number of previous transitions to remember
_DQN.REPLAY_MEMORY_LENGTH = 50000
# BatchSize for Q learning
_DQN.TRAIN_BATCHSIZE = 32

cfg_DQN = _DQN