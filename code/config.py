# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

__C = edict()

# get config by: from config import cfg
cfg = __C

# number of channels of PrimaryCaps
__C.PRIMARY_CAPS_CHANNELS = 32

# iterations of dynamic routing
__C.ROUTING_ITERS = 3

# way to do dynamic iteration 'static' 'dynamic'
__C.ROUTING_WAY = 'dynamic'

# constant m+ in margin loss
__C.M_POS = 0.9

# constant m- in margin loss
__C.M_NEG = 0.1

# down-weighting constant lambda for negative classes
__C.LAMBDA = 0.5

# weight of reconstruction loss
__C.RECONSTRUCT_W = 0.0005

# initial learning rate
__C.LR = 0.001

# learning rate decay step size
__C.STEP_SIZE = 500

# learning rate decay ratio
__C.DECAY_RATIO = 0.96

# choose use bias during conv operations
__C.USE_BIAS = True

# print out loss every x steps
__C.PRINT_EVERY = 10

# snapshot every x iterations
__C.SAVE_EVERY = 500

# number of training iterations
__C.MAX_ITERS = 5000

# training mode
__C.PRIOR_TRAINING = False

# use ckpt to continue training
__C.USE_CKPT = True

# directory for saving data
#__C.DATA_DIR = '../data'
__C.DATA_DIR = '/tmp/tensorflow/mnist/'

# directory for saving check points
__C.TRAIN_DIR = '/tmp/adv_attack_capsnet/output/output_test'

# direcotry for saving tensorboard files
__C.TB_DIR = '/tmp/adv_attack_capsnet/output/tensorboard_test'
