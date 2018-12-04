import  glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.nets.inception_v3 as inception_v3

INPUT_DATA='./flower_processed_data.npy'
CKPY_FILE='./inception_v3.ckpt'

LEARNING_RTAE=0.0001
STEPS=300
BATCH=32
N_CLASS=5


