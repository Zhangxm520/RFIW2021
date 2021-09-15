import tensorflow as tf
import os
import random
import numpy as np


def compute_cosine(x1, x2, axis=1):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=axis))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=axis))
    x1_x2 = tf.reduce_sum(tf.multiply(x1, x2), axis=axis)
    cosin = x1_x2 / (x1_norm * x2_norm)
    return cosin

def mylog(*t,path = 'log.txt'):
    t=" ".join([str(now) for now in t])
    print(t)
    if os.path.isfile(path) == False:
        f = open(path, 'w+')
    else:
        f = open(path, 'a')
    f.write(t + '\n')
    f.close()

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)