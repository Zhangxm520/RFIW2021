import tensorflow as tf
import numpy as np
from keras import backend as K
from Track2.utils import compute_cosine

beta=0.08

@tf.function
def compute2_loss(y_ture,y_pred):
    father_em,mother_em,child_em=tf.split(y_pred,3,1)
    return contrastive2_loss(father_em,child_em)+contrastive2_loss(mother_em,child_em)

def contrastive2_loss(x1,x2):
    x1x2=tf.concat([x1,x2],axis=0)
    x2x1=tf.concat([x2,x1],axis=0)
    cosine_mat=compute_cosine(tf.expand_dims(x1x2,axis=1),tf.expand_dims(x1x2,axis=0),axis=2)/beta

    mask=tf.eye(x1.shape[0])
    mask=tf.concat([mask,mask],axis=0)
    mask = 1.0-tf.concat([mask, mask], axis=1)

    numerators = tf.exp(compute_cosine(x1x2,x2x1,axis=1)/beta)
    denominators=tf.reduce_sum(tf.exp(cosine_mat)*mask,axis=1)
    return -tf.reduce_mean(tf.math.log(numerators/denominators),axis=0)
