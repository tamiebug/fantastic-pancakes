import numpy as np
import tensorflow as tf

from util.utils import easy_scope

def setUp(pooled_regions, pooled_h, pooled_w, feat_channels, namespace="rcnn"):
    """
    This function takes in the roi_pooling_layer output and spits out a bounding box regression and classification score.

    Preconditions:
        This function assumes that the variables accessed by tf.get_variable() already exist.
        They must already have been initialized before calling this function.
        
    Input:
        pooled_regions: A tf.Tensor object with shape (num_regions, pooled_h, pooled_w, num_channels) containing the pooled
                    regions of interestin the image.
    Outputs:
        A tuple containing both:
        A list of scores for a given set of classes.  In the case of the VOC 2007 dataset, there are 20 classes plus one background class.
            Thus, this output should be an np.array of shape (num_regions,21) with a score for every class
        A list of bounding box regressions, with a different bounding box regression for each class.  Each bbox regress is described by
            four floats, so this output will be an np.array of shape (num_regions, 21, 4)
    """

    last_dimension = pooled_h * pooled_w * feat_channels
    with easy_scope(namespace, reuse=True):
        with easy_scope("fc6") as scope:
            flattened_in = tf.reshape(pooled_regions, (-1, last_dimension))
            prevLayer = tf.nn.bias_add(tf.matmul(flattened_in, 
                        tf.get_variable("Weights")), tf.get_variable("Bias"))
        
        prevLayer = tf.nn.relu(prevLayer, name="relu6")

        with easy_scope("fc7", reuse=True):
            prevLayer = tf.nn.bias_add(tf.matmul(prevLayer,
                            tf.get_variable("Weights")), tf.get_variable("Bias"))

        prevLayer = tf.nn.relu(prevLayer, name="relu7")

        # Produce classification probabilities
        with easy_scope("cls_score", reuse=True):
            weights = tf.get_variable("Weights")
            bias = tf.get_variable("Bias")
            scoreLayer = tf.nn.bias_add(tf.matmul(prevLayer,
                            weights), bias,name="out")
        
        probLayer = tf.nn.softmax(scoreLayer, name="cls_prob")

        # Produce regressions (note these are with respect to the individual regions, so the
        # actual regions in the image resulting from these is yet to be calculated
        with easy_scope("bbox_pred", reuse=True) as scope:
            bboxPred = tf.nn.bias_add(tf.matmul(prevLayer,
                        tf.get_variable("Weights")), tf.get_variable("Bias"), name="out")


    return bboxPred, probLayer
