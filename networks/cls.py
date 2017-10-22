import tensorflow as tf
from util.utils import easy_scope


def setUp(pooled_regions, pooled_h, pooled_w, feat_channels,
        trainable=False, namespace="rcnn"):
    """Calculate bounding box regressions and class probabilities

    Preconditions:
        This function assumes that the variables accessed by tf.get_variable() already exist;
        they must already have been initialized before calling this function.

    Positional Inputs:
        pooled_regions -- A tf.Tensor object with shape (num_regions, pooled_h, pooled_w,
            num_channels) containing the pooled regions of interest in the image.
        pooled_h -- A scalar containing the height of the pooled input
        pooled_w -- A scalar containing the width of the pooled input
        feat_channels -- A scalar containing the number of channels in the pooled input

    Outputs:
        A tuple containing both:
        A list of scores for a given set of classes.  In the case of the VOC 2007 dataset,
            there are 20 classes plus one background class.
            Thus, this output should be an np.array of shape (num_regions,21) with a score
            for every class.
        A list of bounding box regressions, with a different bounding box regression for
            each class.  Each bbox regress is described by four floats, so this output
            will be an np.array of shape (num_regions, 21, 4)
    """

    last_dimension = pooled_h * pooled_w * feat_channels
    with easy_scope(namespace, reuse=True), tf.device("/gpu:0"):
        with easy_scope("fc6", reuse=True):
            flattened_in = tf.reshape(pooled_regions, (-1, last_dimension))
            prevLayer = tf.nn.bias_add(tf.matmul(flattened_in,
                tf.get_variable("Weights", trainable=trainable)),
                tf.get_variable("Bias", trainable=trainable))

        prevLayer = tf.nn.relu(prevLayer, name="relu6")

        with easy_scope("fc7", reuse=True):
            prevLayer = tf.nn.bias_add(tf.matmul(prevLayer,
                tf.get_variable("Weights", trainable=trainable)),
                tf.get_variable("Bias", trainable=trainable))

        prevLayer = tf.nn.relu(prevLayer, name="relu7")

        # Produce classification probabilities
        with easy_scope("cls_score", reuse=True):
            weights = tf.get_variable("Weights", trainable=trainable)
            bias = tf.get_variable("Bias", trainable=trainable)
            scoreLayer = tf.nn.bias_add(tf.matmul(prevLayer, weights), bias, name="out")

        # Produce regressions (note these are with respect to the individual regions, so the
        # actual regions in the image resulting from these is yet to be calculated
        with easy_scope("bbox_pred", reuse=True):
            bboxPred = tf.nn.bias_add(tf.matmul(prevLayer,
                tf.get_variable("Weights", trainable=trainable)),
                tf.get_variable("Bias", trainable=trainable), name="out")

        probLayer = tf.nn.softmax(scoreLayer, name="cls_prob")
    return bboxPred, probLayer
