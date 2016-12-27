import numpy as np
import tensorflow as tf

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
    with tf.variable_scope(namespace, reuse=True) as model_scope:
        model_scope.reuse_variables()
        with tf.variable_scope("fc6") as scope:
            flattened_in = tf.reshape(pooled_regions, (-1, last_dimension))
            prevLayer = tf.nn.bias_add(tf.matmul(flattened_in, 
                            tf.get_variable("Weights")), tf.get_variable("Bias"))
        
        prevLayer = tf.nn.relu(prevLayer, name="relu6")

        with tf.variable_scope("fc7") as scope:
            prevLayer = tf.nn.bias_add(tf.matmul(prevLayer,
                            tf.get_variable("Weights")), tf.get_variable("Bias"))

        prevLayer = tf.nn.relu(prevLayer, name="relu7")

        # Produce classification probabilities
        with tf.variable_scope("cls_score") as scope:
            scoreLayer = tf.nn.bias_add(tf.matmul(prevLayer,
                            tf.get_variable("Weights")), tf.get_variable("Bias"),name="out")
        
        probLayer = tf.nn.softmax(scoreLayer, name="cls_prob")

        # Produce regressions (note these are with respect to the individual regions, so the
        # actual regions in the image resulting from these is yet to be calculated
        with tf.variable_scope("bbox_pred") as scope:
            bboxPred = tf.nn.bias_add(tf.matmul(prevLayer,
                        tf.get_variable("Weights")), tf.get_variable("Bias"), name="out")

    return bboxPred, probLayer
