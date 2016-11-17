import tensorflow as tf
import os

dot_slash = os.path.dirname(__file__)

# Making roi_pooling_layer available for import as a library
roi_location = os.path.join(dot_slash,"roi_pool.so")
op_module = tf.load_op_library(roi_location)
roi_pooling_layer = op_module.roi_pooling_layer
