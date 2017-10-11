import tensorflow as tf
from tensorflow.python.framework import ops
import os

dot_slash = os.path.dirname(__file__)

# Making roi_pooling_layer available for import as a library
roi_location = os.path.join(dot_slash,"rpl.so")
op_module = tf.load_op_library(roi_location)
roi_pooling_layer = op_module.roi_pooler

# Maknig nms available for import as a library
nms_location = os.path.join(dot_slash, "nms.so")
nms_module = tf.load_op_library(nms_location)
nms = nms_module.nms

# Making roi_pooling_layer's gradient available for import
roi_grad_location = os.path.join(dot_slash, "rpl_grad.so")
roi_grad_module = tf.load_op_library(roi_grad_location)
roi_pooling_layer_grad = roi_grad_module.roi_pooler_grad

@ops.RegisterGradient("RoiPooler")
def _roi_pool_grad_cc(op, grad):
    return [roi_pooling_layer_grad(op.inputs[0], op.inputs[1], op.inputs[2], grad,
            op.get_attr("pooled_height"), op.get_attr("pooled_width"), op.get_attr("feature_stride")), None, None]
