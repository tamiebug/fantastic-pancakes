import tensorflow as tf
import numpy as np

import os
import unittest

from util import utils
import util.settings as s
import test.testUtils
from test.testUtils import array_equality_assert
from util import frcnn_forward

from layers.custom_layers import roi_pooling_layer
from networks import cls
from networks import loadNetVars

class ClsTest(unittest.TestCase):
    """
    Tests whether the classification part of the network has activations matching a reference.

    It is fed in the features and regions of interest, and is expected to reproduce the correct
    pooled regions of interest (pool5), correct fc7, cls_score, and bbox_pred (correct classi-
    fication scores and bounding box regression box predictions
    """

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.reference_activations = np.load(os.path.join(
            self.base_dir, "activations/test_values.npz"))

    def test_roi_pooling(self):
        """
        Tests the Region of Interest pooling layer (pool5) using actual activation data from the reference network.

        We feed in the rois data, the conv5_3 features data, and the image data (dimensions and scale factor)
        obtained from the test image in test/images/000456.jpg and compare it to the expected activation values.
        """
        # We don't need the actual image pixels, just the dimensionality and scale information.
        _, im_info = frcnn_forward.process_image(
                os.path.join(self.base_dir, "images/000456.jpg"))

        def runGraph(self, im_info):
            with tf.Session() as sess, tf.device("/cpu:0") as dev:
                # the RoI Pooling code currently is CPU only, GPU version not yet developed.
                try:
                    features = self.reference_activations['conv5_3']
                except KeyError:
                    print("Warning:  conv5_3 not found in reference activations.  Something \
                            wrong with .npz file")
                try:
                    rois = self.reference_activations['rois']
                except KeyError:
                    print("Warning: rois not found in reference activations.  Something \
                            wrong with .npz file")

                feats = tf.placeholder("float", [None,None,None])
                rois_placeholder = tf.placeholder("float", [None,4])
                info = tf.placeholder("float", [3])

                # 33.301205 is the diff you get if everything is zeroes.

                pool_layer = roi_pooling_layer(feats, # Need to squeeze out batch number
                                    info,
                                    rois_placeholder,
                                    7, # Pooled height
                                    7, # Pooled width
                                    16,# Feature Stride (16 for vgg)
                                    name='roi_pooling_layer') 
                return sess.run(pool_layer, feed_dict={
                    feats : np.squeeze(features),
                    rois_placeholder : rois[:,[1,2,3,4]],
                    info : im_info
                    })[0]

        result= utils.isolatedFunctionRun(runGraph, False, self, im_info)
        return array_equality_assert(self, result, self.reference_activations['pool5'])

    def test_fc7(self):
        """ Tests the FC layers, by looking at fc7, using the roi pooled input"""
        def runGraph(self):
            with tf.Session() as sess, tf.device("/gpu:0") as dev:
                try:
                    pool5_in = self.reference_activations['pool5']
                except KeyError:
                    print("Warning:  pool5 not found in reference activations.  Something \
                            wrong with .npz file")
                
                pool5 = tf.placeholder("float")
                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH, s.DEF_FRCNN_BIASES_PATH)
                net = cls.setUp(pool5, 7, 7, 512, namespace="rcnn")
                sess.run(tf.global_variables_initializer())
                return sess.run(["rcnn/relu7:0"], feed_dict={pool5 : pool5_in})
                
        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations['fc7'])

    
    def test_cls_score(self):
        """ Tests the cls_score layer via comarison to a reference activation"""
        def runGraph(self):
            with tf.Session() as sess, tf.device("/gpu:0") as dev:
                try:
                    relu7_in = self.reference_activations['fc7']
                except KeyError:
                    print("Warning:  fc7 not found in reference activations.  Something \
                            wrong with .npz file")
                
                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH, s.DEF_FRCNN_BIASES_PATH)
                net = cls.setUp(tf.placeholder("float", name="pool5"), 7, 7, 512, namespace="rcnn")
                sess.run(tf.global_variables_initializer())
                return sess.run(["rcnn/cls_score/out:0"], feed_dict={"rcnn/relu7:0" : relu7_in})
                
        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations['cls_score'])
    
    def test_bbox_pred(self):
        """ Tests the bbox_pred layer via comparison to a reference activation"""
        def runGraph(self):
            with tf.Session() as sess, tf.device("/gpu:0") as dev:
                try:
                    relu7_in = self.reference_activations['fc7']
                except KeyError:
                    print("warning:  fc7 not found in reference activations.  Something \
                            wrong with .npz file")

                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH, s.DEF_FRCNN_BIASES_PATH)
                net = cls.setUp(tf.placeholder("float", name="pool5"), 7, 7, 512, namespace="rcnn")
                sess.run(tf.global_variables_initializer())
                return sess.run(["rcnn/bbox_pred/out:0"], feed_dict={"rcnn/relu7:0" : relu7_in})

        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations['bbox_pred'])

