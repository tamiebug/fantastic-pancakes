import unittest

import os

import numpy as np
import tensorflow as tf

from util import utils
from util import settings as s
from test.testUtils import array_equality_assert
from util import frcnn_forward

from networks import rpn
from networks import loadNetVars

class generateAnchorsTest(unittest.TestCase):

    def test_properNumberOfAnchors(self):
        output = rpn.generateAnchors()
        self.assertEqual(9, len(output))

    def test_generateAnchors(self):
        output = rpn.generateAnchors(ratios=[2, 1, .5])
        expected = [
            [-83.0, -39.0, 100.0, 56.0],
            [-175.0, -87.0, 192.0, 104.0],
            [-359.0, -183.0, 376.0, 200.0],
            [-55.0, -55.0, 72.0, 72.0],
            [-119.0, -119.0, 136.0, 136.0],
            [-247.0, -247.0, 264.0, 264.0],
            [-35.0, -79.0, 52.0, 96.0],
            [-79.0, -167.0, 96.0, 184.0],
            [-167.0, -343.0, 184.0, 360.0]
        ]

        for rowOut, rowExp in zip(output, expected):
            for eleOut, eleExp in zip(rowOut, rowExp):
                self.assertAlmostEqual(eleOut, eleExp, places=5)


class RpnTest(unittest.TestCase):
    """
    Tests whether the region proposal network activations match a reference.

    The Region Proposal Network is fed in the conv5_3 features from the feature extractor
    and the outputs of rpn_bbox_pred and rpn_cls_score are compared to a reference.  Ref-
    erence activations are then fed into the prosoal layer, along with im_info, to test
    the output to the reference rois activations.
    """

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.reference_activations = np.load(os.path.join(
            self.base_dir, "activations/test_values.npz"))
        _, self.im_info = frcnn_forward.process_image(
            os.path.join(self.base_dir, "images/000456.jpg"))

    def test_rpn_cls_score(self):
        """
        Tests the rpn_cls_score and rpn_conv/3x3 layers.

        We feed in conv5_3 activation data from a reference activation gathered from
        test/images/000456.jpg and compare the output from rpn_cls_score
        """

        def runGraph(self):
            with tf.Session() as sess:
                features_activations = self.reference_activations['conv5_3']
                features = tf.placeholder("float", [1, None, None, 512])
                info = tf.placeholder("float", [3])

                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH,
                    s.DEF_FRCNN_BIASES_PATH)
                rpn.Rpn(features, info, namespace="rcnn")
                # for op in tf.get_default_graph().get_operations():
                #    print op.name
                sess.run(tf.global_variables_initializer())
                return sess.run(["rcnn/rpn_cls_score/out:0"], feed_dict={
                    features: features_activations, info: self.im_info})

        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations["rpn_cls_score"])

    def test_rpn_bbox_pred(self):
        """
        Tests the rpn_bbox_pred and rpn_conv/3x3 layers.

        We feed in conv5_3 activation data from a reference activation gathered from
        test/images/000456.jpg and compare the output from rpn_bbox_pred
        """

        def runGraph(self):
            with tf.Session() as sess:
                features_activations = self.reference_activations['conv5_3']
                features = tf.placeholder("float", [1, None, None, 512])
                info = tf.placeholder("float", [3])
                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH,
                    s.DEF_FRCNN_BIASES_PATH)
                rpn.Rpn(features, info, namespace="rcnn")
                sess.run(tf.global_variables_initializer())
                return sess.run(["rcnn/rpn_bbox_pred/out:0"],
                        feed_dict={features: features_activations, info: self.im_info})
        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations["rpn_bbox_pred"])

    def test_proposals_layer(self):
        """Tests the proposal layer

        Reference activations for rpn_bbox_pred and rpn_cls_score are fed into the proposals
        network along with im_info and the regions of interest are compared with the reference
        activation blobs obtained from caffe
        """

        def runGraph(self):
            with tf.Session() as sess:
                # The actual RPN might not be run on the GPU except for conv layers
                score_activations = self.reference_activations['rpn_cls_score']
                bbox_activations = self.reference_activations['rpn_bbox_pred']

                features = tf.placeholder("float", [1, None, None, 512])
                info = tf.placeholder("float", [3])
                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH,
                        s.DEF_FRCNN_BIASES_PATH)
                rpn.Rpn(features, info, namespace="rcnn")
                sess.run(tf.global_variables_initializer())
                return sess.run(["rcnn/proposal_layer/proposal_regions:0"], feed_dict={
                    features: self.reference_activations['conv5_3'],  # isn't really used
                    'rcnn/rpn_cls_score/out:0': score_activations,
                    'rcnn/rpn_bbox_pred/out:0': bbox_activations,
                    info: self.im_info})

        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result,
            self.reference_activations['rois'][:, [1, 2, 3, 4]])



class sampleBoxesTest(unittest.TestCase):
    """Tests the sampleBoxes function"""

    def __init__(self, *args, **kwargs):
        super(sampleBoxesTest, self).__init__(*args, **kwargs)
        self.labeled_boxes = np.array(
            [[0, 1, 0, 1, 3],
             [4, 8, 16, 32, 0],
             [4, 6, 4, 9, 3],
             [2, 9, 1, 13, 2],
             [4, 1, 9, 27, 1],
             [7, 3, 9, 4, 3],
             [10, 3, 14, 9, 2],
             [3, 9, 3, 9, 2]]
        )

    def test_pos_neg_ratio(self):
        """Tests whether positive to negative examples are at most at a 1:1 ratio"""
        for mini_batch_size in range(2,9,2):
            pos_Idx, neg_Idx = rpn.sampleBoxes(self.labeled_boxes, 3, mini_batch_size)
            self.assertTrue(len(pos_Idx) <= len(neg_Idx),
                msg="num_pos={}, num_neg={}".format(len(pos_Idx), len(neg_Idx)))

    def test_neg_indices(self):
        """Tests whether negative indices are correctly identified"""
        for mini_batch_size in range(2,9,2):
            pos_Idx, neg_Idx = rpn.sampleBoxes(self.labeled_boxes, 3, mini_batch_size)
            for idx in neg_Idx:
                self.assertEqual(self.labeled_boxes[idx,4], 3,
                        msg=self.labeled_boxes[neg_Idx,4])

    def test_no_of_indices(self):
        """Tests whether the correct number of indices are being output"""
        for mini_batch_size, correct_size in zip(range(2, 9, 2), [2, 4, 6, 6]):
            pos_Idx, neg_Idx = rpn.sampleBoxes(self.labeled_boxes, 3, mini_batch_size)
            total_len = len(pos_Idx) + len(neg_Idx)
            self.assertEqual(total_len, correct_size,
                msg="Expected num_indices={}, but got {}, with batch size {}".format(
                    correct_size, total_len, mini_batch_size))

    def test_no_duplicates(self):
        """Tests whether the indices returned do not have duplicates"""
        for mini_batch_size in range(2, 9, 2):
            pos_Idx, neg_Idx = rpn.sampleBoxes(self.labeled_boxes, 3, mini_batch_size)
            all_Idx = np.concatenate((pos_Idx, neg_Idx))
            self.assertTrue(len(all_Idx)==len(set(all_Idx)),
                msg="Duplicates found.  List of indices, positive first is {}".format(all_Idx))



if __name__ == "__main__":
    unittest.main()
