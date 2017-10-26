import tensorflow as tf
import numpy as np
from test.testUtils import eval_cpu
from layers.custom_layers import iou_labeler

# Note: iou_labeler labels boxes in the last column as follows
# POSITIVE = 1
# NEGATIVE = -1
# NEITHER = 0


class singleRegionOutputTest(tf.test.TestCase):

    def single_gt_template(self, threshold_neg, threshold_pos, expectation):
        """Test to see if iou_labeler works properly in the binary case"""
        regions = tf.constant([[0, 0, 1, 1], [4, 6, 9, 9], [5, 5, 7, 7]], dtype=tf.float32)
        ground_truth_box = tf.constant([[3, 5, 7, 7]], dtype=tf.float32)
        op = iou_labeler(regions, ground_truth_box, threshold_neg, threshold_pos)
        result = eval_cpu(op, self)
        self.assertAllClose(result, expectation)

    def test_single_gt_A(self):
        expectation = np.array([[0, 0, 1, 1, -1], [4, 6, 9, 9, 0], [5, 5, 7, 7, 1]],
                dtype=np.float32)
        self.single_gt_template(.2, .5, expectation)

    def test_single_gt_B(self):
        expectation = np.array([[0, 0, 1, 1, -1], [4, 6, 9, 9, -1], [5, 5, 7, 7, 1]],
                dtype=np.float32)
        self.single_gt_template(.3, .5, expectation)

    def test_single_gt_C(self):
        expectation = np.array([[0, 0, 1, 1, -1], [4, 6, 9, 9, 0], [5, 5, 7, 7, 1]],
                dtype=np.float32)
        self.single_gt_template(.2, .7, expectation)

    def multiple_gt_template(self, threshold_neg, threshold_pos, expectation):
        """Test to see if iou_labeler works properly in the general case"""
        regions = tf.constant(
            [[0, 0, 1, 1],
            [4, 6, 9, 9],
            [5, 5, 7, 7],
            [5, 0, 7, 2],
            [3, 5, 7, 6]],
            dtype=tf.float32)
        gt_boxes = tf.constant([[3, 5, 7, 7], [1, 1, 6, 6], [6, 6, 8, 8]], dtype=tf.float32)
        op = iou_labeler(regions, gt_boxes, threshold_neg, threshold_pos)
        # Only going to look at the classification index this time to save typing
        result = eval_cpu(op, self)[:, 4]
        self.assertAllClose(result, expectation)

    def test_iou_free_pos(self):
        """Tests if the region with highest IoU overlap w/ a g.t. box gets a free positive"""
        expectation = np.array([-1, 1, 0, 0, 1], dtype=np.float32)
        self.multiple_gt_template(.03, .99, expectation)

    def test_iou_B(self):
        expectation = np.array([-1, 1, 1, -1, 1], dtype=np.float32)
        self.multiple_gt_template(.2, .59, expectation)

    def test_iou_C(self):
        expectation = np.array([-1, 1, 0, -1, 1], dtype=np.float32)
        self.multiple_gt_template(.37, .61, expectation)
