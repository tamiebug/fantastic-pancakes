import tensorflow as tf
import numpy as np

import unittest

from .testUtils import eval_cpu, eval_gpu
from layers.custom_layers import roi_pooling_layer

# RoiPoolingLayer needs an image attributes input to extract a scale factor from
# These tests assume the scaling is 1, so we use the following
dummy_img_attr = np.array([1.,1.,1.])

def createDiagFeatures(width=16, height=16, channels=1, dtype=np.float32,
        asNumpyArray=False, inverted=(False,False,False)):
    ''' Creates a tf.constant tensor with consecutively numbered values '''
    if inverted[0]:
        w = np.arange(width-1, -1, -1, dtype=dtype)
    else:
        w = np.arange(width, dtype=dtype)
    if inverted[1]:
        h = np.arange(height-1, -1, -1, dtype=dtype)
    else:
        h = np.arange(height, dtype=dtype)
    if inverted[2]:
        c = np.arange(channels-1, -1, -1, dtype=dtype)
    else:
        c = np.arange(channels, dtype=dtype)

    out0 = w + width * h[:,np.newaxis]
    out = out0[:,:,np.newaxis] + width*height*c[np.newaxis, np.newaxis, :] 
    if not asNumpyArray:
        return tf.constant(out, dtype=dtype)
    else:
        return out

def assertArrayAlmostEqual(self, A, B, places=6, msg=""):
    B=np.array(B)
    self.assertEqual(A.shape, B.shape, 
            msg = "There is a shape mismatch.  Expected shape %s but got shape %s"%(str(A.shape),str(B.shape)))
    # Under assumption of equal shape, the following runs just fine
    it = np.nditer(A, flags=['multi_index'])
    while not it.finished:
        ind = it.multi_index
        self.assertAlmostEqual(A[ind], B[ind], places=places, 
                msg="At location %s, there is a mismatch.  We expected %d but got %d.\n "%(ind, B[ind], A[ind]) +
                    "Dumping entire comparison array: \n %s"%(str(A) + msg))
        it.iternext()
                    
# Quick shape references
# feature_in.shape = (height,width,channels)
# pooled_out.shape = (num_rois, pooled_h, pooled_w, channels)
# roi_list.shape   = (num_rois, 4) (the 4 dims are [x0, y0, x1, y1])

class shapesTest(tf.test.TestCase):
    
    def test_shape_A(self):
        regions = tf.constant([[0.,0.,2.,2.],[4.,6.,10.,10.]])
        feat_map = tf.random_normal((4,5,6))
        op = roi_pooling_layer(feat_map, dummy_img_attr, regions, 7, 8, 16)
        result = eval_cpu(op,self)
        self.assertEqual(result.shape, (2, 7, 8, 6),
                "expected %s, got %s"%(str((2,7,8,4)), str(result.shape)))

    def test_shape_B(self):
        regions = tf.constant([[33.,109.,204., 19,]])
        feat_map = createDiagFeatures(width=16,height=16,channels=1)
        op = roi_pooling_layer(feat_map, dummy_img_attr, regions, 2, 2, 16)
        result = eval_cpu(op,self)
        self.assertEqual(result.shape, (1,2,2,1), "Got %s"%str(result.shape))


class singleRegionOutputTest(tf.test.TestCase):
    def single_roi_test_template(self, features, expectation):       
        regions = tf.constant([[33.,19.,204.,109.]])
        op =  roi_pooling_layer(features, dummy_img_attr, regions, 2, 2, 16)
        result = eval_cpu(op, self)
        self.assertEqual(result.shape, (1, 2, 2, 1), "Shape incorrect.  Expected %s,\
            but got %s"%(str((1,2,2,1)),str(result.shape)))
        assertArrayAlmostEqual(self,result,expectation)

    def test_regular_input(self):
        features = createDiagFeatures(width=16, height=16, channels=1)
        expectation = [[[[71],[77]],[[119],[125]]]]
        self.single_roi_test_template(features, expectation)
    
    def test_horiz_inverted_input(self):
        features = createDiagFeatures(width=16,height=16, channels=1, inverted=(True, False, False))
        expectation = [[[[77],[71]],[[125],[119]]]]
        self.single_roi_test_template(features,expectation)

    def test_vert_inverted_input(self):
        features = createDiagFeatures(width=16, height=16, channels=1, inverted=(False, True, False))
        expectation = [[[[231],[237]],[[183],[189]]]]
        self.single_roi_test_template(features, expectation)

    def test_both_inverted_input(self):
        features = createDiagFeatures(width=16, height=16, channels=1, inverted=(True, True, False))
        expectation = [[[[237],[231]],[[189],[183]]]]
        self.single_roi_test_template(features, expectation)
    
if __name__ == '__main__':
   tf.test.main
