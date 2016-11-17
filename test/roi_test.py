import tensorflow as tf
import numpy as np

import unittest

from testUtils import eval_cpu, eval_gpu
from custom_layers import roi_pooling_layer

class BasicThingTest(tf.test.TestCase):
    
    def test_output_shape(self):
        regions = tf.constant([[0.,0.,2.,2.,], [4.,6.,10.,10.]])
        feat_map = tf.random_normal((14,14))
        result = eval_cpu(roi_pooling_layer(regions,feat_map),self)
        self.assertEqual(result.shape, (14,14))
        return

if __name__ == '__main__':
    tf.test.main()
