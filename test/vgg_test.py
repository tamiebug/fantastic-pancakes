import numpy as np
import tensorflow as tf

from util import frcnn_forward
from util import utils
from networks import vgg
from util import settings as s

import os
import random
import unittest


REL_TOL = 1e-3
ABS_TOL = 1e-3

class Vgg16SavingTest(tf.test.TestCase):
    """
    Loads the vgg16 base model, saves it, then reloads it once more in order to test that the
    weights and biases are being properly saved and loaded
    """

    def testSave(self):
        # Layers to be tested
        testLayers = ['conv1_1', 'conv2_1', 'conv3_3', 'conv4_3', 'fc6', 'fc7']

        img = tf.placeholder(tf.float32, [1, 224, 224, 3], name="images")
        model = vgg.VGG("frcnn")
        model.buildGraph(img, network_version="VGG16",
            weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
            biasesPath=s.DEF_FRCNN_BIASES_PATH)

        testLayerVariables = []

        for layer in testLayers:
            testLayerVariables.append("frcnn/" + layer + "/Weights:0")
            testLayerVariables.append("frcnn/" + layer + "/Bias:0")


        # Find the weights and biases for several layers in a model
        weightsFn = None
        biasFn = None
        try:
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                output = sess.run(testLayerVariables)
                randNum = None
                while True:
                    # Try different file names until you find one that isn't taken
                    randNum = random.random()
                    weightsFn = str(randNum) + "testWeights"
                    biasFn = str(randNum) + "testBias"
                    if os.path.isfile("models/" + weightsFn) or os.path.isfile("models/" + biasFn):
                        continue
                    else:
                        model.save(weightsFn, biasFn)
                        break
            # Load the weights/biases into a new model
            model2 = vgg.VGG()
            model2.buildGraph(img, train=False,
                weightsPath="models/" + weightsFn + ".npz",
                biasesPath="models/" + biasFn + ".npz")
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                output2 = sess.run(testLayerVariables)
            self.assertEqual(len(output2), 2 * len(testLayers),
                             msg="Incorrect number of output layers")
            # Check to make sure that the values are the same
            for i, var in enumerate(output2):
                np.testing.assert_equal(
                    output[i], output2[i], err_msg="Output number %i was not equal" % i)
        finally:
            if weightsFn is not None and os.path.isfile("models/" + weightsFn + ".npz"):
                os.remove("models/" + weightsFn + ".npz")
            if biasFn is not None and os.path.isfile("models/" + biasFn + ".npz"):
                os.remove("models/" + biasFn + ".npz")


class Vgg16Test(tf.test.TestCase):
    """
    Loads the modified vgg16 base model, which is lacking the final fully-connected layers, ending
    on relu5_3.  It will be tested on the image 000456.png in the images subfolder of test,
    activations and shapes of conv4_1 and conv5_3 compared with the reference model.
    """

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.reference_activations = np.load(os.path.join(
            self.base_dir, "activations/test_values.npz"))

    def test_relu4_1(self):
        """ Tests whether the activations for relu4_1 match a given reference activation. """
        im_data, _ = frcnn_forward.process_image(
            os.path.join(self.base_dir, "images/000456.png"))
        im_data = np.expand_dims(im_data, axis=0)

        def runGraph(self, im):
            with self.test_session() as sess, tf.device("/gpu:0"):
                img = tf.placeholder(tf.float32, im_data.shape, name="images")
                net = vgg.VGG("vgg16test_1")
                net.buildGraph(img, train=False,
                    weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
                    biasesPath=s.DEF_FRCNN_BIASES_PATH,
                    network_version="VGG16CONV"
                               )

                sess.run(tf.global_variables_initializer())
                output = sess.run(net.layers['relu4_1'], feed_dict={img: im_data})
                sess.close()
                # Does output come in list form if only one output is produced? [probably]
                # Blob name is conv4_1, not relu4_1; relu is done in-place by caffe
                return output

        output = utils.isolatedFunctionRun(runGraph, False, self=self, im=im_data)
        return self.assertAllClose(output, self.reference_activations['conv4_1'], REL_TOL, ABS_TOL)

    def test_relu5_3(self):
        """
        Tests whether the activations for relu5_3 match a given reference activation.

        The testing is done with the network being fed in the expected activations at
        relu4_1, allowing us to unit test this section of network.  Reason for this
        particular test is that the VGG16 network's activations may be too large in
        tensorflow, requiring the network to be broken up
        """

        def runGraph(self):
            with self.test_session() as sess, tf.device("/gpu:0"):
                try:
                    conv4_1_in = self.reference_activations['conv4_1']
                except KeyError:
                    print("Warning:  conv4_1 not found in reference_activations.  Something \
                            wrong with .npz file")
                net = vgg.VGG(namespace="vgg16test_2")
                net.buildGraph(tf.placeholder(dtype=tf.float32), train=False,
                    weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
                    biasesPath=s.DEF_FRCNN_BIASES_PATH,
                    network_version="VGG16CONV"
                               )
                conv4_1 = net.layers['conv4_1']
                sess.run(tf.global_variables_initializer())
                output = sess.run(net.layers['relu5_3'], feed_dict={conv4_1: conv4_1_in})
                sess.close()
                return output

        output = utils.isolatedFunctionRun(runGraph, False, self=self)
        return self.assertAllClose(output, self.reference_activations['conv5_3'], REL_TOL, ABS_TOL)

    def test_whole_network(self):
        """ Tests whether the activations for relu5_3 match a given reference activation"""
        im_data, _ = frcnn_forward.process_image(
            os.path.join(self.base_dir, "images/000456.png"))
        im_data = np.expand_dims(im_data, axis=0)

        def runGraph(self, im):
            with self.test_session() as sess, tf.device("/gpu:0"):
                img = tf.placeholder(tf.float32, im_data.shape, name="images")
                net = vgg.VGG("vgg16test_3")
                # These are the layers of VGG16 we don't use when making a VGG16 network
                net.buildGraph(img, train=False,
                    weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
                    biasesPath=s.DEF_FRCNN_BIASES_PATH,
                    network_version="VGG16CONV"
                               )

                sess.run(tf.global_variables_initializer())
                output = sess.run(net.layers['relu5_3'], feed_dict={img: im_data})
                sess.close()
                # Does output come in list form if only one output is produced? [probably]
                # Blob name is conv4_1, not relu4_1; relu is done in-place by caffe
                return output

        output = utils.isolatedFunctionRun(runGraph, False, self=self, im=im_data)
        return self.assertAllClose(output, self.reference_activations['conv5_3'], REL_TOL, ABS_TOL)


if __name__ == '__main__':
    unittest.main()
