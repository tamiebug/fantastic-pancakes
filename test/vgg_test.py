import numpy as np
import caffe
import tensorflow as tf

from util import frcnn_forward
from util import utils
from networks import vgg
from util import settings as s
from test.testUtils import array_equality_assert

import os
import random
import unittest


class ModelOutputGenerator(metaclass=utils.Singleton):
    """
    Singleton class that loads tensorflow and caffe models into memory for testing.
    Tensorflow is loaded using a separate thread to avoid locking the GPU
    """

    def __init__(self):
        # Layer activations that we are going to test
        self.testLayers = ['relu1_1', 'relu2_1', 'relu3_4',
                           'relu4_4', 'pool5', 'relu6', 'relu7', 'prob']
        self.caffeTestLayers = ['conv1_1', 'conv2_1',
                                'conv3_4', 'conv4_4', 'pool5', 'fc6', 'fc7', 'prob']
        self.output = None
        # Images we are testing with
        images = utils.loadImage(s.DEF_TEST_IMAGE_PATHS[0])

        # Set up Tensorflow Model
        def runTensorflow(self, images):
            with tf.Session() as sess:
                with tf.device("/cpu:0"):
                    # Tensorflow does not know how to release /GPU:0 resources
                    # without process termination
                    # Issue #1727 in the tensorflow/tensorflow git repository
                    # Letting it just use CPU for this forward run instead

                    img = tf.placeholder(
                        "float", [1, 224, 224, 3], name="images")
                    model = vgg.VGG()
                    model.buildGraph(img, network_version="VGG19")

                    images = images.reshape((1, 224, 224, 3))

                    sess.run(tf.global_variables_initializer())
                    # print self.model.layers.keys()
                    self.output = sess.run([model.layers[_] for _ in self.testLayers],
                                           feed_dict={img: images})

                    sess.close()

            return

        # Running Tensorflow in its own thread allows it to release control of
        # the GPU
        utils.isolatedFunctionRun(
            runTensorflow, True, self=self, images=images)

        # Set up Caffe Model for comparison
        def runCaffe(self, images):
            if s.CAFFE_USE_CPU:
                caffe.set_mode_cpu()
            self.coffee = caffe.Net(
                s.DEF_PROTOTXT_PATH, s.DEF_CAFFEMODEL_PATH, caffe.TEST)
            """
            transformer = caffe.io.Transformer(
                {'data': self.coffee.blobs['data'].data.shape})
            transformer.set_transpose('data', (2,0,1))
            # Move color channels to left-most dimension
            transformer.set_channel_swap('data', (2,1,0))
            # Swap color channels from RGB to BGR
            transformed_image = transformer.preprocess('data', images)
            self.coffee.blobs['data'].data[...] = transformed_image
            """
            images2 = np.transpose(images, [2, 0, 1])
            transformed_image = np.copy(images2[[2, 1, 0], :, :])
            self.coffee.blobs['data'].data[...] = transformed_image
            self.coffee.forward()

        # Running Caffe in its own thread to avoid conflicts
        utils.isolatedFunctionRun(runCaffe, True, self=self, images=images)
        print("We did runCaffe, yes!")

    def returnBlob(self, layername, flavor):
        """
        Returns a layer of name layername in either the tf or caffe model.
        """
        if flavor == "caffe":
            caffeLayer = self.caffeTestLayers[self.testLayers.index(layername)]
            return self.coffee.blobs[caffeLayer].data
        elif flavor == "tensorflow":
            return self.output[self.testLayers.index(layername)]
        else:
            raise KeyError(
                "Caffe and tensorflow are the only allowed blob flavors")


class Vgg19LayerActivationsTest(unittest.TestCase):

    def setUp(self):
        self.model = ModelOutputGenerator()
        return

    def blob_tensor_equality_assert(self, name, tolerance=.01, testingChannels=[0]):
        # Pass an empty list to testingChannels to test all of them

        # Caffe activations are in [channel, h, w] order,
        # whereas TF activations are in [h, w, channel] order
        # Hence, we must transpose the activations by (0,2,3,1)
        transposeDict = {"relu1_1": (0, 2, 3, 1), "relu2_1": (0, 2, 3, 1),
                         "relu3_4": (0, 2, 3, 1), "relu4_4": (0, 2, 3, 1),
                         "relu4_4": (0, 2, 3, 1), "pool5": (0, 2, 3, 1),
                         "relu6": (0, 1), "relu7": (0, 1),
                         "prob": (0, 1)}

        if testingChannels == []:
            blob_data = self.model.returnBlob(
                name, "caffe").transpose(transposeDict[name])
            tensor_data = self.model.returnBlob(name, "tensorflow")
        else:
            blob_data = self.model.returnBlob(
                name, "caffe")[testingChannels].transpose(transposeDict[name])
            tensor_data = self.model.returnBlob(
                name, "tensorflow")[testingChannels]

        greatest_diff = np.amax(np.absolute(blob_data - tensor_data))
        self.assertLessEqual(greatest_diff, tolerance,
                             msg="Greatest difference was %f" % greatest_diff)

    def test_relu1_1(self):
        self.blob_tensor_equality_assert('relu1_1', .01, [])

    def test_relu2_1(self):
        self.blob_tensor_equality_assert('relu2_1', .01, [0])

    def test_relu3_4(self):
        self.blob_tensor_equality_assert('relu3_4', .01, [0])

    def test_relu4_4(self):
        self.blob_tensor_equality_assert('relu4_4', .01, [0])

    def test_pool5(self):
        self.blob_tensor_equality_assert('pool5', .01, [0])

    def test_relu6(self):
        self.blob_tensor_equality_assert('relu6', .01, [0])

    def test_relu7(self):
        self.blob_tensor_equality_assert('relu7', .01, [0])

    def test_prob(self):
        self.blob_tensor_equality_assert('prob', .01, [0])


class Vgg19SavingTest(unittest.TestCase):
    """
    Loads the vgg19 base model, saves it, then reloads it once more in order to test that the
    weights and biases are being properly saved and loaded
    """

    def testSave(self):
        # Layers to be tested
        testLayers = ['conv1_1', 'conv2_1', 'conv3_4', 'conv4_4', 'fc6', 'fc7']
        img = tf.placeholder("float", [1, 224, 224, 3], name="images")
        model = vgg.VGG("vgg19")
        model.buildGraph(img, network_version="VGG19")

        testLayerVariables = []

        for layer in testLayers:
            testLayerVariables.append("vgg19/" + layer + "/Weights:0")
            testLayerVariables.append("vgg19/" + layer + "/Bias:0")

        # Find the weights and biases for several layers in a model
        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                output = sess.run(testLayerVariables)
                randNum = None
                # Save the weights/biases in a file
                while True:
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
            model2.buildGraph(img, train=False, weightsPath="models/" +
                              weightsFn + ".npz", biasesPath="models/" + biasFn + ".npz")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                output2 = sess.run(testLayerVariables)
            self.assertEqual(len(output2), 2 * len(testLayers),
                             msg="Incorrect number of output layers")
            # Check to make sure that the values are the same
            for i, var in enumerate(output2):
                np.testing.assert_equal(
                    output[i], output2[i], err_msg="Output number %i was not equal" % i)
        finally:
            if os.path.isfile("models/" + weightsFn + ".npz"):
                os.remove("models/" + weightsFn + ".npz")
            if os.path.isfile("models/" + biasFn + ".npz"):
                os.remove("models/" + biasFn + ".npz")


class Vgg16Test(unittest.TestCase):
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
            with tf.Session() as sess, tf.device("/gpu:0"):
                img = tf.placeholder(
                    "float", im_data.shape, name="images")
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
                return array_equality_assert(self, np.expand_dims(output, 0),
                    self.reference_activations['conv4_1'])

        return self.assertTrue(utils.isolatedFunctionRun(runGraph, False, self=self, im=im_data))

    def test_relu5_3(self):
        """
        Tests whether the activations for relu5_3 match a given reference activation.

        The testing is done with the network being fed in the expected activations at
        relu4_1, allowing us to unit test this section of network.  Reason for this
        particular test is that the VGG16 network's activations may be too large in
        tensorflow, requiring the network to be broken up
        """

        def runGraph(self):
            config = tf.ConfigProto()
            # config.log_device_placement = True
            with tf.Session(config=config) as sess, tf.device("/gpu:0"):
                try:
                    conv4_1_in = self.reference_activations['conv4_1']
                except KeyError:
                    print("Warning:  conv4_1 not found in reference_activations.  Something \
                            wrong with .npz file")
                net = vgg.VGG(namespace="vgg16test_2")
                net.buildGraph(tf.placeholder(), train=False,
                    weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
                    biasesPath=s.DEF_FRCNN_BIASES_PATH,
                    network_version="VGG16CONV"
                               )
                conv4_1 = net.layers['conv4_1']
                sess.run(tf.global_variables_initializer())
                output = sess.run(net.layers['relu5_3'], feed_dict={conv4_1: conv4_1_in})
                sess.close()
                return array_equality_assert(self, np.expand_dims(output, 0),
                        self.reference_activations['conv5_3'])

        return self.assertTrue(utils.isolatedFunctionRun(runGraph, False, self=self))

    def test_whole_network(self):
        """ Tests whether the activations for relu5_3 match a given reference activation"""
        im_data, _ = frcnn_forward.process_image(
            os.path.join(self.base_dir, "images/000456.png"))
        im_data = np.expand_dims(im_data, axis=0)

        def runGraph(self, im):
            with tf.Session() as sess, tf.device("/gpu:0"):
                img = tf.placeholder(
                    "float", im_data.shape, name="images")
                net = vgg.VGG("vgg16test_3")
                # These are the layers of VGG19 we don't use when making a VGG16 network
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
                return array_equality_assert(self, np.expand_dims(output, 0),
                    self.reference_activations['conv5_3'])

        return self.assertTrue(utils.isolatedFunctionRun(runGraph, False, self=self, im=im_data))


if __name__ == '__main__':
    unittest.main()
