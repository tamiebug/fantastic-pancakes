import unittest
import tensorflow as tf
import numpy as np
import caffe

from util import utils
import util.settings as s
from networks import vgg


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

                    img = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name="images")
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
