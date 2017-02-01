import numpy as np
import caffe
import tensorflow as tf

from util import frcnn_forward
from util import utils
from networks import vgg19
from util import settings as s
from test import testUtils
from layers.custom_layers import roi_pooling_layer
from networks import cls
from networks import loadNetVars
from networks import rpn

import threading
import os
import sys
import random
import unittest


class ModelOutputGenerator():
    """
    Singleton class that loads tensorflow and caffe models into memory for testing.
    Tensorflow is loaded using a separate thread to avoid locking the GPU
    """
    __metaclass__ = utils.Singleton

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
                    # Tensorflow does not know how to release /GPU:0 resources without process termination
                    # Issue #1727 in the tensorflow/tensorflow git repository
                    # Letting it just use CPU for this forward run instead

                    img = tf.placeholder(
                        "float", [1, 224, 224, 3], name="images")
                    model = vgg19.Vgg19()
                    model.buildGraph(img, train=False)

                    images = images.reshape((1, 224, 224, 3))

                    sess.run(tf.initialize_all_variables())
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
            transformer = caffe.io.Transformer(
                {'data': self.coffee.blobs['data'].data.shape})
            """
			transformer.set_transpose('data', (2,0,1)) 	# Move color channels to left-most dimension
			transformer.set_channel_swap('data', (2,1,0))	# Swap color channels from RGB to BGR
		
			transformed_image = transformer.preprocess('data', images)
			self.coffee.blobs['data'].data[...] = transformed_image
			"""
            images2 = np.transpose(images, [2, 0, 1])
            transformed_image = np.copy(images2[[2, 1, 0], :, :])
            self.coffee.blobs['data'].data[...] = transformed_image
            self.coffee.forward()

        # Running Caffe in its own thread to avoid conflicts
        utils.isolatedFunctionRun(runCaffe, True, self=self, images=images)
        print "We did runCaffe, yes!"

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

        # Caffe activations are in [channel, h, w] order, whereas TF activations are in [h, w, channel] order
        # Hence, we must transpose the activations by (0,2,3,1)
        transposeDict = {"relu1_1": (0, 2, 3, 1), 	"relu2_1": (0, 2, 3, 1),
                         "relu3_4": (0, 2, 3, 1),	"relu4_4": (0, 2, 3, 1),
                         "relu4_4": (0, 2, 3, 1),	"pool5": (0, 2, 3, 1),
                         "relu6": (0, 1),    "relu7": (0, 1),
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
        model = vgg19.Vgg19()
        model.buildGraph(img, train=False)

        testLayerVariables = []

        for layer in testLayers:
            testLayerVariables.append("vgg19/" + layer + "/Weights:0")
            testLayerVariables.append("vgg19/" + layer + "/Bias:0")

        # Find the weights and biases for several layers in a model
        try:
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
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
            model2 = vgg19.Vgg19()
            model2.buildGraph(img, train=False, weightsPath="models/" +
                              weightsFn + ".npz", biasesPath="models/" + biasFn + ".npz")
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
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

def array_equality_assert(self, nparray, ref_nparray, tolerance=.01):
    """
    Looks up activation in list of activations to see whether it is as expected.

    Assumes that self is a subclass of unittest.TestCase

    Parameters:
    self        -- subclass of unittest.TestCase on which asserts will be run.
    nparray     -- numpy array that is undergoing testing
    ref_nparray -- reference numpy array that nparray is being compared to
    tolerance   -- maximum amount of difference between the entries of nparray 
                        allowed before failing the test.
    """
    # Check for shape mismatches
    self.assertTrue(nparray.shape==ref_nparray.shape,
            msg="Unequal shapes.  Ref. shape is {}, array shape is {}".format(ref_nparray.shape, nparray.shape))
    greatest_diff = np.amax(np.absolute(ref_nparray - nparray))
    self.assertLessEqual(greatest_diff, tolerance, 
        msg="Greatest difference was %f" % greatest_diff)
    return greatest_diff <= tolerance


class Vgg16Test(unittest.TestCase):
    """
    Loads the modified vgg16 base model, which is lacking the final fully-connected layers, ending
    on relu5_3.  It will be tested on the image 000456.jpg in the images subfolder of test, 
    activations and shapes of conv4_1 and conv5_3 compared with the reference model.
    """

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.reference_activations = np.load(os.path.join(
            self.base_dir, "activations/test_values.npz"))


    def test_relu4_1(self):
        """ Tests whether the activations for relu4_1 match a given reference activation. """
        im_data, _ = frcnn_forward.process_image(
                os.path.join(self.base_dir, "images/000456.jpg"))
        im_data = np.expand_dims(im_data, axis=0)

        def runGraph(self, im):
            with tf.Session() as sess, tf.device("/gpu:0") as dev:
                img = tf.placeholder(
                    "float", im_data.shape, name="images")
                net = vgg19.Vgg19("vgg16test_1") 
                    # These are the layers of VGG19 we don't use when making a VGG16 network
                vgg16_cutoffs = ['conv3_4', 'relu3_4', 'conv4_4', 'relu4_4', 'conv5_4', 'relu5_4',
                                'pool5','fc6','relu6','fc7', 'relu7', 'fc8', 'prob']
                # Since we're only testing up to relu4_1, we can remove everything after it
                relu4_1_cutoffs = ['conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1',
                        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3']
                cutoffs = vgg16_cutoffs + relu4_1_cutoffs
                net.buildGraph(img, train=False,
                    weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
                    biasesPath=s.DEF_FRCNN_BIASES_PATH,
                    cutoff=cutoffs
                    )

                sess.run(tf.initialize_all_variables())
                output = sess.run(net.layers['relu4_1'], feed_dict={img:im_data})
                sess.close()
                # Does output come in list form if only one output is produced? [probably]
                # Blob name is conv4_1, not relu4_1; relu is done in-place by caffe
                return array_equality_assert(self, np.expand_dims(output[0],0), self.reference_activations['conv4_1'])

        return self.assertTrue( utils.isolatedFunctionRun( runGraph, False, self=self, im=im_data) )

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
            #config.log_device_placement = True
            with tf.Session(config=config) as sess, tf.device("/gpu:0") as dev:
                try:
                    conv4_1_in = self.reference_activations['conv4_1']
                except KeyError:
                    print("Warning:  conv4_1 not found in reference_activations.  Something \
                            wrong with .npz file")
                conv4_1 = tf.placeholder(
                       "float", conv4_1_in.shape, name="conv4_1")


                vgg16_cutoffs = ['conv3_4', 'relu3_4', 'conv4_4', 'relu4_4', 'conv5_4', 'relu5_4',
                                'pool5','fc6','relu6','fc7', 'relu7', 'fc8', 'prob']
                before_relu4_1_cutoffs = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3' , 'relu3_3', 'pool3',
                        'conv4_1', 'relu4_1']

                cutoffs = vgg16_cutoffs + before_relu4_1_cutoffs
                net = vgg19.Vgg19(namespace="vgg16test_2") 
                net.buildGraph(conv4_1, train=False,
                    weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
                    biasesPath=s.DEF_FRCNN_BIASES_PATH,
                    cutoff=cutoffs
                    )
                sess.run(tf.initialize_all_variables())
                output = sess.run(net.layers['relu5_3'], feed_dict={conv4_1 : conv4_1_in})
                sess.close()
                return array_equality_assert(self, np.expand_dims(output[0],0), self.reference_activations['conv5_3'])

        return self.assertTrue(utils.isolatedFunctionRun(runGraph, False, self=self))

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
                sess.run(tf.initialize_all_variables())
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
                sess.run(tf.initialize_all_variables())
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
                sess.run(tf.initialize_all_variables())
                return sess.run(["rcnn/bbox_pred/out:0"], feed_dict={"rcnn/relu7:0" : relu7_in})

        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations['bbox_pred'])

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

                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH, s.DEF_FRCNN_BIASES_PATH)
                net = rpn.Rpn(features, info, namespace="rcnn")
                #for op in tf.get_default_graph().get_operations():
                #    print op.name
                sess.run(tf.initialize_all_variables())
                return sess.run(["rcnn/rpn_cls_score/out:0"], feed_dict={
                    features : features_activations, info : self.im_info})

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
                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH, s.DEF_FRCNN_BIASES_PATH)
                net = rpn.Rpn(features, info, namespace="rcnn")
                sess.run(tf.initialize_all_variables())
                return sess.run(["rcnn/rpn_bbox_pred/out:0"], 
                        feed_dict={features : features_activations, info : self.im_info})
        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations["rpn_bbox_pred"])

    def test_proposals_layer(self):
        """ 
        Tests the proposal layer
            
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
                loadNetVars.extractLayers("rcnn", s.DEF_FRCNN_WEIGHTS_PATH, s.DEF_FRCNN_BIASES_PATH)
                net = rpn.Rpn(features, info, namespace="rcnn")
                sess.run(tf.initialize_all_variables())
                return sess.run(["rcnn/proposal_regions:0"], feed_dict={
                    features : self.reference_activations['conv5_3'], # isn't really used
                    'rcnn/rpn_cls_score/out:0' : score_activations,
                    'rcnn/rpn_bbox_pred/out:0' : bbox_activations,
                    info            : self.im_info})

        result = utils.isolatedFunctionRun(runGraph, False, self)[0]
        return array_equality_assert(self, result, self.reference_activations['rois'][:,[1,2,3,4]])

                
if __name__ == '__main__':
    unittest.main()
