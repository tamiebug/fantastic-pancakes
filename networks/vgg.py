import numpy
import tensorflow as tf
import collections

import util.settings as s
from . import loadNetVars
from util.utils import easy_scope


class VGG():
    def __init__(self, namespace="vgg"):
        self.namespace = namespace
        self.vgg19LayerNames = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'fc6', 'relu6', 'fc7', 'relu7', 'prob']
        self.prevLayer = None
        self.layers = collections.OrderedDict()

    def addLayer(self, name, trainable=False):
        """
        Adds a layer to the computation graph.

        This function makes two assumptions:
            1.  This function assumes that self.prevLayer is not None.  Thus, prevLayer
            must be set to the output of a previous Tensorflow Op (Possibly a
            tf.Placeholder for input to VGG, for example.)

            2.  This function assumes that all weights are loaded and properly
            namespaced by the layer name;  The weights and biases for a layer named
            conv5_3 must be variables already loaded in memory and namespaced in
            Tensorflow by conv5_3/Weights and conv5_3/Biases, respecitvely.  If these
            variables do not already exist before calling this function, it will
            raise an error.
        """

        assert (self.prevLayer is not None), ("VGG Internal Error:  Initial prevLayer"
                "not set before calling addLayer for the first time.")

        if name.startswith('conv'):
            self.layers[name] = self.createConvLayer(name, trainable=trainable)
        elif name.startswith('relu'):
            self.layers[name] = tf.nn.relu(self.prevLayer, name)
        elif name.startswith('pool'):
            self.layers[name] = tf.nn.max_pool(self.prevLayer, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME', name=name)
        elif name.startswith('fc'):
            if name is 'fc6':
                self.layers[name] = self.createFirstFcLayer(
                    name, trainable=trainable)
            else:
                self.layers[name] = self.createFcLayer(name, trainable=trainable)
        elif name.startswth('drop'):
            self.layers[name] = tf.nn.dropout(self.prevLayer,
                    0.5, name=name)
        elif name == "prob":
            self.layers[name] = tf.nn.softmax(
                self.prevLayer, name=name)
        else:
            raise ValueError("{} is not a recognized layer name.  All valid names"
                    " either start with the string conv, relu, pool, fc, drop, "
                    " or prob, indicating a convolutional, ReLU, max pooling, "
                    " fully connected, dropout, or softmax layer, respectively. ")

        # Cannot forget: Must set prevLayer to just-added layer
        self.prevLayer = self.layers[name]

    def makeLayout(self, name):
        """ Constructs the layout in a list of dicts to be used by buildGraph """

        retList = []

        if name is "VGG16":
            for layerName in self.vgg19LayerNames:
                if (layerName is 'conv3_4' or layerName is 'conv4_4'
                        or layerName is 'conv5_4' or layerName is 'relu3_4'
                        or layerName is 'relu4_4' or layerName is 'relu5_4'):
                    continue
                else:
                    retList.append({"name": layerName,
                        "trainable": None,
                        "device": None})
        elif name is "VGG19":
            for layerName in self.vgg19LayerNames:
                retList.append({"name": layerName,
                    "trainable": None,
                    "device": None})
        elif name is "VGG16CONV":
            for layerName in self.vgg19LayerNames:
                if (layerName is 'conv3_4' or layerName is 'conv4_4'
                        or layerName is 'conv5_4' or layerName is 'relu3_4'
                        or layerName is 'relu4_4' or layerName is 'relu5_4'):
                    continue
                elif layerName is 'relu5_3':
                    retList.append({"name": layerName,
                        "trainable": None,
                        "device": None})
                    break
                else:
                    retList.append({"name": layerName,
                        "trainable": None,
                        "device": None})
        elif name is "VGG19CONV":
            for layerName in self.vgg19LayerNames:
                if layerName is 'relu5_4':
                    retList.append({"name": layerName,
                        "trainable": None,
                        "device": None})
                    break
                else:
                    retList.append({"name": layerName,
                        "trainable": None,
                        "device": None})
        else:
            raise KeyError("Name {} not found in list of network_version's".format(name))

        return retList

    def createConvLayer(self, name, trainable=True):
        """Creates a convolutional Tensorflow layer given its name.

        Assumes that properly named bias and weight variables are already loaded in memory
        """

        with easy_scope(name):
            conv = tf.nn.conv2d(self.prevLayer, tf.get_variable(
                "Weights", trainable=trainable), [1, 1, 1, 1], padding="SAME")
            bias = tf.nn.bias_add(conv, tf.get_variable("Bias", trainable=trainable))
        return bias

    def createFirstFcLayer(self, name, trainable=True):
        """Creates the first fully connected layer

        This layer converts the  output of the last convolutional layer to the
        input for the next fully connected ones.  Returns the bias layer.
        """

        INPUT_SIZE = 25088
        # OUTPUT_SIZE = 4096

        with easy_scope(name, reuse=True):
            flattenedInput = tf.reshape(self.prevLayer, [-1, INPUT_SIZE])
            layer = tf.nn.bias_add(
                tf.matmul(flattenedInput, tf.get_variable("Weights", trainable=trainable)),
                tf.get_variable("Bias", trainable=trainable))

        return layer

    def createFcLayer(self, name, trainable=True):
        """Creates a fully connected layer

        Loads the weights from the weightsDict and biasesDict dictionaries using
        their key value name and returns the bias layer.
        """

        with easy_scope(name, reuse=True):
            layer = tf.nn.bias_add(
                tf.matmul(self.prevLayer, tf.get_variable("Weights", trainable=trainable)),
                tf.get_variable("Bias", trainable=trainable))

        return layer

    def buildGraph(self, prevLayer, train=False, train_starting_at=None,
            weightsPath=s.DEF_WEIGHTS_PATH, biasesPath=s.DEF_BIASES_PATH,
            network_version="VGG16", device="/gpu:0", custom_layout=None):
        """Builds up the computation graph based on the given parameters.

        Positional arguments:
            prevLayer -- VGG must be connected to the output of another op.
                When making a vgg16 network, for example, the input may be
                a tf.Placeholder, as is usually the case.

        Keyword arguments:
            train -- If True, sets all variables in the created computation
                graph to be trainable.  If False(default), then it sets them
                to not be trainable, unless overriden by a following option.
            train_starting_at -- If set to the name of a layer, sets all
                layers to trainable after and including that layer.  Overrides
                "train" keyword argument.
            weightsPath -- Path to .npz file containing properly namespaced
                weights for this network.  See loadNetVars for how to properly
                do this.
            biasesPath -- Path to .npz file containing properly namespaced
                biases for this network.  See above
            network_version -- If it is desired to, for example, create
                a VGG16 network, the default "VGG16" suffices.  The options
                are "VGG16", "VGG19", "VGG16CONV", and "VGG19CONV".  The
                latter two have as their last layers the last convolutional
                outputs of their respective CNNs
            device -- The device onto which to place all operations.  By default
                set to "/gpu:0"; running convolutions on CPUs is not fun.
            custom_layout -- In case one desires to make a custom VGG-like
                convolutional neural network, the exact layout of the neural network
                can be provided in the internally used format.  See class method
                makeLayout(self, name) above for an example
        """

        # Extracts the information from .npz files and puts them into properly
        # scoped tf.Variable(s)
        loadNetVars.extractLayers(self.namespace, weightsPath, biasesPath)

        # Set up the network layout
        layout = []
        if network_version is not None:
            _layout = self.makeLayout(network_version)

            # Set default device and trainability for layers
            for layer in _layout:
                layer["device"] = device
                layer["trainable"] = train

            # Set trainability when using train_starting_at
            if train_starting_at is not None:
                setTrainingTrue = False
                for layer in layout:
                    if layer["name"] is train_starting_at:
                        setTrainingTrue = True
                    layer["trainable"] = setTrainingTrue

            # Set dropout for any fully connected layers being trained
            for layer in _layout:
                if layer["name"].startswith('fc') and layer["trainable"] is True:
                    dropoutLayer = {}
                    dropoutLayer["name"] = layer["name"].replace('fc', 'drop', 1)
                    dropoutLayer["device"] = device
                    dropoutLayer["trainable"] = None
                    layout.apppend(dropoutLayer)
                    layout.append(layer)
                else:
                    # No dropout needed
                    layout.append(layer)

        # In the case of a custom layout passed in
        if custom_layout is not None:
            layout = custom_layout

        # Actualize the layout
        with easy_scope(self.namespace, reuse=True):
            self.prevLayer = prevLayer
            for layer in layout:
                with tf.device(layer["device"]):
                    self.addLayer(layer["name"], trainable=layer["trainable"])

        print("VGG computational graph successfully actualized!  See layers"
            " attribute to inspect its ops.")

        return self.prevLayer

    def save(self, weightsName, biasesName, overwrite=False):
        """Saves the current weights and biases for the model

        They are saved as files named weightsName and biasesName, respectively,
        in the models/ directory.  If a file with this name already exists,
        will raise an error unless overwrite=True, in which case it will overwrite
        the files.
        """

        weightsVar = []
        biasVar = []
        for var in tf.global_variables():
            if var.name.startswith(self.namespace):
                if "Weights" in var.name:
                    weightsVar.append(var)
                if "Bias" in var.name:
                    biasVar.append(var)
        fullVarList = biasVar + weightsVar
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(fullVarList)

        theDict = {var.name: output[i] for i, var in enumerate(fullVarList)}
        weightsDict = {}
        biasDict = {}
        for key, val in theDict.items():
            if "Weights" in key:
                weightsDict[key.split(self.namespace)
                            [-1].split('/Weights:0')[0]] = val
            if "Bias" in key:
                biasDict[key.split(self.namespace)
                         [-1].split('/Bias:0')[0]] = val
        numpy.savez("models/" + weightsName, **weightsDict)
        numpy.savez("models/" + biasesName, **biasDict)
