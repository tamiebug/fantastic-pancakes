import numpy
import tensorflow as tf
import util.settings as s

def _fixModelPath(path):
    # Ensures that sloppily constructed model paths still work
    if not path.endswith(".npz"):
        return path + ".npz"
    else:
        return path

def extractLayers(scope, weightsPath, biasesPath):
    # Loads parameters from .npz files, one for the weights and another for the biases
    # Assumes that the proper variable names exist in the given scope somewhere
    weightsPath = _fixModelPath(weightsPath)
    biasesPath = _fixModelPath(biasesPath)

    # Raw numpy values.  Need to be loaded into variables.
    weightsDict = numpy.load(weightsPath)
    biasesDict = numpy.load(biasesPath)

    # Here, we do a for loop looping through all of the names, "name".
    for name, weights_tnsr in weightsDict.iteritems():
        with tf.variable_scope(scope) as model_scope:
            with tf.variable_scope(name) as layer_scope:		
                tf.get_variable("Weights", trainable=True, initializer=tf.constant(weights_tnsr))

    for name, biases_tnsr in biasesDict.iteritems():
        with tf.variable_scope(scope) as model_scope:
            with tf.variable_scope(name) as layer_scope:
                tf.get_variable("Bias", trainable=True, initializer=tf.constant(biases_tnsr))
    return
