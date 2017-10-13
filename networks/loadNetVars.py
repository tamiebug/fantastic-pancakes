import numpy
import tensorflow as tf
import util.settings as s
from util.utils import easy_scope

def _fixModelPath(path):
    # Ensures that sloppily constructed model paths still work
    if not path.endswith(".npz"):
        return path + ".npz"
    else:
        return path

def extractLayers(scope, weightsPath, biasesPath, device="/cpu:0"):
    """
    Function that extracts weights and biases into properly scoped variables.
    
    Positional arguments:
    scope -- tf.VariableScope ( or string representing a scope ) to place variables in
    weightsPath -- path to .npz file containing the weights
    biasesPath -- path to .npz file containing the biases

    Keyword arguments:
    device -- device on which to place the created variables
    """

    weightsPath = _fixModelPath(weightsPath)
    biasesPath = _fixModelPath(biasesPath)

    # Raw numpy values.  Need to be loaded into variables.
    weightsDict = numpy.load(weightsPath)
    biasesDict = numpy.load(biasesPath)

    # Here, we do a for loop looping through all of the names, "name".
    with tf.device(device) as dev:
        with easy_scope(scope) as model_scope:
            warning = False
            for name, weights_tnsr in weightsDict.items():    
                if name.startswith("/"):
                    name = name[1:]
                if name.endswith("/"):
                    name = name[:-1]
                with easy_scope(name) as layer_scope:
                    try:
                        tf.get_variable("Weights", trainable=False, initializer=tf.constant(weights_tnsr))
                        tf.get_variable("Bias", trainable=False, initializer=tf.constant(biasesDict[name]))
                    except ValueError:
                        # Values were loaded elsewhere
                        warning = True
            if warning:
                print(("extractLayers()  Warning : Some variable names already exist."+
                        "  If unintentional, please choose a different scope name."))
            
    return
