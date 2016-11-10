import numpy
import tensorflow as tf
import settings as s

def extractLayers(scope, weightsPath, biasesPath):
	# Loads parameters from .npz files, one for the weights and another for the biases
	# Assumes that the proper variable names exist in the given scope somewhere
	if not weightsPath.startswith("models/"):
		weightsPath = "models/" + weightsPath
	if not biasesPath.startswith("models/"):
		biasesPath = "models/" + biasesPath
	if not weightsPath.endswith(".npz"):
		weightsPath = weightsPath + ".npz"
	if not biasesPath.endswith(".npz"):
		biasesPath = biasesPath + ".npz"
	
	# Raw numpy values.  Need to be loaded into variables.
	weightsDict = numpy.load(weightsPath)
	biasesDict = numpy.load(biasesPath)

	# Here, we do a for loop looping through all of the names, "name".
	for name, weights_tnsr in weightsDict.iteritems():
		with tf.variable_scope(scope) as model_scope:
			with tf.variable_scope(name) as layer_scope:		
				tf.get_variable("Weights", trainable=True, 
								initializer=tf.constant(weights_tnsr))
	for name, biases_tnsr in biasesDict.iteritems():
		with tf.variable_scope(scope) as model_scope:
			with tf.variable_scope(name) as layer_scope:
				tf.get_variable("Bias", 	trainable=True, 
								initializer=tf.constant(biases_tnsr))

	return
