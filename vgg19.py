import numpy
import tensorflow as tf
import settings as s

class Vgg19():
	def __init__(self, namespace="vgg19"):
		self.namespace=namespace
		
	def _extractCaffeLayers(self, weightsPath, biasesPath):
		# Loads parameters from .npz files, one for the weights and another for the biases

		if not weightsPath.startswith("models/"):
			weightsPath = "models/" + weightsPath
		if not biasesPath.startswith("models/"):
			biasesPath = "models/" + biasesPath
		if not weightsPath.endswith(".npz"):
			weightsPath = weightsPath + ".npz"
		if not biasesPath.endswith(".npz"):
			biasesPath = biasesPath + ".npz"

		self.weightsDict = numpy.load(weightsPath)
		self.biasesDict = numpy.load(biasesPath)

 
	def buildGraph(self, img, train=False, weightsPath=s.DEF_WEIGHTS_PATH,
					biasesPath=s.DEF_BIASES_PATH):
		# Takes as input a Tensorflow placeholder or layer and whether
		# the graph is being trained or whether it is trained.
		self._extractCaffeLayers(weightsPath, biasesPath)
			
		def createConvLayer(bottom, name, trainable=True):
			# Creates a convolutional Tensorflow layer given the name
			# of the layer.  This name is looked up in the weighsDict and
			# biasesDict in order to obtain the parameters to construct the
			# layer

			weightValues = self.weightsDict[name]
			biasValues = self.biasesDict[name]

			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Weights")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				conv = tf.nn.conv2d(bottom, weights, [1,1,1,1], padding="SAME")
				bias = tf.nn.bias_add(conv, biases)	
			return bias

		def createFirstFcLayer(bottom, name, trainable=True):
			# Creates the first fully connected layer which converts the
			# output of the last convolutional layer to the input for the next
			# fully connected ones.  Returns the bias layer.
			
			INPUT_SIZE = 25088
			OUTPUT_SIZE = 4096
			
			weightValues = self.weightsDict[name]	
			biasValues = self.biasesDict[name]
			
			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Weights")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				
				flattenedInput = tf.reshape(bottom, [-1, INPUT_SIZE])
				layer = tf.nn.bias_add(tf.matmul(flattenedInput, weights), biases)
			return layer

		def createFcLayer(bottom, name, trainable=True):
			# Creates a fully connected layer with INPUT_SIZE inputs and
			# INPUT_SIZE outputs.  Loads the weights from the weightsDict and
			# biasesDict dictionaries using they key value name and returns the
			# bias layer.

			INPUT_SIZE = 4096
			weightValues = self.weightsDict[name]
			biasValues = self.biasesDict[name]
			
			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Weights")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				layer = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

			return layer

		# All layer types have been defined, it is now time to actually make the model
		self.layers = {}
		layerNames = [	'conv1_1', 'relu1_1', 'conv1_2', 'relu2_2', 'pool1',
						'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 
						'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
						'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
						'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5' ]

		with tf.variable_scope(self.namespace) as scope:
			# We start out with the input img
			prevLayer = img
			for layername in layerNames:
				if layername.startswith('conv'):
					self.layers[layername] = createConvLayer(prevLayer, layername, train)
				elif layername.startswith('pool'):
					self.layers[layername] = tf.nn.max_pool(prevLayer, ksize=[1,2,2,1], 
						strides=[1,2,2,1], padding='VALID', name=layername)
				elif layername.startswith('relu'):
					self.layers[layername] = tf.nn.relu(prevLayer, layername)
				else:
					print("Error in layerNames in vgg19.py.  %s was not a conv, relu, nor pool"%layername)		
				prevLayer = self.layers[layername]
				
			self.layers['fc6'] = createFirstFcLayer(prevLayer, 'fc6', trainable=train)
			self.layers['relu6'] = tf.nn.relu(self.layers['fc6'], 'relu6')
				
			# If we are training the model, we need to activate dropout.
			if train:
				self.layers['drop6'] = tf.nn.dropout(self.layers['relu6'], 0.5, name='drop6')
				prevLayer = self.layers['drop6']
			else:
				prevLayer = self.layers['relu6']
				 
			self.layers['fc7'] = createFcLayer(prevLayer, 'fc7', trainable=train)
			self.layers['relu7'] = tf.nn.relu(self.layers['fc7'], 'relu7')

			# If we are training the model, we need to activate dropout.
			if train:
				self.layers['drop7'] = tf.nn.dropout(self.layers['relu7'], 0.5, name='drop7')
				prevLayer = self.layers['drop7']
			else:
				prevLayer = self.layers['relu7']
			
			self.layers['fc8'] = createFcLayer(prevLayer, 'fc8', trainable=train)
			self.layers['prob'] = tf.nn.softmax(self.layers['fc8'], name='prob')
			
	def save(self, weightsName, biasesName, overwrite=False):
		"""
		Saves the current weights and biases for the model as files named
		weightsName and biasesName, respectively, in the models/ directory.
		If a file with this name already exists, will raise error unless
		overwrite=True, in which case it will overwrite the files.
		"""
	
		weightsVar = []
		biasVar = []
		for var in tf.all_variables():
			if var.name.startswith(self.namespace):
				if "Weights" in var.name:
					weightsVar.append(var)
				if "Bias" in var.name:
					biasVar.append(var)
		fullVarList = biasVar + weightsVar			
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			output = sess.run(fullVarList)
		
		theDict = {var.name:output[i] for i,var in enumerate(fullVarList)}
		weightsDict = {}
		biasDict = {}
		for key, val in theDict.iteritems():
			if "Weights" in key:
				weightsDict[key.split(self.namespace)[-1].split('/Weights:0')[0]] = val
			if "Bias" in key:
				biasDict[key.split(self.namespace)[-1].split('/Bias:0')[0]] = val
		numpy.savez("models/"+weightsName, **weightsDict)
		numpy.savez("models/"+biasesName, **biasDict)		
