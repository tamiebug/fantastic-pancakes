import numpy
import tensorflow as tf
import settings as s

class Vgg19():
	def __init__(self):
		return

	def _extractCaffeLayers(self, weightsPath=s.DEF_WEIGHTS_PATH,
							biasesPath=s.DEF_BIASES_PATH):
		# Loads parameters from .npz files, one for the weights and another for the biases
		self.weightsDict = numpy.load(weightsPath)
		self.biasesDict = numpy.load(biasesPath)

	def buildGraph(self, img, train=False):
		# Takes as input a Tensorflow placeholder or layer and whether
		# the graph is being trained or whether it is trained.
		self._extractCaffeLayers()
		
		def createFirstConvLayer(bottom, name, trainable=True):
			# Creats a convolutional Tensorflow layer with its weights
			# permuted so that it takes RGB instead of BGR images as input
			# Returns the bias layer so it can be plugged into other layers
		
			weightValues = self.weightsDict[name]
			biasValues = self.biasesDict[name]
						
			# Tensorflow order	: [width, height, in_channels, out_channels]
			# Caffe order		: [out_channels, in_channels, width, height]
			# Hence, to translate from Caffe to Tensorflow
			weightValues = weightValues.transpose((2,3,1,0))
		
			# Converting BGR to RGB
			weightValues2 = numpy.copy(weightValues[:,:,[2,1,0],:])
			
			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues2, trainable=trainable, name="Filter")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				conv = tf.nn.conv2d(bottom, weights, [1,1,1,1], padding="SAME")
				bias = tf.nn.bias_add(conv, biases)
			return bias				
			
		def createConvLayer(bottom, name, trainable=True):
			# Creates a convolutional Tensorflow layer given the name
			# of the layer.  This name is looked up in the weighsDict and
			# biasesDict in order to obtain the parameters to construct the
			# layer

			# Tensorflow order	: [height, width, in_channels, out_channels]
			# Caffe order		: [out_channels, in_channels, width, height]
			# Hence, to translate from Caffe to Tensorflow
			weightValues = self.weightsDict[name].transpose((2,3,1,0))
			
			biasValues = self.biasesDict[name]

			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Filter")
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
			assert weightValues.shape == (OUTPUT_SIZE, INPUT_SIZE)

			# Reshape the weights to their unsquashed form 
			weightValues = weightValues.reshape((OUTPUT_SIZE, 512, 7, 7))

			# Transpose the weights so that it takes as input tensors in the
			# tensorflow order instead of the caffe order
			weightValues = weightValues.transpose((2, 3, 1, 0))
			weightValues = weightValues.reshape(INPUT_SIZE, OUTPUT_SIZE)
			
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

			# Swapping in_channel and out_channel for tf
			weightValues = weightValues.transpose((1,0))
			biasValues = self.biasesDict[name]
			
			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Weights")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				layer = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

			return layer

		# All layer types have been defined, it is now time to actually make the model
		self.layers = {}
		layerNames = [	           'relu1_1', 'conv1_2', 'relu2_2', 'pool1',
						'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 
						'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
						'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
						'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5' ]


		# The first convolutional layer is special!  We must do it separately.
		self.layers['conv1_1'] = createFirstConvLayer(img, 'conv1_1', trainable=train)
		
		# We start out with the input img
		prevLayer = self.layers['conv1_1']
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
			self.layers['drop6'] = tf.nn.dropout(self.layers['relu6'], name='drop6')
			prevLayer = self.layers['drop6']
		else:
			prevLayer = self.layers['relu6']
			 
		self.layers['fc7'] = createFcLayer(prevLayer, 'fc7', trainable=train)
		self.layers['relu7'] = tf.nn.relu(self.layers['fc7'], 'relu7')

		# If we are training the model, we need to activate dropout.
		if train:
			self.layers['drop7'] = tf.nn.dropout(self.layers['relu7'], name='drop7')
			prevLayer = self.layers['drop7']
		else:
			prevLayer = self.layers['relu7']
		
		self.layers['fc8'] = createFcLayer(prevLayer, 'fc8', trainable=train)
		self.layers['prob'] = tf.nn.softmax(self.layers['fc8'], name='prob')
		
	def saveModel(self, weightsName, biasesName, overwrite=False):
		"""
		Saves the current weights and biases for the model as files named
		weightsName and biasesName, respectively, in the models/ directory.
		If a file with this name already exists, will raise error unless
		overwrite=True, in which case it will overwrite the files.
		"""
		numpy.savez(weightsName)
		numpy.savez(biasesName)
		