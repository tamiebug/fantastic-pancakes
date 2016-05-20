import tensorflow as tf
import caffe
import numpy
import settings as s

class Vgg19():
	def __init__(self):
		return

	def _extractCaffeLayers(self, trainable=True):

		# We need to load, instead, from a numpy file.
		caffeVgg = caffe.Net(s.DEF_PROTOTXT_PATH, s.DEF_CAFFEMODEL_PATH, caffe.TEST)
		caffeVggLayers = { caffeVGG._layer_names[i]: layer for i, layer in enumerate(caffeVGG.layers) }
		return caffeVggLayers
		
	def _buildGraph(self, img, train=False):
		caffeVggLayers = _extractCaffeLayers(self, trainable=True)
		
		def createFirstConvLayer(bottom, caffeLayer, name, trainable=True):
			
			
			weightValues = layer.blobs[0].data.transpose((2,3,1,0))
			biasValues = layer.blobs[1].data
			
			# Input channels is the 3rd component
			weightValues = numpy.copy(weightValues[:,:,[2,1,0],:]
				
			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Filter")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				conv = tf.nn.conv2d(bottom, weights, [1,1,1,1], padding="SAME")
				bias = tf.nn.bias_add(conv, biases)
				
			return bias				
			
		def createConvLayer(bottom, caffeLayer, name, trainable=True):
			
			weightValues = layer.blobs[0].data.transpose((2,3,1,0))
			biasValues = layer.blobs[1].data

			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Filter")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				conv = tf.nn.conv2d(bottom, weights, [1,1,1,1], padding="SAME")
				bias = tf.nn.bias_add(conv, biases)
				
			return bias

		def createFirstFcLayer(bottom, caffeLayer, name, trainable=True):
			INPUT_SIZE = 25088
			OUTPUT_SIZE = 4096
			weightValues = layer.blobs[0].data
			assert weightValues.shape == (OUTPUT_SIZE, INPUT_SIZE)
			weightValues = weightValues.reshape((INPUT_SIZE, 512, 7, 7))
			weightValues = weightValues.transpose((2, 3, 1, 0))
			weightValues = weightValues.reshape(OUTPUT_SIZE, INPUT_SIZE)

			biasValues = layer.blobs[1].data
			
			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Weights")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				
				flattenedInput = tf.reshape(bottom, [-1, INPUT_SIZE])
				layer = tf.nn.bias_add(tf.matmul(flattenedInput, weights), biases)
			
			return layer

		def createFcLayer(bottom, caffeLayer, name, trainable=True):
			INPUT_SIZE = 4096
			weightValues = layer.blobs[0].data
			# Swapping in_channel and out_channel for tf
			weightValues = weightValues.transpose((1,0))
			biasValues = layer.blobs[1].data
			
			with tf.variable_scope(name) as scope:
				weights = tf.Variable(weightValues, trainable=trainable, name="Weights")
				biases = tf.Variable(biasValues, trainable=trainable, name="Bias")
				layer = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

			return layer

		# All layer types have been defined, it is now time to actually make the model
		self.layers = {}
		layerNames = [	'conv1_1', 'reul1_1', 'conv1_2', 'relu2_2', 'pool1',
						'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 
						'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
						'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool3',
						'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool3' ]
		prevLayer = img
		for layername in layerNames:
			if layername.startswith('conv'):
				caffeLayer = caffeVggLayers[layername]
				self.layers[layername] = createConvLayer(prevLayer, caffeLayer, layername, train)
			elif layername.startswith('pool'):
				self.layers[layername] = tf.nn.max_pool(prevLayer, ksize=[1,2,2,1], 
					strides=[1,2,2,1], padding='SAME', name=layername)
			elif: layername.startswith('relu'):
				self.layers[layername] = tf.nn.relu(prevLayer, layername)
			else:
				print("Error in layerNames in vgg19.py.  %s was not a conv, relu, nor pool"%layername)		
			prevLayer = self.layers[layername]
		self.layers['fc6'] = createFirstFcLayer(prevLayer, 'fc6', 'fc6', trainable=train)
		self.layers['relu6'] = tf.nn.relu(self.layers['fc6'], 'relu6')
		if train:
			self.layers['drop6'] = tf.nn.dropout(self.layers['relu6'], name='drop6')
			prevLayer = self.layers['drop6']
		else:
			prevLayer = self.layers['relu6']
			 
		self.layers['fc7'] = createFcLayer(prevLayer, 'fc7', 'fc7', trainable=train)
		self.layers['relu7'] = tf.nn.relu(self.layers['fc6'], 'relu7')
		if train:
			self.layers['drop7'] = tf.nn.dropout(self.layers['relu7'], name='drop7')
			prevLayer = self.layers['drop7']
		else:
			prevLayer = self.layers['relu7']
		
		self.layers['fc8'] = createFcLayer(prevLayer, 'fc7', 'fc7', trainable=train)
		self.prob = tf.nn.softmax(self.layers['fc8'], name='prob')
		
	def getOutput(self):
		return layers['prob']
