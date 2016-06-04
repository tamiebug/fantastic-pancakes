import numpy as np
import caffe
import tensorflow as tf

import unittest
import utils
import vgg19
import settings as s

class vgg19Tests(unittest.TestCase):	
	def setUp(self):
		
		# Layer activations that we are going to test
		self.testLayers = ['relu1_1', 'relu2_1', 'relu3_4', 'relu4_4', 'pool5', 'relu6', 'relu7', 'prob']
		
		# Set up Tensorflow Model
		self.sess = tf.InteractiveSession()
		self.img = tf.placeholder("float", [None, 224, 224, 3], name="images") 
		self.model = vgg19.Vgg19()
		self.model.buildGraph(self.img, train=False)
		
		images = utils.loadImage(s.DEF_TEST_IMAGE_PATHS[0])
		images = images.reshape((1,224,224,3))
		
		self.sess.run(tf.initialize_all_variables())
		#print self.model.layers.keys()
		self.output = self.sess.run([self.model.layers[_] for _ in self.testLayers], 
								feed_dict={ self.img: images})
		
		# Set up Caffe Model for comparison
		self.coffee = caffe.Net(s.DEF_PROTOTXT_PATH, s.DEF_CAFFEMODEL_PATH, caffe.TEST)
		transformer = caffe.io.Transformer({'data' : self.coffee.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1)) 	# Move color channels to left-most dimension
		transformer.set_channel_swap('data', (2,1,0))	# Swap color channels from RGB to BGR
		
		transformed_image = transformer.preprocess('data', images)
		self.coffee.blobs['data'].data[...] = transformer(images)
		self.coffee.forward()
		
		
	def tearDown(self):
		tf.reset_default_graph()
		self.sess.close()
		
	def blob_tensor_equality_assert(self, name, tolerance=.01, testingChannels=[0]):
		# Pass an empty list to testingChannels to test all of them
		if testingChannels == []:
			blob_data = self.coffee.blobs[name]
			tensor_data = self.output[self.testLayers.index(name)]
		else:
			blob_data = self.coffee.blobs[name][testingChannels]
			tensor_data = self.output[self.testLayers.index(name)][testingChannels]
		greatest_diff = np.absolute(blob_data - tensor_data)
		assertLessEqual(greatest_diff, tolerance, "Greatest difference was %f"%greatest_diff)
		
	def test_relu1_1(self):
		self.blob_tensor_equality_assert(self, 'relu1_1', .001, [])
		
	def test_relu2_1(self):
		self.blob_tensor_equality_assert(self, 'relu2_1', .001, [0])
		
	def test_relu3_4(self):
		self.blob_tensor_equality_assert(self, 'relu3_4', .001, [0])
		
	def test_relu4_4(self):
		self.blob_tensor_equality_assert(self, 'relu4_4', .001, [0])
	
	def test_pool5(self):
		self.blob_tensor_equality_assert(self, 'pool5', .001, [0])
		
	def test_relu6(self):
		self.blob_tensor_equality_assert(self, 'relu6', .001, [0])
		
	def test_relu7(self):
		self.blob_tensor_equality_assert(self, 'relu7', .001, [0])
		
	def test_prob(self):
		self.blob_tensor_equality_assert(self, 'prob', .001, [0])
		
		
if __name__ == '__main__':
	unittest.main()
