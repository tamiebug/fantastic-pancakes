import numpy as np
import caffe
import tensorflow as tf

import unittest
import utils
import vgg19
import settings as s

import threading


class ModelOutputGenerator():
	__metaclass__ = utils.Singleton
	def __init__(self):
		# Layer activations that we are going to test
		self.testLayers = ['relu1_1', 'relu2_1', 'relu3_4', 'relu4_4', 'pool5', 'relu6', 'relu7', 'prob']
		self.output = None
		# Images we are testing with
		images = utils.loadImage(s.DEF_TEST_IMAGE_PATHS[0])	
		
		# Set up Tensorflow Model
		def runTensorflow(mog, images):
			with tf.Session() as sess:
				with tf.device("/cpu:0"):
					# Tensorflow does not know how to release /GPU:0 resources without process termination
					# Issue #1727 in the tensorflow/tensorflow git repository
					# Letting it just use CPU for this forward run instead
					
					img = tf.placeholder("float", [1, 224, 224, 3], name="images") 
					model = vgg19.Vgg19()
					model.buildGraph(img, train=False)

					images = images.reshape((1,224,224,3))
					
					sess.run(tf.initialize_all_variables())
					#print self.model.layers.keys()
					mog.output = sess.run([model.layers[_] for _ in mog.testLayers], 
											feed_dict={ img: images})
					
					sess.close()
			return
		
		t = threading.Thread(target=runTensorflow, kwargs={'mog':self,'images':images})
		t.start()
		t.join()	
		# Your thread is dead.  Release the resources, pretty please...
		
		# Set up Caffe Model for comparison
		self.coffee = caffe.Net(s.DEF_PROTOTXT_PATH, s.DEF_CAFFEMODEL_PATH, caffe.TEST)
		transformer = caffe.io.Transformer({'data' : self.coffee.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1)) 	# Move color channels to left-most dimension
		transformer.set_channel_swap('data', (2,1,0))	# Swap color channels from RGB to BGR
	
		transformed_image = transformer.preprocess('data', images)
		self.coffee.blobs['data'].data[...] = transformed_image
		self.coffee.forward()

		
	def returnBlob(self, layername, flavor):
		"""
		Returns a layer of name layername in either the tf or caffe model.
		"""
		if flavor=="caffe":
			return self.coffee.blobs[layername].data
		elif flavor=="tensorflow":
			return self.output[self.testLayers.index(layername)]
		else:
			raise KeyError("Caffe and tensorflow are the only allowed blob flavors")
		
class vgg19Tests(unittest.TestCase):				
	def setUp(self):
		self.model = ModelOutputGenerator()
		return
		
	def blob_tensor_equality_assert(self, name, tolerance=.01, testingChannels=[0]):
		# Pass an empty list to testingChannels to test all of them
		if testingChannels == []:
			blob_data = self.model.returnBlob(name, "caffe").transpose
			tensor_data = self.model.returnBlob(name, "tensorflow")
		else:
			blob_data = self.model.returnBlob(name, "caffe")[testingChannels]
			tensor_data = self.model.returnBlob(name, "tensorflow")[testingChannels]
		greatest_diff = np.absolute(blob_data - tensor_data)
		unittest.assertLessEqual(greatest_diff, tolerance, "Greatest difference was %f"%greatest_diff)
		
	def test_relu1_1(self):
		self.blob_tensor_equality_assert('relu1_1', .001, [])
		
	def test_relu2_1(self):
		self.blob_tensor_equality_assert('relu2_1', .001, [0])
		
	def test_relu3_4(self):
		self.blob_tensor_equality_assert('relu3_4', .001, [0])
		
	def test_relu4_4(self):
		self.blob_tensor_equality_assert('relu4_4', .001, [0])
	
	def test_pool5(self):
		self.blob_tensor_equality_assert('pool5', .001, [0])
		
	def test_relu6(self):
		self.blob_tensor_equality_assert('relu6', .001, [0])
		
	def test_relu7(self):
		self.blob_tensor_equality_assert('relu7', .001, [0])
		
	def test_prob(self):
		self.blob_tensor_equality_assert('prob', .001, [0])
		
		
if __name__ == '__main__':
	unittest.main()
