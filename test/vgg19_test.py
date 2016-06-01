import unittest
import vgg19
import numpy as np
import caffe
import tensorflow as tf

class vgg19Tests(unittest.TestCase):	
	def setUp(self):
		self.sess = tf.InteractiveSession()
		self.img = tf.placeholder("float", [None, 224, 224, 3], name="images") 
		self.model = vgg.Vgg19()
		self.model.buildGraph(self.img, train=False)
		
	def tearDown(self):
		tf.reset_default_graph()
		self.sess.close()
		
	def testVGG(self):
		images = np.array([ utils.loadImage(path) for path in s.DEF_TEST_IMAGE_PATHS ])
		images = images.reshape((3,224,224,3))

		self.sess.run(tf.initialize_all_variables())
		print "Running Model"
		print self.model.layers.keys()
		output = self.sess.run(model.layers['prob'], feed_dict={ img: images})
		print "Output categories"
		for i, out in enumerate(output):
			print "%i:"%i,
			print np.argmax(out)
	
if __name__ = '__main__':
	unittest.main()
