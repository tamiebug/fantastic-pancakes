import numpy as np
import skimage.io
import skimage.transform
import vgg19
import settings as s
import utils
import tensorflow as tf

def testVGG():
	
	images = np.array([ utils.loadImage(path) for path in s.DEF_TEST_IMAGE_PATHS ])
	images = images.reshape((3,224,224,3))
	
	print "Images successfully loaded!"	
	img = tf.placeholder("float", [None, 224, 224, 3], name="images")
	print "Creating model"
	model = vgg19.Vgg19()
	print "Building model in TensorFlow....",
	model.buildGraph(img, train=False)
	print "Model built"
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		print "Running Model"
		print model.layers.keys()
		output = sess.run(model.layers['prob'], feed_dict={ img: images})
		print "Output categories"
		for i, out in enumerate(output):
			print "%i:"%i,
			print np.argmax(out)
	
testVGG()	
