import numpy
import caffe
import tensorflow
import settings as s
from urllib import urlretrieve
import skimage.io
import skimage.transform:

def loadImage(imgPath):
 	print imgPath
	img = skimage.io.imread(imgPath)
	img = img / 255.0	
	# We now need to crop the center square from our image
	shortSide = min(img.shape[0:2])
	x_1 = (img.shape[0] - shortSide) / 2
	x_2 = (img.shape[0] + shortSide) / 2
	y_1 = (img.shape[1] - shortSide) / 2
	y_2 = (img.shape[1] + shortSide) / 2
	centerSquare = img[x_1:x_2, y_1:y_2]
	return skimage.transform.resize(centerSquare, (224,224))
	

def downloadModel():
	# prototxt for the vgg19 model
	def progressBar(blockCount, blockSize, fileSize):
		# Progress bar for download, passed to urlretrieve
		return
	
	urlretrieve(s.DEF_CAFFEMODEL_DL, s.DEF_CAFFEMODEL_PATH, progressBar)
	urlretrieve(s.DEF_PROTOTXT_DL, s.DEF_PROTOTXT_PATH, progressBar)
	return

def coffeeMachine():
	coffee = caffe.Net(s.DEF_PROTOTXT_PATH, s.DEF_CAFFEMODEL_PATH, caffe.TEST)
	caffeVggWeights = { name: blobs[0].data for name, blobs in coffee.params.iteritems() }
	caffeVggBiases = { name: blobs[1].data for name, blobs in coffee.params.iteritems() }
	numpy.savez(s.DEF_WEIGHTS_PATH, **caffeVggWeights)
	numpy.savez(s.DEF_BIASES_PATH, **caffeVggBiases)
	print "Coffee successfully brewed"