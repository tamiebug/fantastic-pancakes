import numpy
import caffe
import tensorflow
import settings as s
from urllib import urlretrieve
import skimage.io
import skimage.transform

class Singleton(type):
	""" Simple Singleton for use as metaclass """
	_instances = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]
		

def loadImage(imgPath):
	""" 
	Loads an image from imgPath, crops the center square, and subtracts the RGB
	values from VGG_MEAN
	"""
	img = skimage.io.imread(imgPath)
	# We now need to crop the center square from our image
	shortSide = min(img.shape[0:2])
	x_1 = (img.shape[0] - shortSide) / 2
	x_2 = (img.shape[0] + shortSide) / 2
	y_1 = (img.shape[1] - shortSide) / 2
	y_2 = (img.shape[1] + shortSide) / 2
	centerSquare = img[x_1:x_2, y_1:y_2]
	
	rescaledImg = skimage.transform.resize(centerSquare, (224,224))
	
	# Subtract VGG_MEAN from every pixel.  VGG_MEAN is in BGR, but image is
	# RGB, hence the reverse order.
	rescaledImg[:,:,0] -= s.VGG_MEAN[2]
	rescaledImg[:,:,1] -= s.VGG_MEAN[1]
	rescaledImg[:,:,2] -= s.VGG_MEAN[0]
	return rescaledImg
	
def downloadModel():
	"""
	Download the vgg19 model (.caffemodel and .prototxt) files and save them to the
	DEF_CAFFEMODEL_PATH and DEF_PROTOTXT_PATH directories
	"""
	# prototxt for the vgg19 model
	def progressBar(blockCount, blockSize, fileSize):
		# Progress bar for download, passed to urlretrieve
		return
	
	urlretrieve(s.DEF_CAFFEMODEL_DL, s.DEF_CAFFEMODEL_PATH, progressBar)
	urlretrieve(s.DEF_PROTOTXT_DL, s.DEF_PROTOTXT_PATH, progressBar)
	return

def coffeeMachine(prototxtPath=s.DEF_PROTOTXT_PATH, caffemodelPath=s.DEF_CAFFEMODEL_PATH):
	"""
	Extract the weights and biases from the .caffemodel and save it in npz files named
	DEF_WEIGHTS_PATH and DEF_BIASES_PATH
	"""	
	coffee = caffe.Net(prototxtPath, caffemodelPath, caffe.TEST)
	caffeVggWeights = { name: blobs[0].data for name, blobs in coffee.params.iteritems() }
	caffeVggBiases = { name: blobs[1].data for name, blobs in coffee.params.iteritems() }
	numpy.savez(s.DEF_WEIGHTS_PATH, **caffeVggWeights)
	numpy.savez(s.DEF_BIASES_PATH, **caffeVggBiases)
	print "Coffee successfully brewed"
