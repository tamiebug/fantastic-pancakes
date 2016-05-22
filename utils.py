import caffe
import numpy
import settings as s
from urllib import urlretrieve

def coffeeMachine():
	coffee = caffe.Net(s.DEF_PROTOTXT_PATH, s.DEF_CAFFEMODEL_PATH, caffe.TEST)
	caffeVggWeights = { coffee._layer_names[i]: layer.blobs[0].data for i, layer in enumerate(coffee.layers) }
	caffeVggBiases = { coffee._layer_names[i]: layer.blobs[1].data for i, layer in enumerate(coffee.layers) }
	numpy.savez(s.DEF_WEIGHTS_PATH,**caffeVggWeights)
	numpy.savez(s.DEF_BIASES_PATH,**caffeVggBiases)

def downloadModel():
	# Function stub, this function is supposed to download the caffemodel and
	# prototxt for the vgg19 model
	def progressBar(blockCount, blockSize, fileSize):
		# Progress bar for download, passed to urlretrieve
		return
	
	urlretrieve(s.DEF_CAFFEMODEL_DL, s.DEF_CAFFEMODEL_PATH, progressBar)
	urlretrieve(s.DEF_PROTOTXT_DL, s.DEF_PROTOTXT_PATH, progressBar)
	return
