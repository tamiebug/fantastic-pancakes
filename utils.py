import numpy
import settings as s
from urllib import urlretrieve

def coffeeMachine():
	import caffe
	coffee = caffe.Net(s.DEF_PROTOTXT_PATH, s.DEF_CAFFEMODEL_PATH, caffe.TEST)
	"""
	print("Length of layernames = %i, Length of coffeelayers = %i"%(len(coffee._layer_names), len(coffee.layers)))
	caffeVggBiases = {}
	for i, layer in enumerate(coffee.layers):
		print("We are at layer %i"%i)
		caffeVggBiases[coffee._layer_names[i]] = layer.blobs[0].data
	"""
	caffeVggWeights = { name: blobs[0].data for name, blobs in coffee.params.iteritems() }
	caffeVggBiases = { name: blobs[1].data for name, blobs in coffee.params.iteritems() }
	numpy.savez(s.DEF_WEIGHTS_PATH, **caffeVggWeights)
	numpy.savez(s.DEF_BIASES_PATH, **caffeVggBiases)

def downloadModel():
	# prototxt for the vgg19 model
	def progressBar(blockCount, blockSize, fileSize):
		# Progress bar for download, passed to urlretrieve
		return
	
	urlretrieve(s.DEF_CAFFEMODEL_DL, s.DEF_CAFFEMODEL_PATH, progressBar)
	urlretrieve(s.DEF_PROTOTXT_DL, s.DEF_PROTOTXT_PATH, progressBar)
	return
