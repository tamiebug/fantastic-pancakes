import numpy as np
import tensorflow as tf
import settings as s
import loadNetVars

class Rpn():
	def __init__(self, namespace="rpn"):
		self.namespace=namespace


	def buildGraph(self, features, train=False):

		def createConvLayer(bottom, name, stride=[1,1,1,1]):
		# Creates a convolutional Tensorflow layer given the name
		# of the layer.  Expects a tf.Variable with name
		# model_scope/layer_scope/Weights and one with name
		# model_scope/layer_scope/Bias to already exist.
			with tf.variable_scope(name) as scope:
				scope.reuse_variables()
				prevLayer = tf.nn.conv2d(bottom, tf.get_variable("Weights"), stride,
									padding="SAME")
				prevLayer = tf.nn.bias_add(prevLayer, tf.get_variable("Bias"))
			return prevLayer

		with tf.variable_scope(self.namespace) as model_scope:
			
			layer3x3 = createConvLayer(features, "rpn_conv/3x3")
			
			# Region Proposal Network - Probabilities
			prevLayer = createConvLayer(layer3x3, "rpn_cls_score")
			prevLayer = tf.reshape(prevLayer, (0,2,-1,0))			
			prevLayer = tf.nn.softmax(prevLayer, name="rpn_cls_prob")
			rpnProb = tf.reshape(prevLayer, (1,18,14,14))
			
			# Region Proposal Network - Bounding Box Proposal Regression
			rpnBboxPrer = tf.createConvLayer(layer3x3, "rpn_bbox_pred")
			
	# Non Maximum Suppression... etc.! 




def generateAnchors(ratios=[.5,1,2], scales=[8,16,32], base=[1,1,16,16]):
	# Generates a list of anchors based on a list of aspect ratios, a base
	# size, and a list of scales.  Anchors are of the form [x0,y0,x1,y1], where
	# x0,y0 are the coordinates of the input pixel on the bottom-left of the anchor and
	# x1,y1 are the coordinates of the input pixel on the bottom-right of the anchor.
	
	return scaleUpAnchors( aspectRatioAnchors(base,ratios) , scales )

# Main inner methods for generateAnchors

def aspectRatioAnchors(baseAnchor,aspectRatios):
	# Produces anchors of various aspect ratios based on a base size
	wh_base = toWidthHeight(baseAnchor)
	anchors  = []
	for ratio in aspectRatios:
		w = np.round( np.sqrt(wh_base[0] * wh_base[1] * ratio) )
		h = np.round( w / ratio )
		anchors.append(toWidthHeightInverse([w,h,wh_base[2],wh_base[3]]))
		
	return anchors

def scaleUpAnchors(anchors, scales):
	# Takes the given anchors and creates a new list of anchors, with one at each scale.
	out_anchors = []	
	for anchor in anchors:
		wh_anchor = toWidthHeight(anchor)
		for scale in scales:
			scaled_wh_anchor = [wh_anchor[0]*scale, wh_anchor[1]*scale,
								wh_anchor[2],wh_anchor[3]]
			out_anchors.append(toWidthHeightInverse(scaled_wh_anchor))
		
	return out_anchors

# Auxiliary methods for the two methods above

def toWidthHeight(anchor):
	# Transforms an anchor to a new format [w,h,x,y];
	# width, height, center x coord, center y coordinate

	# Since in an anchor [x0,y0,x1,y1] we are represesnting not corner coordinates but
	# coordinates of pixels that compose the corner, actual widths go across the pixels
	# themselves, adding one to the total widths and heights of the regions covered by the
	# pixels themseles. 
	width = anchor[2]-anchor[0]+1
	height = anchor[3]-anchor[1]+1

	x = .5*(anchor[0]+anchor[2])
	y = .5*(anchor[1]+anchor[3])
	
	return [width,height,x,y]

def toWidthHeightInverse(wh):
	# Transforms the output of toWidthHeight, in the form [width,height,x,y],
	# where width and height are the width and height of an anchor and x,y
	# the x and y coordinates of its center, back to the regular representation
	anchor = [0.,0.,0.,0.]

	# We need to subtract 1 from the widths and the heights because they are heights 
	# and widths of the areas taken up by the pixels, not of the number of pixels themselves.
	anchor[0] = wh[2] - .5*(wh[0]-1)
	anchor[1] = wh[3] - .5*(wh[1]-1)
	anchor[2] = wh[2] + .5*(wh[0]-1)
	anchor[3] = wh[3] + .5*(wh[1]-1)
	
	return anchor


