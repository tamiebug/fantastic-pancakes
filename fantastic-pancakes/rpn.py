import numpy as np
import tensorflow as tf
import utils.settings as s
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
			rpnBboxPred = tf.createConvLayer(layer3x3, "rpn_bbox_pred")
			
				

def proposalLayer(anchors, feature_stride, iou_threshold, pre_nms_keep, post_nms_keep, \
					scores, bbox_regressions, feature_h, feature_w, img_w, img_h, \
					minimum_dim \
					):
	""" 
	Propose the actual regions given outputs of previous conv layers

	Arguments:

	anchors 		-- A list of anchors on which the proposal regions will be based

	feature_stride 	-- Ratio of number of features in the last layer of the base
	convolutional layer to the number of pixels in the image input layer.  Equal
	to 16 for VGGNet and ZF

	iou_threshold	-- Ratio determining the IoU (Intersection over Union) radius
	to be used for non-maximum suppression

	pre_nms_keep	-- Regions proposed are scored by the previous convolutional
	network, this is the number of top scoring regions to keep before performing
	NMS (non-maximum suppression)

	post_nms_keep	-- NMS reduces the number of regions greatly.  Number of regions
	to keep after performing NMS (non-maximum suppression).

	bbox_regressions-- List of regressions for each initial anchor at each position
	in the feature map input to the region proposal network. (is a tf.Variable object)

	scores			-- List of scores ranking the aformentioned regions after regressions.
	(is a tf.Variable)

	feature_h		-- Height of feature map
	
	feature_w		-- Width of feature map

	img_w			-- Input layer image width (in pixels)
	
	img_h			-- Input layer image height (in pixels)

	minimum_dim		-- Smallest allowed anchor side dimension.
	"""
	
	baseAnchors = generateAnchors(ratios=[2,1,.5])

	shiftedAnchors = generateShiftedAnchors(baseAnchors, feature_h, feature_w)

	regressedAnchors = regressAnchcors(shiftedAnchors, bbox_regressions,
							len(baseAnchors), feature_h, feature_w)

	clippedAnchors = clipRegions(regressedAnchors, img_w, img_h)

	p_scores, p_anchors = prundedScoresAndAnchors(clippedAnchors, minimum_dim,
							feature_h, feature_w, scores)  

	top_scores, top_score_indices = tf.nn.top_k(p_scores, k=pre_nms_keep)
	top_anchors = tf.gather(p_anchors, top_score_indices)

	# Changing shape into form [num_boxes,4] as required by tf.image.non_max_suppression
	top_anchors_transposed = tf.tranpose(top_anchors,(0,1))

	post_nms_indices = tf.image.non_max_suppression(top_anchors_transposed, top_scores, 
							post_nms_keep, iou_threshold=iou_threshold)
	
	# To be able to select elements from top_anchors with these indices, we need to transpose it back
	
	top_anchors_detransposed = tf.transpose(top_anchors_transposed, (0,1))
		
	final_anchors = tf.gather(top_anchors, post_nms_indices)
	final_scores = tf.gather(top_scores, post_nms_indices)

	# Remember to reshape final_anchors
	return final_anchors

def prunedScoresAndAnchors(anchors, numBaseAnchors, minimum_dim, feature_h, feature_w, scores):
	""" Return list of anchors and scores larger than a given minimum size """
	x1,y1,x2,y2 = tf.unpack(anchors, 1)

	w = tf.sub(x2,x1)
	h = tf.sub(y2,y1)

	width_suffices = tf.greater_equal(w, tf.constant([minimum_dim+1]))
	height_suffices = tf.greater_equal(h, tf.constant([minimum_dim+1]))
	
	both_suffice = tf.logical_and(width_suffices, height_suffices)
	
	# Current dimensionality is (1, numBaseAnchors, feature_h, feature_w)
	# For ease of indexing and NMS, we will squash this to
	# ( numBaseAnchors * feature_h * feature_w) to work around tensorflow limitations
	
	both_suffice_reshape = tf.reshape(both_suffice, (numBaseAnchors*feature_h*feature_w))

	# We can only select indices from the first dimension, so we need to do a transpose
	both_suffice_transposed = tf.transpose(both_suffice_reshape, (0,1))
	
	indices = np.where(both_suffice_transpose)
	
	# Massage both anchors and scores into compatible forms.
	anchors_reshape = tf.reshape(anchors, (4, numBaseAnchors*feature_h*feature_w))
	anchors_transpose = tf.transpose(anchors_reshape, (0,1))
	scores_reshape = tf.reshape(scores, (numBaseAnchors*feature_h*feature_w))
	scores_transpose = tf.transpose(scores_transpose, (0,1))
	
	anchors_gathered = np.gather(anchors_transpose, indices)
	scores_gathered = np.gather(scores_transpose, indices)
	
	# We really just need lists of anchors in the end, so these shapes will do for now.
	return anchors_gathered, scores_gathered

	
def clipRegions(anchors, img_w, img_h):
	""" Clip anchors so that all lie entirely within image """

	x1, y1, x2, y2 = tf.unpack(anchors, 1)
	
	zero = tf.constant([0.])
	max_x = tf.constant([img_w])
	max_y = tf.constant([img_h])

	x1_clipped = tf.maximum(tf.minimum(zero, x1), max_x)
	x2_clipped = tf.maximum(tf.minimum(zero, x2), max_x)
	y1_clipped = tf.maximum(tf.minimum(zero, y1), max_y)
	y2_clipped = tf.maximum(tf.minimum(zero, y2), max_y)

	# Pack 'em back up
	retVal = tf.pack([x1_clipped, y1_clipped, x2_clipped, y2_clipped],1)

	return retVal

def generateShiftedAnchors(anchors, feature_h, feature_w):
	""" Generate shifted anchors to be regressed into the final RPN output

	A score is created for every anchor at each feature.  Using feature_stride,
	we then determine at which locations in the image the areas these scores represent
	reside.  There are, using defaults, nine total anchors at each position, given in the
	input anchors.  We must shift these anchors to each x,y location, for a total of
	feature_w * feature_h * len(anchors) anchors.
	"""

	x_locations = range(0, feature_w) * feature_stride
	y_locations = range(0, feature_h) * feature_stride

	shifted_anchors = np.zeros((1, 4*len(anchors), feature_h, feature_w)) 
	for x in x_locations:
		for y in y_locations:
			for i, anchor in enumerate(anchors):
				shifted_anchors[0,y,x,4*i:4*(i+1)] = \
					[anchor[0] + x, anchor[1] + y,
					anchor[2] + x, anchor[3] + y]
	
	# Output has shape (1,4*len(anchors), feature_h, feature_w))
	return shifted_anchors

def regressAnchors(anchors, bbox_regressions, numBaseAnchors, feature_h, feature_w):
	""" Given preliminary bounding boxes, regress them to their final location
	
	The bounding box regressions are outputs of convolutional layers and of
	the form (dx,dy,dw,dh), where dw and dh are the natural logarithms of
	the desired changes in width and height and where dx and dy are unscaled
	x and y displacements; they must be mulitiplied by the width and height of a given
	bounding box to arrive at the actual displacements.
	"""

	# First we have to reshape our anchors list to more easily access elements
	# anchors.shape = (1,4*numBaseAnchors, feature_h, feature_w)

	reshapedAnchors = tf.reshape(anchors, (1,4, numBaseAnchors, feature_h, feature_w))
	reshapedBbox_regs = tf.reshape(bbox_regressions, (1,4, numBaseAnchors, feature_h, feature_w))
	
	x1, y1, x2, y2 = tf.unpack(reshapedAnchors, 1)
	dx, dy, dw, dh = tf.unpack(reshapedBbox_regs, 1)

	# We must get the anchors into the same width/height x/y format as the bbox_regressions
	x = tf.div(tf.add(x1, x2), tf.constant([2.]))
	y = tf.div(tf.add(y1, y2), tf.constant([2.]))
	w = tf.add(tf.sub(x2, x1), tf.constant([1.]))
	h = tf.add(tf.sub(y2, y1), tf.constant([1.]))

	# The dx and dy given by the regression must be scaled by w and h and added
	# to the anchors
	x_new = tf.add(tf.mul(dx,w), x)
	y_new = tf.add(tf.mul(dy,h), y)

	# Since logarithms of the values in question are easier to learn (no regression means
	# a logarithm of the change being zero), we learn the logarithms of h, w.
	w_new = tf.mul(tf.exp(dw),w)
	h_new = tf.mul(tf.exp(dh),h)

	# Transform back to the original (x1,y1,x2,y2) coordinate system
	x1_final = tf.sub(x_new, tf.mul(tf.constant([.5], w_new)))
	y1_final = tf.sub(y_new, tf.mul(tf.constant([.5], h_new)))
	x2_final = tf.add(x_new, tf.mul(tf.constant([.5], w_new)))
	y2_final = tf.add(y_new, tf.mul(tf.constant([.5], h_new)))

	# Stack our anchors back up
	regressedAnchors = tf.pack([x1_final, y1_final, x2_final, y2_final], 1)
	
	# The output shape differs from the input shape;  Output shape is
	# regressedAnchors.shape = (1, 4, numBaseAnchors, feature_h, feature_y)
	return regressedAnchors
	
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


