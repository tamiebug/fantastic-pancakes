import numpy as np
import tensorflow as tf
from layers.custom_layers import nms
from util import settings as s
from . import loadNetVars
from util.utils import easy_scope

def Rpn(features, image_attr, train=False, namespace="rpn"):
    """ Region proposal network.  Proposes regions to later be pooled and classified/regressed

    Inputs:
    features    - A tf.Tensor object of rank 4, dimensions (batch, height, width, channel),
        since this is the standard tensorflow order.

    image_attr  - A tf.Tensor object of rank 1, with values [img_h, img_w, scaling_factor], 
        where these values are described below:
        img_w           - This network currently does not support variable image sizes.  img_w 
        is the width of the input images to the base network
        img_h           - Same as above, but the height instead of the width.
        scaling_factor  - An input to the base vgg16 network is scaled such that it's as large 
            as possible with shortest side less than 600 pixels and longest side less than 100 
            pixels, both inclusive.  scaling_factor is the scaling factor used to effect this
            transformation.
        Output:
        A tf.tensor object of rank 2 with dimensions (num_rois, 4), where the second dimension
        is of the form {x0, y0, x1, y1}
    """

    def createConvLayer(bottom, name, stride=[1, 1, 1, 1]):
        # Creates a convolutional Tensorflow layer given the name
        # of the layer.  Expects a tf.Variable with name
        # model_scope/layer_scope/Weights and one with
        # model_scope/layer_scope/Bias to already exist.
        with easy_scope(name,reuse=True):
            prevLayer = tf.nn.conv2d(bottom, tf.get_variable("Weights"), stride,
                                     padding="SAME")
            prevLayer = tf.nn.bias_add(prevLayer, tf.get_variable("Bias"), name="out")
        return prevLayer

    with easy_scope(namespace, reuse=True):
        layer3x3 = createConvLayer(features, "rpn_conv/3x3")
        layer3x3 = tf.nn.relu(layer3x3, "rpn_relu/3x3")

        # Region Proposal Network - Probabilities
        prevLayer = createConvLayer(layer3x3, "rpn_cls_score")

        # Assuming that feat_w = feat_h = 14, and that the number of anchors is 9,
        # we have the output should be of shape (9,14,14,2).

        # However,a tf.nn.conv2d cannot create batches out of thin air.  Hence, the
        # rpn_cls_score should create a (1, 14, 14, 9*2) instead, which we reshape to
        # (1, 14, 14, 2, 9), transpose to (9, 14, 14, 2, 1), then tf.squeeze the last
        # dimension out to arrive at the desired wonderful shape of (9, 14, 14,
        # 2).  The last dimension of rpn_cls_score is unpacked from (9*2) to (2,9) and
        # not (9,2) since this is how the weights imported from caffe are packed.

        feature_h = tf.gather_nd(tf.shape(features), [1])
        feature_w = tf.gather_nd(tf.shape(features), [2])

        prevLayer = tf.reshape(prevLayer, (1, feature_h, feature_w, 2, 9))
        prevLayer = tf.transpose(prevLayer, (4, 1, 2, 3, 0))
        prevLayer = tf.squeeze(prevLayer)

        prevLayer = tf.reshape(prevLayer, (-1,2))

        rpnScores  = tf.nn.softmax(prevLayer,dim=-1, name="rpn_cls_prob_raw")
        rpnScores = tf.identity(rpnScores, name="wtf")

        rpnScores = tf.reshape(rpnScores, (9, feature_h, feature_w, 2))

        _ , rpnScores = tf.unpack(rpnScores, num=2, axis=-1)

        rpnScores = tf.identity(rpnScores, name="rpn_cls_prob")
        # Region Proposal Network - Bounding Box Proposal Regression
        rpnBboxPred = createConvLayer(layer3x3, "rpn_bbox_pred")
        
        # We want to resehape rpnBboxPred just like we did the scores.
        # Only difference is that we reshape to (9,14,14,4) instead of
        # (9,14,14,2) (in the case of feat_h=feat_w=14)

        prevLayer = tf.reshape(rpnBboxPred, (1, feature_h, feature_w, 9, 4))
        prevLayer = tf.transpose(prevLayer, (3,1,2,4,0))
        rpnBboxPred = tf.squeeze(prevLayer)

        out = proposalLayer(s.DEF_FEATURE_STRIDE,
                            s.DEF_IOU_THRESHOLD,
                            s.DEF_PRE_NMS_KEEP,
                            s.DEF_POST_NMS_KEEP,
                            rpnScores,
                            rpnBboxPred,
                            feature_h,
                            feature_w,
                            image_attr,
                            s.DEF_MIN_PROPOSAL_DIMS
                            )
        return out


def proposalLayer(feature_stride, iou_threshold, pre_nms_keep, post_nms_keep,
                  scores, bbox_regressions, feature_h, feature_w, img_attr,
                  minimum_dim):
    """ 
    Propose the actual regions given outputs of previous conv layers

    Arguments:
    feature_stride      -- Ratio of number of features in the last layer of the base
    convolutional layer to the number of pixels in the input.

    iou_threshold	-- Ratio determining the IoU (Intersection over Union) radius
    to be used for non-maximum suppression

    pre_nms_keep	-- Regions proposed are scored by the previous convolutional
    network, this is the number of top scoring regions to keep before performing
    NMS (non-maximum suppression)

    post_nms_keep	-- NMS reduces the number of regions greatly.  Number of regions
    to keep after performing NMS (non-maximum suppression).

    bbox_regressions    -- List of regressions for each initial anchor at each position
    in the feature map input to the region proposal network. (is a tf.Variable object)

    scores			-- List of scores ranking the aformentioned regions after regressions.
    (is a tf.Variable)

    feature_h		-- Height of feature map

    feature_w		-- Width of feature map

    img_attr            -- tf.Tensor object containing the following values in order
        img_w		-- Input layer image width (in pixels)
        img_h		-- Input layer image height (in pixels)
        scale_factor    -- Factor by which original image scaled before being fed into base
                            classification layer

    minimum_dim		-- Smallest allowed anchor side dimension.

    Output:
    A tuple consisting of...
    A tf.Tensor object of rank two with shape (num_rois, 4), the second dimension having the
    structure (x0, y0, x1, y1)
    And, A tf.Tensor object of rank one with shape (num_rois) containing scores

    """

    baseAnchors = generateAnchors(ratios=[2, 1, .5])

    shiftedAnchors = generateShiftedAnchors(baseAnchors, feature_h, feature_w,
            feature_stride)

    regressedAnchors = regressAnchors(shiftedAnchors, bbox_regressions)

    clippedAnchors = clipRegions(regressedAnchors, img_attr)

    p_anchors, p_scores = prunedScoresAndAnchors(clippedAnchors, 
            scores, minimum_dim, img_attr)

    pre_nms_keep = 6000
    top_scores, top_score_indices = tf.nn.top_k(p_scores, k=pre_nms_keep, name="top_scores")

    # Modifying the top_score_indices to be able to be properly used with gather_nd
    top_score_indices = tf.cast(top_score_indices, tf.int32, name="top_score_indices")
    top_score_indices = tf.expand_dims(top_score_indices, axis=1, name="top_score_indices_expanded")
    top_anchors = tf.gather_nd(p_anchors, top_score_indices, name="top_anchors")

    # We want nms to keep everything that passes the IoU test
    post_nms_indices = nms(top_anchors, top_scores,
                        post_nms_keep, iou_threshold=iou_threshold, name="post_nms_indices")

    # Expanding nms_indices for use with tf.gather_nd
    post_nms_indices = tf.expand_dims(post_nms_indices, axis=1, name="post_nms_indices_expanded")
    final_anchors = tf.gather_nd(top_anchors, post_nms_indices, 
            name='proposal_regions')
    final_scores = tf.gather_nd(top_scores, post_nms_indices, 
                                    name='proposal_region_scores')

    return final_anchors, final_scores


def prunedScoresAndAnchors(anchors, scores, minimum_dim, im_attr):
    """ Return list of anchors and scores larger than a given minimum size

        It is assumed that the shape of scores is (numAnchors, feat_h, feat_w, 2)
        We output tensors of shape (?,2) for scores_gathered and (?,4) for
        anchors_gathered
    """
    anchors = tf.transpose(anchors, (1,2,0,3))
    scores = tf.transpose(scores, (1,2,0))
    anchors = tf.reshape(anchors, (-1,4))
    scores = tf.reshape(scores, (-1,))

    x1, y1, x2, y2 = tf.unpack(anchors, num=4, axis=-1)

    w = x2 - x1 + 1.
    h = y2 - y1 + 1.

    # Gotta scale the minimum_dim by the scale factor before use
    minimum_dim = tf.constant([s.DEF_MIN_PROPOSAL_DIMS], dtype=tf.float32)
    minimum_dim = minimum_dim * im_attr[2]

    width_suffices = tf.greater_equal(w, minimum_dim, name="geqw")
    height_suffices = tf.greater_equal(h, minimum_dim,name="geqh")

    both_suffice = tf.logical_and(width_suffices, height_suffices)
    indices = tf.where(both_suffice)

    # The actual grabbing of indexed values happens here 
    anchors_gathered = tf.gather_nd(anchors, indices, name="pruned_anchors")
    scores_gathered = tf.gather_nd(scores, indices, name="pruned_scores")
    return anchors_gathered, scores_gathered


def clipRegions(anchors, img_attr, axis=-1):
    """ Clip anchors so that all lie entirely within image """

    # Input anchors will be of shape
    # (numBaseAnchors, feature_h, feature_w, 4)
    x1, y1, x2, y2 = tf.unpack(anchors,num=4,axis=axis)

    zero = tf.constant([0.])
    
    max_x = [tf.sub(img_attr[1] * img_attr[2], tf.constant([1.]), name="clip_img_w")]
    max_y = [tf.sub(img_attr[0] * img_attr[2], tf.constant([1.]), name="clip_img_h")]

    x1_clipped = tf.minimum(tf.maximum(zero, x1), max_x)
    x2_clipped = tf.minimum(tf.maximum(zero, x2), max_x)
    y1_clipped = tf.minimum(tf.maximum(zero, y1), max_y)
    y2_clipped = tf.minimum(tf.maximum(zero, y2), max_y)

    # Pack 'em back up
    retVal = tf.pack([x1_clipped, y1_clipped, x2_clipped, y2_clipped], axis, name="clipped_anchors")

    return retVal

def generateShiftedAnchors(anchors, feature_h, feature_w, feature_stride):
    """ Generate shifted anchors to be regressed into the final RPN output

    A score is created for every anchor at each feature.  Using feature_stride,
    we then determine at which locations in the image the areas these scores represent
    reside.  There are, using defaults, nine total anchors at each position, given in the
    input anchors.  We must shift these anchors to each x,y location, for a total of
    feature_w * feature_h * len(anchors) anchors.
    """
   
    # The scaling factor I seek is actually the reciprocal, since I want
    # to transform back to original image coordinates, not go from img coords
    # to input coordinates
    feature_stride = tf.constant(feature_stride, dtype=tf.float32)
    x_locations = tf.to_float(tf.range(0, feature_w))
    y_locations = tf.to_float(tf.range(0, feature_h))

    x_zeros = tf.zeros([feature_w])
    y_zeros = tf.zeros([feature_h])

    x_stack = tf.pack([x_locations, x_zeros, x_locations, x_zeros], axis=1)
    y_stack = tf.pack([y_zeros, y_locations, y_zeros, y_locations], axis=1)

    x_reshaped_stack = tf.reshape(x_stack, (1, 1, feature_w, 4))
    y_reshaped_stack = tf.reshape(y_stack, (1, feature_h, 1, 4))

    # I <3 broadcasting
    raw_anchor_shifts = tf.add(x_reshaped_stack, y_reshaped_stack)

    # Transform to scaled image coordinates
    less_raw_anchor_shifts = feature_stride * raw_anchor_shifts

    # Add extra dimensions to anchors for proper broadcasting
    expanded_anchors = tf.expand_dims(tf.expand_dims(tf.constant(anchors),dim=1),dim=1) - [1.]
    return tf.add(less_raw_anchor_shifts, expanded_anchors, name="shifted_anchors")


def regressAnchors(anchors, bbox_regression, axis=-1):
    """ Given preliminary bounding boxes, regress them to their final location

    The bounding box regressions are outputs of convolutional layers and of
    the form (dx,dy,dw,dh), where dw and dh are the natural logarithms of
    the desired changes in width and height and where dx and dy are unscaled
    x and y displacements; they must be mulitiplied by the width and height of a given
    bounding box to arrive at the actual displacements.
    """
    # Our shifted anchors come in the shape (numBaseAnchors, feat_h, feat_w, 4)
    # We want to separate out the 4 regression variables, {dx,dy,dw,dh}, out into their
    # own fith dimension for the output of the regression as well!

    # (Actually, we're going to assume that the regressions are ALSO in the form
    # (numBaseAnchors, feat_h, feat_w, 4) !  This can be enforced at another stage.

    x1, y1, x2, y2 = tf.unpack(anchors,num=4, axis=axis)
    dx, dy, dw, dh = tf.unpack(bbox_regression,num=4, axis=axis)

    # We must get the anchors into the same width/height x/y format as the
    # bbox_regressions
    w = x2 - x1 + [1.]
    h = y2 - y1 + [1.]
    x = w / [2.] + x1
    y = h / [2.] + y1

    # The dx and dy given by the regression must be scaled by w and h and added
    # to the anchors
    x_new = dx*w + x
    y_new = dy*h + y

    # Since logarithms of the values in question are easier to learn (no regression means
    # a logarithm of the change being zero), we learn the logarithms of h, w.
    w_new = tf.exp(dw) * w
    h_new = tf.exp(dh) * h

    # Transform back to the original (x1,y1,x2,y2) coordinate system
    x1_final = x_new - [.5] * w_new
    y1_final = y_new - [.5] * h_new
    x2_final = x_new + [.5] * w_new
    y2_final = y_new + [.5] * h_new

    # Stack our anchors back up
    regressedAnchors = tf.pack([x1_final, y1_final, x2_final, y2_final], axis,
            name="regressed_anchors")

    # The output shape is the same as the input shape;  Output shape is
    # regressedAnchors.shape = (numBaseAnchors, feature_h, feature_w, 4)
    return regressedAnchors


def generateAnchors(ratios=[.5, 1, 2], scales=[8, 16, 32], base=[1, 1, 16, 16]):
    # Generates a list of anchors based on a list of aspect ratios, a base
    # size, and a list of scales.  Anchors are of the form [x0,y0,x1,y1], where
    # x0,y0 are the coordinates of the input pixel on the bottom-left of the anchor and
    # x1,y1 are the coordinates of the input pixel on the bottom-right of the
    # anchor.

    return scaleUpAnchors(aspectRatioAnchors(base, ratios), scales)

# Main inner methods for generateAnchors


def aspectRatioAnchors(baseAnchor, aspectRatios):
    # Produces anchors of various aspect ratios based on a base size
    wh_base = toWidthHeight(baseAnchor)
    anchors = []
    for ratio in aspectRatios:
        w = np.round(np.sqrt(wh_base[0] * wh_base[1] * ratio))
        h = np.round(w / ratio)
        anchors.append(toWidthHeightInverse([w, h, wh_base[2], wh_base[3]]))

    return anchors


def scaleUpAnchors(anchors, scales):
    # Takes the given anchors and creates a new list of anchors, with one at
    # each scale.
    out_anchors = []
    for anchor in anchors:
        wh_anchor = toWidthHeight(anchor)
        for scale in scales:
            scaled_wh_anchor = [wh_anchor[0] * scale, wh_anchor[1] * scale,
                                wh_anchor[2], wh_anchor[3]]
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
    width = anchor[2] - anchor[0] + 1
    height = anchor[3] - anchor[1] + 1

    x = .5 * (anchor[0] + anchor[2])
    y = .5 * (anchor[1] + anchor[3])

    return [width, height, x, y]


def toWidthHeightInverse(wh):
    # Transforms the output of toWidthHeight, in the form [width,height,x,y],
    # where width and height are the width and height of an anchor and x,y
    # the x and y coordinates of its center, back to the regular representation
    anchor = [0., 0., 0., 0.]

    # We need to subtract 1 from the widths and the heights because they are heights
    # and widths of the areas taken up by the pixels, not of the number of
    # pixels themselves.
    anchor[0] = wh[2] - .5 * (wh[0] - 1)
    anchor[1] = wh[3] - .5 * (wh[1] - 1)
    anchor[2] = wh[2] + .5 * (wh[0] - 1)
    anchor[3] = wh[3] + .5 * (wh[1] - 1)

    return anchor
