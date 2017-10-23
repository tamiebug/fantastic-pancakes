import numpy as np
import tensorflow as tf

from layers.custom_layers import nms
from layers.custom_layers import iou_labeler

from util import settings as s
from util.utils import easy_scope


def Rpn(features, image_attr, train_net=None, namespace="rpn"):
    """ Region proposal network.  Proposes regions to later be pooled and classified/regressed

    Inputs:
    features    - A tf.Tensor object of rank 4, dimensions (batch, height, width, channel),
        since this is the standard tensorflow order.

    image_attr  - A tf.Tensor object of rank 1, with values [img_h, img_w, scaling_factor],
        where these values are described below:
    train_net   - Can be set to either None (default), "TRAIN_RPN", or "TRAIN_R-CNN".  When
        set to one of the latter 'TRAIN_' settings, it initalizes the network differently,
        for training instead of for prediction.

        Output:
        A tf.tensor object of rank 2 with dimensions (num_rois, 4), where the second dimension
        is of the form {x0, y0, x1, y1}
    """
    train = False
    if train_net is not None:
        train = True

    def createConvLayer(bottom, name, stride=[1, 1, 1, 1]):
        """ Creates a conv layer given a name.

        Precondtions:
            Expects a tf.Variable with name model_scope/layer_scope/Weights
            and one with model_scope/layer_scope/Bias to already exist.

        Inputs:
            bottom  - A tf.Tensor containing activations
            name    - A string with a name for this layer
            stride  - A list containing the stride to apply for this convolution.
                        Most likely does not need to be changed from its default.
        Outputs:
            A tf.Tensor containing the output of the convolution.
        """
        with easy_scope(name, reuse=True):
            prevLayer = tf.nn.conv2d(bottom, tf.get_variable("Weights", trainable=train),
                    stride, padding="SAME")
            prevLayer = tf.nn.bias_add(prevLayer, tf.get_variable("Bias", trainable=train),
                    name="out")
        return prevLayer

    with easy_scope(namespace, reuse=True):
        layer3x3 = createConvLayer(features, "rpn_conv/3x3")
        layer3x3 = tf.nn.relu(layer3x3, "rpn_relu/3x3")

        # Region Proposal Network - Probabilities
        prevLayer = createConvLayer(layer3x3, "rpn_cls_score")

        # Assuming that feat_w = feat_h = 14, and that the number of anchors is 9,
        # we have the output should be of shape (9,14,14,2).

        # However, a tf.nn.conv2d cannot create batches out of thin air.  Hence, the
        # rpn_cls_score should create a (1, 14, 14, 9*2) instead, which we reshape to
        # (1, 14, 14, 2, 9), transpose to (9, 14, 14, 2, 1), then tf.squeeze the last
        # dimension out to arrive at the desired wonderful shape of (9, 14, 14,
        # 2).  The last dimension of rpn_cls_score is unpacked from (9*2) to (2,9) and
        # not (9,2) since this is how the weights imported from caffe are packed.

        with easy_scope("create_rpn_score_batches"), tf.device("/cpu:0"):
            feature_h = tf.shape(features)[1]
            feature_w = tf.shape(features)[2]
            prevLayer = tf.reshape(prevLayer, (1, feature_h, feature_w, 2, 9))
            prevLayer = tf.transpose(prevLayer, (4, 1, 2, 3, 0))
            prevLayer = tf.squeeze(prevLayer)

            if train_net is not "TRAIN_RPN":
                rpnScores = tf.nn.softmax(prevLayer, dim=-1, name="rpn_cls_prob_raw")
                _, rpnScores = tf.unstack(rpnScores, num=2, axis=-1)

            rpnScores = tf.identity(rpnScores, name="rpn_cls_prob")

        with tf.device("/gpu:0"):
            # Region Proposal Network - Bounding Box Proposal Regression
            rpnBboxPred = createConvLayer(layer3x3, "rpn_bbox_pred")

        with easy_scope("create_rpn_bbox_batches"), tf.device("/cpu:0"):
            # We want to reshape rpnBboxPred just like we did the scores.
            # Only difference is that we reshape to (9,14,14,4) instead of
            # (9,14,14,2) (in the case of feat_h=feat_w=14)

            prevLayer = tf.reshape(rpnBboxPred, (1, feature_h, feature_w, 9, 4))
            prevLayer = tf.transpose(prevLayer, (3, 1, 2, 4, 0))
            rpnBboxPred = tf.squeeze(prevLayer)

        if train_net is not "TRAIN_RPN":
            out = proposalLayer(
                s.DEF_FEATURE_STRIDE,
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

        else:
            return rpnScores, rpnBboxPred, feature_h, feature_w, image_attr


def proposalLayer(feature_stride, iou_threshold, pre_nms_keep, post_nms_keep,
                  scores, bbox_regressions, feature_h, feature_w, image_attr,
                  minimum_dim, device="/cpu:0"):
    """Propose the actual regions given outputs of previous conv layers

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

    scores		-- List of scores ranking the aformentioned regions after regressions.

    feature_h		-- Height of feature map

    feature_w		-- Width of feature map

    image_attr          -- tf.Tensor object containing the following values in order
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

    with easy_scope(name="proposal_layer"), tf.device(device):
        baseAnchors = generateAnchors(ratios=[2, 1, .5])

        shiftedAnchors = generateShiftedAnchors(baseAnchors, feature_h, feature_w,
                feature_stride)

        regressedAnchors = regressAnchors(shiftedAnchors, bbox_regressions)

        clippedAnchors = clipRegions(regressedAnchors, image_attr)

        p_anchors, p_scores, _ = prunedScoresAndAnchors(clippedAnchors,
                scores, minimum_dim, image_attr)

        pre_nms_keep = 6000
        top_scores, top_score_indices = tf.nn.top_k(p_scores, k=pre_nms_keep, name="top_scores")

        top_anchors = tf.gather(p_anchors, top_score_indices, name="top_anchors", axis=0)

        # We want nms to keep everything that passes the IoU test
        post_nms_indices = nms(top_anchors, top_scores,
                            post_nms_keep, iou_threshold=iou_threshold, name="post_nms_indices")

        final_anchors = tf.gather(top_anchors, post_nms_indices, axis=0,
                name="proposal_regions")
        final_scores = tf.gather(top_scores, post_nms_indices, axis=0,
                name="proposal_region_scores")

    return final_anchors, final_scores


def prunedScoresAndAnchors(anchors, scores, minimum_dim, im_attr):
    """ Return list of anchors and scores larger than a given minimum size

    Inputs:
        anchors -- tf.Tensor of shape (numAnchors, feat_h, feat_w, 4) containing
            the boxes to be pruned.
        scores -- tf.Tensor of shape (numAnchors, feat_h, feat_w) containing
            objectness scores for each of the boxes.
    Output:
        We output tensors of shape (?,2) for scores_gathered and (?,4) for
        anchors_gathered.  The indices of the chosen scores is also returned.
    """
    anchors = tf.transpose(anchors, (1, 2, 0, 3))
    scores = tf.transpose(scores, (1, 2, 0))
    anchors = tf.reshape(anchors, (-1, 4))
    scores = tf.reshape(scores, (-1,))

    x1, y1, x2, y2 = tf.unstack(anchors, num=4, axis=-1)

    w = x2 - x1 + 1.
    h = y2 - y1 + 1.

    # Gotta scale the minimum_dim by the scale factor before use
    minimum_dim = tf.constant([s.DEF_MIN_PROPOSAL_DIMS], dtype=tf.float32)
    minimum_dim = minimum_dim * im_attr[2]

    width_suffices = tf.greater_equal(w, minimum_dim, name="geqw")
    height_suffices = tf.greater_equal(h, minimum_dim, name="geqh")

    both_suffice = tf.logical_and(width_suffices, height_suffices)
    indices = tf.where(both_suffice)

    # The actual grabbing of indexed values happens here
    anchors_gathered = tf.gather_nd(anchors, indices, name="pruned_anchors")
    scores_gathered = tf.gather_nd(scores, indices, name="pruned_scores")
    return anchors_gathered, scores_gathered, indices


def clipRegions(anchors, image_attr, axis=-1):
    """ Clip anchors so that all lie entirely within image """

    # Input anchors will be of shape
    # (numBaseAnchors, feature_h, feature_w, 4)

    with tf.device("/cpu:0"):
        x1, y1, x2, y2 = tf.unstack(anchors, num=4, axis=axis)

        zero = tf.constant([0.])

        max_x = [tf.subtract(image_attr[1] * image_attr[2], tf.constant([1.]), name="clip_img_w")]
        max_y = [tf.subtract(image_attr[0] * image_attr[2], tf.constant([1.]), name="clip_img_h")]

        x1_clipped = tf.minimum(tf.maximum(zero, x1), max_x)
        x2_clipped = tf.minimum(tf.maximum(zero, x2), max_x)
        y1_clipped = tf.minimum(tf.maximum(zero, y1), max_y)
        y2_clipped = tf.minimum(tf.maximum(zero, y2), max_y)

        # Pack 'em back up
        retVal = tf.stack([x1_clipped, y1_clipped, x2_clipped, y2_clipped],
                axis, name="clipped_anchors")

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

    x_stack = tf.stack([x_locations, x_zeros, x_locations, x_zeros], axis=1)
    y_stack = tf.stack([y_zeros, y_locations, y_zeros, y_locations], axis=1)

    x_reshaped_stack = tf.reshape(x_stack, (1, 1, feature_w, 4))
    y_reshaped_stack = tf.reshape(y_stack, (1, feature_h, 1, 4))

    # I <3 broadcasting
    raw_anchor_shifts = tf.add(x_reshaped_stack, y_reshaped_stack)

    # Transform to scaled image coordinates
    less_raw_anchor_shifts = feature_stride * raw_anchor_shifts

    # Add extra dimensions to anchors for proper broadcasting
    expanded_anchors = tf.expand_dims(tf.expand_dims(tf.constant(anchors), axis=1), axis=1) - [1.]
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

    with tf.device("/cpu:0"):
        x1, y1, x2, y2 = tf.unstack(anchors, num=4, axis=axis)
        dx, dy, dw, dh = tf.unstack(bbox_regression, num=4, axis=axis)

        # We must get the anchors into the same width/height x/y format as the
        # bbox_regressions
        w = x2 - x1 + [1.]
        h = y2 - y1 + [1.]
        x = w / [2.] + x1
        y = h / [2.] + y1

        # The dx and dy given by the regression must be scaled by w and h and added
        # to the anchors
        x_new = dx * w + x
        y_new = dy * h + y

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
        regressedAnchors = tf.stack([x1_final, y1_final, x2_final, y2_final], axis,
                name="regressed_anchors")

    # The output shape is the same as the input shape;  Output shape is
    # regressedAnchors.shape = (numBaseAnchors, feature_h, feature_w, 4)
    return regressedAnchors


def calculateRegressions(anchors, boxes, axis=-1):
    """ Given anchor boxes and ground truth boxes, find regressions

    Inputs:
        anchors -- tf.Tensor of shape (...,4) containing anchors with respect to which
            we desire to find regressions
        boxes -- tf.Tensor of same shape as anchors which we are trying to regress to

    Output:
        The output of this function are compatible with the output of the convolutional
        layers and of the form (dx, dy, dw, dh).  If the output of this function is
        fed into regressAnchors, we should recover the original ground truth boxes once more.
    """

    with tf.device("/cpu:0"):
        ax1, ay1, ax2, ay2 = tf.unstack(anchors, num=4, axis=-1)
        bx1, by1, bx2, by2 = tf.unstack(anchors, num=4, axis=-1)

        # Calculate the center coordinates for both boxes
        aw = ax2 - ax1 + [1.]
        ah = ay2 - ay1 + [1.]
        ax = aw / [2.] + ax1
        ay = ah / [2.] + ay1

        bw = bx2 - bx1 + [1.]
        bh = by2 - by1 + [1.]
        bx = bw / [2.] + bx1
        by = bh / [2.] + by1

        # We are regressing from the anchor to the box
        dx = (bx - ax) / aw
        dy = (by - ay) / ah
        dw = tf.log(bw / aw)
        dh = tf.log(bh / ah)

        # Stack the regressions up
        regressions = tf.stack([dx, dy, dw, dh], axis, name="calculated_regressions")
    return regressions


def generateAnchors(ratios=[.5, 1, 2], scales=[8, 16, 32], base=[1, 1, 16, 16]):
    """ Generates a list of anchors based on the input parameters

    Inputs:
        ratios -- List of desired aspect ratios
        scales -- List of desired scales; these are multipliers applied to the
            "base" anchor
        base -- List of four integers of the form [x0,y0,x1,y1], representing the
            bottom left and top right corner of the base anchor from which all the
            others are derived through changes in aspect ratios and scales

    Output:
        len(ratios) * len(scales) anchors (each one of the form [x0,y0,x1,y1]
    """
    return scaleUpAnchors(aspectRatioAnchors(base, ratios), scales)

# Main inner methods for generateAnchors


def aspectRatioAnchors(baseAnchor, aspectRatios):
    """Produces anchors of various aspect ratios based on a base size """
    wh_base = toWidthHeight(baseAnchor)
    anchors = []
    for ratio in aspectRatios:
        w = np.round(np.sqrt(wh_base[0] * wh_base[1] * ratio))
        h = np.round(w / ratio)
        anchors.append(toWidthHeightInverse([w, h, wh_base[2], wh_base[3]]))

    return anchors


def scaleUpAnchors(anchors, scales):
    """Takes the given anchor list and returns a new one, with one anchor at each scale."""
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
    """Transforms an anchor in [x0,y0,x1,y1] format to [w,h,x,y]
    Where [w,h,x,y] stans for width, height, center x coord, center y coordinate
    """
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
    """ Transforms from [w, h, x, y] to [x0,y0,x1,y1] format"""
    anchor = [0., 0., 0., 0.]

    # We need to subtract 1 from the widths and the heights because they are heights
    # and widths of the areas taken up by the pixels, not of the number of
    # pixels themselves.
    anchor[0] = wh[2] - .5 * (wh[0] - 1)
    anchor[1] = wh[3] - .5 * (wh[1] - 1)
    anchor[2] = wh[2] + .5 * (wh[0] - 1)
    anchor[3] = wh[3] + .5 * (wh[1] - 1)

    return anchor


def sampleBoxes(labeled_boxes, num_classes, mini_batch_size):
    """ Samples labeled_boxes, returning randomly selected positive and negative examples

    Inputs:
        labeled_boxes -- a tf.Tensor of shape (n, 5), with rows of the form
            [x0, y0, x1, y1, class], where class is a classification of the box
            such that background is indicated by class=num_classes
        num_classes -- number of classes (not including the background class)
        mini_batch_size -- number of indices to return

    Output:
        Two lists of indices, one positives and the other negatives, randomly sampled
        from labeled_boxes
    """
    positive_box_indices = np.where(labeled_boxes[:, 4] < (num_classes - .5))
    negative_box_indices = np.logical_not(positive_box_indices)

    num_pos = positive_box_indices.shape[0]
    num_neg = negative_box_indices.shape[0]

    # We want to have a ratio of positive to negative samples of up to 1:1,
    # padding the positives with negatives if they are not enough.
    pos_to_choose = min(num_pos, num_neg, mini_batch_size // 2)
    neg_to_choose = min(num_neg, mini_batch_size - pos_to_choose)

    # Now we need to randomly sample from both
    pos_Idx = np.random.choice(num_pos, pos_to_choose, replace=False)
    neg_Idx = np.random.choice(num_neg, neg_to_choose, replace=False)

    return pos_Idx, neg_Idx


def calculateRpnLoss(rpnRawScores, rpnBboxPred, feature_h, feature_w, image_attr, gt_boxes):
    """ Calculates the loss for the region proposal network

    Inputs:
        rpnRawScores -- tf.Tensor object containing the objectness scores for each region
            before application of softmax.
        rpnBboxPred -- tf.Tensor object containing the bounding-box regressions for each
            region.  Must be of a shape compatible with rpnRawScores, i.e. only differing
            in shape int he last dimension
        feature_h -- Height of the convolutional input to the RPN
        feature_w -- Width of the convolutional input to the RPN
        gt_boxes -- Ground-truth boxes with which we are calculating the loss with respect to.
            Must be in the format (num_gt_boxes, 5), where the rows are of the form
            [x0, y0, x1, y1, class], where class is the category to which each ground truth
            box belongs to

    Output:
        The loss for this minibatch
    """

    iou_threshold = s.DEF_IOU_THRESHOLD_TRAIN
    pre_nms_keep = s.DEF_PRE_NMS_KEEP
    post_nms_keep = s.DEF_POST_NMS_KEEP_TRAIN
    num_classes = 2
    mini_batch_size = 128
    minimum_dim = s.DEF_MIN_PROPOSAL_DIMS

    with easy_scope(name="proposal_layer_test"), tf.device("/cpu:0"):
        baseAnchors = generateAnchors(ratios=[2, 1, .5])
        shiftedAnchors = generateShiftedAnchors(baseAnchors, feature_h, feature_w,
                s.DEF_FEATURE_STRIDE)
        regressedAnchors = regressAnchors(shiftedAnchors, rpnBboxPred)
        clippedAnchors = clipRegions(regressedAnchors, image_attr)

        # Softmax for rpnScores
        rpnScores = tf.nn.softmax(rpnRawScores, dim=-1, name="rpn_cls_prob_raw")
        _, rpnScores = tf.unstack(rpnScores, num=2, axis=-1)

        p_anchors, p_scores, p_indices = prunedScoresAndAnchors(
            clippedAnchors, rpnScores, minimum_dim, image_attr)

        top_scores, top_score_indices = tf.nn.top_k(p_scores, k=pre_nms_keep, name="top_scores")

        # We are going to need the p_indices to pass through all of the refinement that the
        # top_scores and top_anchors go through so that we can use them to select unregressed
        # anchors, original regression values, and the raw scores down the line.
        top_indices = tf.gather(p_indices, top_score_indices)

        # Modifying the top_score_indices to be able to be used with gather_nd
        top_score_indices = tf.cast(top_score_indices, tf.int32, name="top_scores_indices")
        top_score_indices = tf.expand_dims(top_score_indices, axis=1,
            name="top_score_indices_expanded")

        top_anchors = tf.gather_nd(p_anchors, top_score_indices, name="top_anchors")

        # We want nms to keep everything that passes the IoU test
        post_nms_indices = nms(top_anchors, top_scores, post_nms_keep,
            iou_threshold=iou_threshold, name="post_nms_indices")

        # These are the indices used to collect the unti-now unmodified collections:
        #       - The original regressions
        #       - The raw, pre-softmax scores (of shape (-1, 2))
        #       - The original anchors, before application of regressions
        final_indices = tf.gather(top_indices, post_nms_indices)
        final_indices = tf.expand_dims(final_indices, axis=1,
            name="final_indices")

        # Expanding nms_indices for use with tf.gather_nd
        post_nms_indices = tf.expand_dims(post_nms_indices, axis=1,
            name="post_nms_indices_expanded")

        final_anchors = tf.gather_nd(top_anchors, post_nms_indices,
                name="proposal_regions")

        # The original anchors
        final_raw_anchors = tf.gather_nd(tf.reshape(
            shiftedAnchors, (-1, 4)), final_indices)

        # Original regression values for loss calculation
        final_raw_regressions = tf.gather_nd(tf.reshape(
            rpnBboxPred, (-1, 4)), final_indices)

        # Original objectness scores to be used with softmax_cross_entropy
        final_raw_scores = tf.gather_nd(tf.reshape(
            rpnRawScores, (-1, 2)), final_indices)

        labeled_boxes = iou_labeler(final_anchors, gt_boxes, iou_threshold)

        # Sample boxes and raw scores for loss
        posIdx, negIdx = tf.py_func(lambda x: sampleBoxes(x, num_classes, mini_batch_size),
            labeled_boxes, [tf.float32, tf.float32])

        positive_raw_scores = tf.gather_nd(final_raw_scores, posIdx)
        negative_raw_scores = tf.gather_nd(final_raw_scores, negIdx)

        # There is no regression loss for negative examples.  For the positives, we need
        # to find the gt regression from anchor to gt boxes
        positive_anchors = tf.gather_nd(final_raw_anchors, posIdx)
        positive_gt_boxes = tf.gather(gt_boxes, labeled_boxes[:, 4])
        positive_gt_regs = calculateRegressions(positive_anchors, positive_gt_boxes, axis=-1)
        positive_raw_regressions = tf.gather_nd(final_raw_regressions, posIdx)

        # Flatten regressions before passing into the huber loss function
        flat_pred_regs = tf.reshape(positive_raw_regressions, [-1])
        flat_gt_regs = tf.reshape(positive_gt_regs, [-1])

        reg_loss = tf.losses.huber_loss(flat_pred_regs, flat_gt_regs, delta=1.0)

        # Class-agnostic log loss for positive examples
        # Need to create a whole bunch of [0,1]s of the right length
        num_pos = tf.shape_n(positive_raw_scores)[0]
        gt_onehot = tf.one_hot(tf.ones(num_pos, dtype=tf.int32), 2)
        cls_loss_pos = tf.losses.softmax_cross_entropy(gt_onehot, positive_raw_scores)

        # Log-loss for the negative examples
        num_neg = tf.shape_n(negative_raw_scores)[0]
        gt_onehot = tf.one_hot(tf.zeros(num_pos, dtype=tf.int32), 2)
        cls_loss_neg = tf.losses.softmax_cross_entropy(gt_onehot, negative_raw_scores)

        # Adding up the losses
        reg_loss /= num_pos
        cls_loss = (cls_loss_pos / num_pos) + (cls_loss_neg / num_neg)

        return reg_loss + cls_loss
