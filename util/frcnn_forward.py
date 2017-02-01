import tensorflow as tf
import matplotlib.pyplot as plt
from networks.vgg19 import Vgg19
from util import settings as s
from util.utils import easy_scope
from layers.custom_layers import roi_pooling_layer
from layers.custom_layers import nms
from networks import cls
from networks import rpn
from util import utils
import skimage.transform
import numpy as np
import sys
from itertools import izip
import networks.loadNetVars
import cv2

def faster_rcnn(image, image_attributes):
    """ Builds a Faster R-CNN network
        
    Inputs:
        image           - tf.Tensor object containing image to be processed
        image_attribures- tf.Tensor object containing image height, width, and
                            scaling factor used to resize original image
    Outputs:
        bboxes          - tf.Tensor object with bounding boxes for detections
        cls_scores      - tf.Tensor object with classification scores for each
                            category in VOC 2007
        rpn_scores      - tf.Tensor with objectness scores, ranking each detection's
                            likelihood to be an object.
    """
    pooled_h = 7
    pooled_w = 7
    feature_channels = 512 # Property of vgg16 network
    num_classes = 21
    condifence_threshold = 0.8

    vgg16_base = Vgg19('frcnn')
    vgg16_base.buildGraph(image, train=False,
            weightsPath=s.DEF_FRCNN_WEIGHTS_PATH,
            biasesPath=s.DEF_FRCNN_BIASES_PATH,
            cutoff=['conv3_4', 'relu3_4', 'conv4_4', 'relu4_4', 'conv5_4', 'relu5_4',
                    'pool5','fc6','relu6','fc7', 'relu7', 'fc8', 'prob']
            )
    print("Base vgg16 network initialized!")
    features = vgg16_base.layers['relu5_3']
    proposed_regions, rpn_scores = rpn.Rpn(features, image_attributes,
                                    train=False, namespace='frcnn')
    print("Region Proposal Network set up!")

    with easy_scope('frcnn'):
        pooled_regions = roi_pooling_layer(features, proposed_regions, 
                                        pooled_h, pooled_w, 16,name='roi_pooling_layer')[0]
    print("RoI pooling set up!")
    bbox_reg, cls_scores = cls.setUp(pooled_regions, pooled_h, pooled_w, feature_channels,
                                        namespace="frcnn")
    
    # cls_score is (300,21) ; bbox_reg is (300,84)
    bbox_reg = tf.reshape(bbox_reg, (-1, 21, 4))

    # set proposed_regions shape to (300,1,4)
    proposed_regions_reshape = tf.expand_dims(proposed_regions, dim=1)
    reg_roi = rpn.regressAnchors(proposed_regions_reshape, bbox_reg, axis=-1)

    # Clip all regions to image boundaries
    reg_roi = rpn.clipRegions(reg_roi, img_attributes,axis=2)

    # Throw away everything not meeting a certain score threshold
    geq_indices = tf.greater_equal(cls_scores, [confidence_threshold])
    reg_roi = tf.gather_nd(reg_roi, geq_indices)
    cls_scores = tf.gather_nd(cls_scores geq_indices)
    
    # Unapck both the regions and scores by class.
    reg_rois = tf.unpack(reg_roi, num=21, axis=1)
    bbox_scores = tf.unpack(cls_scores, num=21, axis=1) 
        
    # There are 20 classes, each in their own list.  Background is not stored
    out_scores = [[] for _ in xrange(20)]
    out_regions = [[] for _ in xrange(20)]

    # We skip the first class since it is the background class.
    for i, (regs, scores) in enumerate(izip(reg_rois, bbox_scores)[1:]):
        # Perform NMS, but keep all of the indices (#indices < 300)
        inds = nms(regs, scores, 300, iou_threshold=0.3)
        regs = tf.gather_nd(regs, inds)
        scores = tf.gather_nd(scores, inds)
        out_scores[i] = scores 
        out_regions[i] = regs

    return out_regions, out_scores

def process_image(imPath):
    """ Loads and preprocesses an image located in path 'image'"""
    scaleToMin = 600.
    scaleToMax = 1000.

    img = skimage.io.imread(imPath)
    img = img - np.array(s.FRCNN_MEAN)
    shorterDim = np.argmin(img.shape[:-1])
    longerDim = 0 if shorterDim==1 else 1
    ratio = scaleToMin / img.shape[shorterDim]

    if img.shape[longerDim] * ratio > scaleToMax:
        ratio = scaleToMax / img.shape[longerDim]

    # resize image using ratio
    resized_image = cv2.resize(img, None, None,fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    return resized_image, np.array([img.shape[0], img.shape[1], ratio])

def demo(img):
    """ Performs a forward pass through the Faster RCNN network.

    Inputs:
    img             -- Path to an image file

    Outputs:
    bounding_boxes  -- List of detections in the image
    scores          -- List of scores for each detection, one score for each category
                        in the pascal VOC dataset
    """

    net_img_input = tf.placeholder( "float", name="image_input" )
    net_img_attr_input = tf.placeholder( "float" , name="image_attr" )
    image, image_attr = process_image(img) 
    
    print("Checking for weights/biases, downloading them if they do not exist...")
    #utils.downloadFasterRcnn()

    
    # The three variables to the left are tensor objects that will contain the values we
    # want to extract from the network
    print("Setting up Faster R-CNN...")
    features, rpn_regions, rpn_scores = faster_rcnn(net_img_input, net_img_attr_input)

    output = []

    def find_ops_output_in_list(operations,ops_list):
        ret = []
        for full_op in ops_list:
            for op in operations:
                if op in full_op.name:
                    ret.append(full_op.name + ":0")
        return ret
            
    with tf.Session() as sess :
        with tf.device("/cpu:0") as dev:
            ops = sess.graph.get_operations()
            sess.run(tf.initialize_all_variables())
            for op in ops:
                #print(op.name)
                pass

            opsWanted = ['DEBUG_1']
            output = sess.run(  #find_ops_output_in_list(opsWanted, ops).append(rpn_scores),
                                [rpn_regions, rpn_scores],
                                feed_dict = {   net_img_input: np.expand_dims(image,0),
                                                net_img_attr_input : image_attr }
    
                            )
        print("printing op results from ops in {} for debugging".format(opsWanted))
        print output[0]
        print output[1]

    def num_trues(array, txt):
        count = 0
        for ele in np.nditer(array):
            if ele:
                count += 1
        print("{} had {} trues out of {} elements".format(txt, count, array.size))

    """
    print("Faster R-CNN successfully run")

    print("Generating visualizations...")

    # Code adapted from py-faster-rcnn/test/demo.py

    classes = ( '__background__',
                'airplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
    for _class, cls_detections in izip(classes, output[4]):
        if _class == '__background__':
            continue
        draw_boxes(img, cls_detections, _class)
        
    """
    
def draw_boxes(img, detections, name, thresh=0.8):
    """ Draw detected bounding boxes 
        Adapted from Faster R-CNN function vis_detections
    """
    inds = np.where(detections[:,-1] >= thresh)[0]
    if len(inds) == 0:
        return

    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(img, aspect='equal')
    for i in inds:
        bbox = detections[i, :4]
        score = detections[i, -1]
        
        ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1], fill=False,
                    edgecolor='red', linewidth=3.5)
                )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(return_class(cls_scores), scores),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        ax.set_title(('{} detections with '
                        'p({} | box) >= {:.1f}').format(name, name, thresh),
                        fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()

if __name__ == '__main__':
    demo(sys.argv[1])


