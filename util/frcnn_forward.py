import tensorflow as tf
import matplotlib.pyplot as plt
from networks.vgg19 import Vgg19
from util import settings as s
from layers.custom_layers import roi_pooling_layer
from networks import cls
from networks import rpn
from util import utils
import skimage.transform
import numpy as np
import sys

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
    with tf.variable_scope('frcnn') as scope:
        pooled_regions = roi_pooling_layer(features, image_attributes, proposed_regions, 
                                        pooled_h, pooled_w, 16,name='roi_pooling_layer')[0]
    print("RoI pooling set up!")
    cls_scores, bbox_reg = cls.setUp(pooled_regions, pooled_h, pooled_w, feature_channels,
                                        namespace="frcnn")
    
    bboxes = rpn.regressAnchors(proposed_regions, bbox_reg, axis=1)

    return bboxes, cls_scores, rpn_scores

def process_image(imPath):
    """ Loads and preprocesses an image located in path 'image'"""
    scaleToMin = 600.
    scaleToMax = 1000.

    img = skimage.io.imread(imPath)
    shorterDim = np.argmin(img.shape[0:2])
    longerDim = 0 if shorterDim==1 else 1

    img =- np.array(s.FRCNN_MEAN)

    ratio = scaleToMin / img.shape[shorterDim]

    if img.shape[longerDim] * ratio > scaleToMax:
        ratio = scaleToMax / img.shape[longerDim]

    # resize image using ratio
    resized_image = skimage.transform.rescale(img, ratio)
    # resized_image = skimage.linear_interpolation(information)
    return resized_image, np.array([img.shape[1], img.shape[0], ratio])

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
    utils.downloadFasterRcnn()

    # The three variables to the left are tensor objects that will contain the values we
    # want to extract from the network
    print("Setting up Faster R-CNN...")
    bboxes_t, cls_scores_t, bbox_scores_t = faster_rcnn(net_img_input, net_img_attr_input)
    
    print("Faster R-CNN has been set up")

    output = []
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(tf.global_variable_initializer())
            print("Running Faster R-CNN on input...")
            output = sess.run(  [bboxes_t, cls_scores_t, bbox_scores_t],
                                feed_dict={ net_img_input : image, 
                                            net_img_attr_input : image_attr})
        sess.close()
    # Let bboxes be regressed in the model.  Fuck doing it out here, the anchor regression is
    # written in terms of tf operations ANYWAYS
    # func: ..networks.rpn.regressAnchors(proposal_regions, bbox_pred)
    print("Faster R-CNN successfully run")
    draw_boxes(img, bboxes, output[1], output[3])

def draw_boxes(img, bboxes, bbox_scores, cls_scores, thresh=0.5):
    """ Draw detected bounding boxes 
        Adapted from Faster R-CNN function vis_detections
    """
    inds = np.where(bbox_scores >= thresh)[0]
    if len(inds) == 0:
        return

    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(img, aspect='equal')
    for i in inds:
        bbox = bboxes[i,:]
        score = dets[i,:]
        
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
                        'p({} | box) >= {:.1f}').format(return_class(cls_scores), scores, thresh),
                        fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()

if __name__ == '__main__':
    demo(sys.argv[1])


