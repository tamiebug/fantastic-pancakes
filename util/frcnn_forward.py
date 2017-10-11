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

import networks.loadNetVars
import cv2

def faster_rcnn(image, image_attributes):
    """ Builds a Faster R-CNN network
        
    Inputs:
        image           - tf.Tensor object containing image to be processed
        image_attribures- tf.Tensor object containing image height, width, and
                            scaling factor used to resize original image
    Outputs:
        out_regions     - list of tf.Tensor objects with bounding boxes for detections, 
                            sans background class
        out_scores      - list of tf.Tensor objects with classification scores for each
                            category in VOC 2007, sans background
    """
    pooled_h = 7
    pooled_w = 7
    feature_channels = 512 # Property of vgg16 network
    num_classes = 21
    confidence_threshold = 0.8
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
        pooled_regions = roi_pooling_layer(tf.squeeze(features), image_attributes, proposed_regions,
                                        pooled_h, pooled_w, 16,name='roi_pooling_layer')
    print("RoI pooling set up!")
    bbox_reg, cls_scores = cls.setUp(pooled_regions, pooled_h, pooled_w, feature_channels,
                                        namespace="frcnn")
    with easy_scope('frcnn'):
        with easy_scope('reshape_cls_output'):
            # cls_score is (300,21) ; bbox_reg is (300,84)
            bbox_reg = tf.reshape(bbox_reg, (-1, 21, 4))

            # Set proposed_regions shape to (300,1,4)
            proposed_regions_reshape = tf.expand_dims(proposed_regions, axis=1)

            # Rescale the Regions of Interest to the proper scale
            proposed_regions_reshape = proposed_regions_reshape / image_attributes[2]

        with easy_scope('clip_regress_unpack_output'):
            # Regress the Regions of Interest into class-specific detection boxes
            reg_roi = rpn.regressAnchors(proposed_regions_reshape, bbox_reg, axis=-1)

            # Clip all regions to image boundaries
            reg_roi = rpn.clipRegions(reg_roi, image_attributes, axis=-1)
            
            # Unpack both the regions and scores by class
            reg_rois = tf.unstack(reg_roi, num=21, axis=1)
            bbox_scores = tf.unstack(cls_scores, num=21, axis=1) 
                
        with easy_scope('non_max_suppression'):
            # There are 20 classes, each in their own list.  Background is not stored
            out_scores = [[] for _ in range(20)]
            out_regions = [[] for _ in range(20)]

            # We skip the first class since it is the background class.
            for i, (regs, scores) in enumerate(zip(reg_rois[1:], bbox_scores[1:])):
                # Perform NMS, but keep all of the indices (#indices < 300)
                inds = nms(regs, scores, 300, iou_threshold=0.3)
                regs = tf.gather(regs, inds)
                scores = tf.gather(scores, inds)
                out_scores[i] = scores 
                out_regions[i] = regs
            return out_regions, out_scores

def process_image(imPath):
    """ Loads and preprocesses an image located in path 'image'"""
    scaleToMin = 600.
    scaleToMax = 1000.

    img = skimage.io.imread(imPath)
    #img = np.load("/home/tamie/Downloads/rawest_img.npz")['arr_0']
    img = img - np.array(s.FRCNN_MEAN)
    shorterDim = np.argmin(img.shape[:-1])
    longerDim = 0 if shorterDim==1 else 1
    ratio = scaleToMin / img.shape[shorterDim]

    if img.shape[longerDim] * ratio > scaleToMax:
        ratio = scaleToMax / img.shape[longerDim]

    # resize image using ratio
    resized_image = cv2.resize(img, None, None,fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    return resized_image, np.array([img.shape[0], img.shape[1], ratio])

def demo(img, threshold=0.5, gpu_fraction=1.0):
    """ Performs a forward pass through the Faster RCNN network.

    Inputs:
    img             -- Path to an image file

    Outputs:
    bounding_boxes  -- List of detections in the image
    scores          -- List of scores for each detection, one score for each category
                        in the pascal VOC dataset
    threshold       -- Threshold for object detection, can be between 0 and 1 inclusive.
                        The greater the value, the more picky the model will be with detections.
    """

    net_img_input = tf.placeholder("float", name="image_input", shape=(1,600,800,3))
    net_img_attr_input = tf.placeholder("float" , name="image_attr")
    
    print("Checking for weights/biases, downloading them if they do not exist...")
    utils.grabFasterRCNNParams()

    
    # The three variables to the left are tensor objects that will contain the values we
    # want to extract from the network
    print("Setting up Faster R-CNN...")
    out_regions, out_scores = faster_rcnn(net_img_input, net_img_attr_input)

    output = []
    config = tf.ConfigProto(log_device_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    print("Using GPU memory fraction {}".format(gpu_fraction))
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(s.add_root('test/'), sess.graph)
        image, image_attr = process_image(img)
        image = np.expand_dims(image, 0)
        sess.run(tf.global_variables_initializer())

        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) 
        output = sess.run([out_regions, out_scores], 
                feed_dict={net_img_input : image, net_img_attr_input : image_attr},
                run_metadata=run_metadata,
                options=run_options,
                )
        #writer.add_summary(summary)
        writer.add_run_metadata(run_metadata, "cats")
    print("Faster R-CNN successfully run")
    print("Generating visualizations...")

    # Code adapted from py-faster-rcnn/test/demo.py
    classes = ( '__background__',
                'airplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
    draw_boxes(img, output[0], output[1], classes[1:], thresh=threshold)
    
def draw_boxes(img, regions, scores, names, thresh=0.5):
    """ 
    Draw detected bounding boxes 
    Adapted from Faster R-CNN function vis_detections

    Inputs:
        img     - Input image to be classified
        regions - Object detection regions in the image
        scores  - Object detection scores for regions in the image
        names   - Class names
    """

    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(skimage.io.imread(img), aspect='equal')

    total_detections = 0
    print(("Shapes are {} , {} , and {}".format(len(regions), len(scores), len(names))))
    
    for class_regions, class_scores, class_name in zip(regions, scores, names):
        inds = np.where(np.greater_equal(class_scores, thresh))[0]
        if len(inds) == 0:
            continue
        for bbox, score in zip(class_regions[inds], class_scores[inds]):
            total_detections += 1
            ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=3.5)
                    )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    ax.set_title(('{} detections with '
                    'p(class | box) >= {:.1f}').format(total_detections, thresh),
                    fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

if __name__ == '__main__':
    demo(sys.argv[1])
