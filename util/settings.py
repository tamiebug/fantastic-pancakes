import os

# The parent directory of the settings directory is the home directory of the project.
def add_root(path):
    ROOT_DIR = os.path.abspath(os.path.join(
                os.path.dirname(__file__), os.pardir))
    return os.path.join(ROOT_DIR,path)

VGG_MEAN = [103.939, 116.779, 123.68]

# copy from FRCNN
FRCNN_MEAN = [[[102.9801, 115.9465, 112.7717]]]

DEF_CAFFEMODEL_PATH = add_root("models/VGG_ILSVRC_19_layers.caffemodel")
DEF_PROTOTXT_PATH = add_root("models/VGG_2014_19.prototxt")
DEF_WEIGHTS_PATH = add_root("models/VGG_2014_19_weights.npz")
DEF_BIASES_PATH = add_root("models/VGG_2014_19_biases.npz")
DEF_CATNAMES_PATH = add_root("models/synset.txt")

DEF_CAFFEMODEL_DL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel"
DEF_PROTOTXT_DL = "https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt25721/VGG_ILSVRC_19_layers_deploy.prototxt"

DEF_FRCNN_DL = "https://people.eecs.berkeley.edu/~rbg/faster-rcnn-data/faster_rcnn_models.tgz"
DEF_FRCNN_PATH = add_root("models/faster_rcnn_models.tgz")
DEF_FRCNN_CAFFEMODEL_PATH = add_root( 
        "models/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel")
DEF_FRCNN_PROTOTXT_PATH = add_root("models/faster_rcnn_models/VGG16_faster_rcnn_final.prototxt")

DEF_FRCNN_WEIGHTS_PATH = add_root("models/faster_rcnn_models/VGG16_faster_rcnn_weights.npz")
DEF_FRCNN_BIASES_PATH = add_root("models/faster_rcnn_models/VGG16_faster_rcnn_biases.npz")

DEF_TEST_IMAGE_PATHS = [add_root( "test/images/Cat.jpg"),
                        add_root( "test/images/EnglishSetter.jpg"),
                        add_root( "test/images/KitFox.jpg")]

# If you wish to use a CPU for extraction of caffe weights to avoid GPU memory overuse or
# hassle of caffe GPU configuration, set this to True
CAFFE_USE_CPU = True


# Proposal layer constants
DEF_FEATURE_STRIDE = 16
DEF_IOU_THRESHOLD = .7
DEF_PRE_NMS_KEEP = 12000
DEF_POST_NMS_KEEP = 2000
DEF_MIN_PROPOSAL_DIMS = 16 #In image pixels
