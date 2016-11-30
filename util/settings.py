VGG_MEAN = [103.939, 116.779, 123.68]
DEF_CAFFEMODEL_PATH = "models/VGG_ILSVRC_19_layers.caffemodel"
DEF_PROTOTXT_PATH = "models/VGG_2014_19.prototxt"
DEF_WEIGHTS_PATH = "models/VGG_2014_19_weights.npz"
DEF_BIASES_PATH = "models/VGG_2014_19_biases.npz"
DEF_CATNAMES_PATH = "models/synset.txt"

DEF_CAFFEMODEL_DL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel"
DEF_PROTOTXT_DL = "https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt25721/VGG_ILSVRC_19_layers_deploy.prototxt"

DEF_TEST_IMAGE_PATHS = ["test/images/Cat.jpg", "test/images/EnglishSetter.jpg", "test/images/KitFox.jpg"]

# If you wish to use a CPU for extraction of caffe weights to avoid GPU memory overuse or
# hassle of caffe GPU configuration, set this to True
CAFFE_USE_CPU = True


# Proposal layer constants

DEF_FEATURE_STRIDE = 16
DEF_IOU_THRESHOLD = .7
DEF_PRE_NMS_KEEP = 12000
DEF_POST_NMS_KEEP = 2000
DEF_MIN_PROPOSAL_DIMS = 16 #In image pixels
