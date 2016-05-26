#import vgg19
import settings as s
import utils

print "Downloading model....",
utils.downloadModel()
print "  Model downloaded"
utils.coffeeMachine()


import tensorflow as tf
img = tf.placeholder("float", [None, 224, 224, 3], name="images")
model = vgg19.Vgg19()
print "Building model in TensorFlow....",
model.buildGraph(img, train=False)
print "  Model built"
