import numpy
import caffe
import tensorflow
import settings as s

from urllib import urlretrieve
import threading
import os
import os.path
import skimage.io
import skimage.transform
import tarfile

class Singleton(type):
    """ Simple Singleton for use as metaclass """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
        

def loadImage(imgPath):
    """ 
    Loads an image from imgPath, crops the center square, and subtracts the RGB
    values from VGG_MEAN
    """
    img = skimage.io.imread(imgPath)
    # We now need to crop the center square from our image
    shortSide = min(img.shape[0:2])
    x_1 = (img.shape[0] - shortSide) / 2
    x_2 = (img.shape[0] + shortSide) / 2
    y_1 = (img.shape[1] - shortSide) / 2
    y_2 = (img.shape[1] + shortSide) / 2
    centerSquare = img[x_1:x_2, y_1:y_2]
    
    rescaledImg = skimage.transform.resize(centerSquare, (224,224))
    
    # Subtract VGG_MEAN from every pixel.  VGG_MEAN is in BGR, but image is
    # RGB, hence the reverse order.
    rescaledImg[:,:,0] -= s.VGG_MEAN[2]
    rescaledImg[:,:,1] -= s.VGG_MEAN[1]
    rescaledImg[:,:,2] -= s.VGG_MEAN[0]
    return rescaledImg
	
def downloadModel():
    """
    Download the vgg19 model (.caffemodel and .prototxt) files and save them to the
    DEF_CAFFEMODEL_PATH and DEF_PROTOTXT_PATH directories
    """
    # prototxt for the vgg19 model
    def progressBar(blockCount, blockSize, fileSize):
        # Progress bar for download, passed to urlretrieve
        return

    urlretrieve(s.DEF_CAFFEMODEL_DL, s.DEF_CAFFEMODEL_PATH, progressBar)
    urlretrieve(s.DEF_PROTOTXT_DL, s.DEF_PROTOTXT_PATH, progressBar)
    return

def downloadFasterRcnn():
    """
    Download the .caffemodel for the faster r-cnn model and save them to the
    DEF_FRCNN_CAFFEMODEL_PATH path
    """
    filesExist = os.path.isfile(s.DEF_FRCNN_WEIGHTS_PATH)
    filesExist = filesExist and os.path.isFile(s.DEF_FRCNN_BIASES_PATH)

    if filesExist:
        print ("Faster RCNN files already exist.  Delete them if you wish to re-download")
    urlretrieve(s.DEF_FRCNN_DL, s.DEF_FRCNN_PATH)
    tar = tarfile.open(s.DEF_FRCNN_DL)
    tar.extractall()
    tar.close()

    # Check to make sure extraction worked
    if not os.path.isfile(s.DEF_FRCNN_CAFFEMODEL_PATH):
        print ("Extraction failed.  Check if s.DEF_FRCNN_CAFFEMODE_PATH is correct.")
    return

def fasterCoffeeMachine(protoxtPath=s.DEF_FRCNN_PROTOTXT_PATH, caffemodelPath=s.DEF_FRCNN_CAFFEMODEL_PATH):
    """ Extract the weights and biases from a .caffemodel file and save in a npz file"""
    
    # Extract Caffe weights and biases
    coffee = caffe.Net(prototxtPath, caffemodelPath, caffe.TEST)
    caffeVggWeights = { name: blobs[0].data for name, blobs in coffee.params.iteritems() }
    caffeVggBiases = { name: blobs[1].data for name, blobs in coffee.params.iteritems() }
    
    # There are a few conv layers without "conv" in name, but treated the same
    devious_convs = ["rpn_cls_score", "rpn_bbox_pred"]

    for name in caffeVggWeights:
        # As before, the conv layers are all identically processed.
        if ("conv" in name) or (name in devious_convs) :
            # Tensorflow order  : [ width, height, in_channels, out_channels ]
            # Caffe order       : [ out_channels, in_channels, width, height ]
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2,3,1,0))
            if "conv1_1" in name:
                # We have to additionally convert BGR to RGB for the first layer
                caffeVggWeights['conv1_1'] = numpy.copy(caffeVggWeights['conv1_1'][:,:,[2,1,0],:])
        elif "fc6" in name:
            # This depends on pooling_w, pooling_h and channels in output of the base convo. network
            # Very hackish, should really be reading util.settings constants or something
            INPUT_SIZE = 512*7*7

            # This part insn't hackish, output is always 4096
            OUTPUT_SIZE = 4096
            # Reshape the weights to their unsquashed form
            caffeVggWeights[name] = caffeVggWeights[name].reshape((-1, 512, 7, 7))
            
            # Transpose the weights so that they are in the tensorflow instead of caffe order
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2,3,1,0))
            caffeVggWeights[name] = caffeVggWeights[name].reshape(INPUT_SIZE, OUTPUT_SIZE)
        elif "fc" in name:
            # Since elif, not "fc6"			
            # Tensorflow order	: [in_channels, out_channels]
            # Caffe order		: [out_channels, in_channels]
            caffeVggWeights[name] = caffeVggWeights[name].transpose((1,0))
         


def coffeeMachine(prototxtPath=s.DEF_PROTOTXT_PATH, caffemodelPath=s.DEF_CAFFEMODEL_PATH):
    """
    Extract the weights and biases from the .caffemodel and save it in npz files named
    DEF_WEIGHTS_PATH and DEF_BIASES_PATH
    """	
    # Extract Caffe weights and biases
    coffee = caffe.Net(prototxtPath, caffemodelPath, caffe.TEST)
    caffeVggWeights = { name: blobs[0].data for name, blobs in coffee.params.iteritems() }
    caffeVggBiases = { name: blobs[1].data for name, blobs in coffee.params.iteritems() }

    for name in caffeVggWeights:
        if "conv" in name:
            # Tensorflow order	: [width, height, in_channels, out_channels]
            # Caffe order		: [out_channels, in_channels, width, height]
            # Hence, to translate from Caffe to Tensorflow
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2,3,1,0))
            if "conv1_1" in name:
                # Converting BGR to RGB
                caffeVggWeights['conv1_1'] = numpy.copy(caffeVggWeights['conv1_1'][:,:,[2,1,0],:])

        elif "fc6" in name:
            INPUT_SIZE = 25088
            OUTPUT_SIZE = 4096

            # Reshape the weights to their unsquashed form
            caffeVggWeights[name] = caffeVggWeights[name].reshape((OUTPUT_SIZE, 512, 7, 7))
            
            # Transpose the weights so that it is in the tensorflow instead of caffe order
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2,3,1,0))
            caffeVggWeights[name] = caffeVggWeights[name].reshape(INPUT_SIZE, OUTPUT_SIZE)

        elif "fc" in name:
            # Since elif, not "fc6"			
            # Tensorflow order	: [in_channels, out_channels]
            # Caffe order		: [out_channels, in_channels]
            caffeVggWeights[name] = caffeVggWeights[name].transpose((1,0))
            
        else:
            # Error in loading model, raise exception
            raise StandardError("Warning, model being saved as .npz file has non-standard field names")
            
    numpy.savez(s.DEF_WEIGHTS_PATH, **caffeVggWeights)
    numpy.savez(s.DEF_BIASES_PATH, **caffeVggBiases)
    print "Coffee successfully brewed"

def isolatedFunctionRun(func, textSuppress, *args, **kwargs):
    """
    Runs the function func, with arguments *args and **kwargs, in its own thread.
    If textSupress = True, all console output will be redirected to os.devnull
    """
    # Open two os.devnull

    nulls = [os.open(os.devnull, os.O_RDWR) , os.open(os.devnull, os.O_RDWR)]
    if textSuppress:
        old = os.dup(1), os.dup(2)
        # Set stderr and stdout to null
        os.dup2(nulls[0], 1)
        os.dup2(nulls[1], 2)

    t = threading.Thread(target=func, args=args, kwargs=kwargs)
    t.start()
    t.join()

    if textSuppress:
        # Restore stderr and stdout to previous state
        os.dup2(old[0],1)
        os.dup2(old[1],2)
        # Close the os.devnulls	
        os.close(nulls[0])
        os.close(nulls[1])	
    print "wow!"
    return
