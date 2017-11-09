import numpy
import tensorflow as tf
from . import settings as s
import skimage.io
import skimage.transform
import tarfile

from urllib.request import urlretrieve
import threading
import os
import sys
from functools import partial
from contextlib import contextmanager
import json
import subprocess
import zipfile


def eprint(text):
    """ Prints text to stderr """
    print(text, file=sys.stderr)


def genProgressBar(**kwargs):
    """
    Generates a _progressBar callback with given named arguments

    Possible named argument options:
        name:
            Name of file to be downloaded (default is "file")
        symbol:
            Symbol used in download bar (default is '#')
        barSize:
            Size of download bar (default is 50)

    Returns:
        progressBar callback to be used with the likes of urlrequest
    """
    def _progressBar(blockCount, blockSize, fileSize, name="file", symbol='#', barSize=50):
        """ Simple, customizable progress bar for downloads """
        totalBlocks = -(-fileSize // blockSize)  # Integer division rounded up
        ratioComplete = blockCount / totalBlocks
        print("Downloading {} : [{}{}]%".format(name,
            symbol * int(ratioComplete * barSize),
            ' ' * (barSize - int(ratioComplete * barSize)),
            int(ratioComplete * 100)), end="\r")
        return

    return partial(_progressBar, **kwargs)


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
    values from VGG_MEAN.  Not to be used for Faster-RCNN, but rather for VGG19/16
    by itself.
    """
    img = skimage.io.imread(imgPath)
    # We now need to crop the center square from our image
    shortSide = min(img.shape[0:2])
    x_1 = (img.shape[0] - shortSide) / 2
    x_2 = (img.shape[0] + shortSide) / 2
    y_1 = (img.shape[1] - shortSide) / 2
    y_2 = (img.shape[1] + shortSide) / 2
    centerSquare = img[x_1:x_2, y_1:y_2]

    rescaledImg = skimage.transform.resize(centerSquare, (224, 224))

    # Subtract VGG_MEAN from every pixel.  VGG_MEAN is in BGR, but image is
    # RGB, hence the reverse order.
    rescaledImg[:, :, 0] -= s.VGG_MEAN[2]
    rescaledImg[:, :, 1] -= s.VGG_MEAN[1]
    rescaledImg[:, :, 2] -= s.VGG_MEAN[0]
    return rescaledImg


def grabFasterRCNNParams(weights_path=s.DEF_FRCNN_WEIGHTS_NPZ_DL,
        biases_path=s.DEF_FRCNN_BIASES_NPZ_DL, download_anyway=False):
    """ Downloads and saves .npz files for Faster RCNN weights and biases """

    weights_destination = s.DEF_FRCNN_WEIGHTS_PATH
    biases_destination = s.DEF_FRCNN_BIASES_PATH

    from os.path import isfile

    if not isfile(weights_destination) or download_anyway:
        urlretrieve(weights_path, weights_destination,
            genProgressBar(name="Faster RCNN weights"))

    if not isfile(biases_destination) or download_anyway:
        urlretrieve(biases_path, biases_destination,
            genProgressBar(name="Faster RCNN biases"))
    return


def downloadModel():
    """
    Download the vgg19 model (.caffemodel and .prototxt) files and save them to the
    DEF_CAFFEMODEL_PATH and DEF_PROTOTXT_PATH directories
    """
    urlretrieve(s.DEF_CAFFEMODEL_DL, s.DEF_CAFFEMODEL_PATH,
        genProgressBar(name="VGG19 caffemodel"))
    urlretrieve(s.DEF_PROTOTXT_DL, s.DEF_PROTOTXT_PATH,
        genProgressBar(name="VGG19 prototxt"))
    return


def downloadFasterRcnn():
    """
    Download the .caffemodel for the faster r-cnn model and save them to the
    DEF_FRCNN_CAFFEMODEL_PATH path
    """
    filesExist = os.path.isfile(s.DEF_FRCNN_WEIGHTS_PATH)
    filesExist = filesExist and os.path.isfile(s.DEF_FRCNN_BIASES_PATH)

    if filesExist:
        print("Faster RCNN files already exist.  Delete them if you wish to re-download")
        return
    urlretrieve(s.DEF_FRCNN_DL, s.DEF_FRCNN_PATH)
    tar = tarfile.open(s.DEF_FRCNN_PATH)
    tar.extractall()
    tar.close()

    # Check to make sure extraction worked
    if not os.path.isfile(s.DEF_FRCNN_CAFFEMODEL_PATH):
        print("Extraction failed.  Check if s.DEF_FRCNN_CAFFEMODE_PATH is correct.")

    print("Extracting weights from caffe into numpy")
    fasterCoffeeMachine(prototxtPath=s.DEF_FRCNN_PROTOTXT_PATH,
                        caffemodelPath=s.DEF_FRCNN_CAFFEMODEL_PATH)
    return


def fasterCoffeeMachine(prototxtPath=s.DEF_FRCNN_PROTOTXT_PATH,
        caffemodelPath=s.DEF_FRCNN_CAFFEMODEL_PATH):
    """ Extract the weights and biases from a .caffemodel file and save in a npz file"""
    import caffe

    # Extract Caffe weights and biases
    coffee = caffe.Net(prototxtPath, caffemodelPath, caffe.TEST)
    caffeVggWeights = {name: blobs[0].data for name, blobs in coffee.params.items()}
    caffeVggBiases = {name: blobs[1].data for name, blobs in coffee.params.items()}

    # There are a few conv layers without "conv" in name, but treated the same
    devious_convs = ["rpn_cls_score", "rpn_bbox_pred"]
    for name in caffeVggWeights:
        print(("{} : {}".format(name, caffeVggWeights[name].shape)))
        # As before, the conv layers are all identically processed.
        if ("conv" in name) or (name in devious_convs):
            # Tensorflow order  : [ width, height, in_channels, out_channels ]
            # Caffe order       : [ out_channels, in_channels, width, height ]
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2, 3, 1, 0))
            if "conv1_1" in name:
                # We have to additionally convert BGR to RGB for the first layer
                caffeVggWeights['conv1_1'] = numpy.copy(
                    caffeVggWeights['conv1_1'][:, :, [2, 1, 0], :])
        elif "fc6" in name:
            # This depends on pooling_w, pooling_h and channels in output of the base conv net
            # Very hackish, should really be reading util.settings constants or something
            INPUT_SIZE = 512 * 7 * 7

            # This part insn't hackish, output is always 4096
            OUTPUT_SIZE = 4096
            # Reshape the weights to their unsquashed form
            caffeVggWeights[name] = caffeVggWeights[name].reshape((-1, 512, 7, 7))

            # Transpose the weights so that they are in the tensorflow instead of caffe order
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2, 3, 1, 0))
            caffeVggWeights[name] = caffeVggWeights[name].reshape(INPUT_SIZE, OUTPUT_SIZE)
        elif "fc" in name or name == "bbox_pred" or name == "cls_score":
            # Since elif, not "fc6"
            # Tensorflow order	: [in_channels, out_channels]
            # Caffe order		: [out_channels, in_channels]
            caffeVggWeights[name] = caffeVggWeights[name].transpose((1, 0))
        else:
            print(("Warning, following layer not being saved: {}".format(name)))
    numpy.savez(s.DEF_FRCNN_WEIGHTS_PATH, **caffeVggWeights)
    numpy.savez(s.DEF_FRCNN_BIASES_PATH, **caffeVggBiases)
    print("Coffee successfully brewed")


def coffeeMachine(prototxtPath=s.DEF_PROTOTXT_PATH, caffemodelPath=s.DEF_CAFFEMODEL_PATH):
    """
    Extract the weights and biases from the .caffemodel and save it in npz files named
    DEF_WEIGHTS_PATH and DEF_BIASES_PATH
    """
    import caffe

    # Extract Caffe weights and biases
    coffee = caffe.Net(prototxtPath, caffemodelPath, caffe.TEST)
    caffeVggWeights = {name: blobs[0].data for name, blobs in coffee.params.items()}
    caffeVggBiases = {name: blobs[1].data for name, blobs in coffee.params.items()}

    for name in caffeVggWeights:
        if "conv" in name:
            # Tensorflow order	: [width, height, in_channels, out_channels]
            # Caffe order		: [out_channels, in_channels, width, height]
            # Hence, to translate from Caffe to Tensorflow
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2, 3, 1, 0))
            if "conv1_1" in name:
                # Converting BGR to RGB
                caffeVggWeights['conv1_1'] = numpy.copy(
                    caffeVggWeights['conv1_1'][:, :, [2, 1, 0], :])

        elif "fc6" in name:
            INPUT_SIZE = 25088
            OUTPUT_SIZE = 4096

            # Reshape the weights to their unsquashed form
            caffeVggWeights[name] = caffeVggWeights[name].reshape((OUTPUT_SIZE, 512, 7, 7))

            # Transpose the weights so that it is in the tensorflow instead of caffe order
            caffeVggWeights[name] = caffeVggWeights[name].transpose((2, 3, 1, 0))
            caffeVggWeights[name] = caffeVggWeights[name].reshape(INPUT_SIZE, OUTPUT_SIZE)

        elif "fc" in name:
            # Since elif, not "fc6"
            # Tensorflow order	: [in_channels, out_channels]
            # Caffe order		: [out_channels, in_channels]
            caffeVggWeights[name] = caffeVggWeights[name].transpose((1, 0))

        else:
            # Error in loading model, raise exception
            raise Exception("Warning, model being saved as .npz file has non-standard field names")

    numpy.savez(s.DEF_WEIGHTS_PATH, **caffeVggWeights)
    numpy.savez(s.DEF_BIASES_PATH, **caffeVggBiases)
    print("Coffee successfully brewed")


def isolatedFunctionRun(func, textSuppress, *args, **kwargs):
    """
    Runs the function func, with arguments *args and **kwargs, in its own thread.
    If textSupress = True, all console output will be redirected to os.devnull
    """
    # Open two os.devnull

    nulls = [os.open(os.devnull, os.O_RDWR), os.open(os.devnull, os.O_RDWR)]
    if textSuppress:
        old = os.dup(1), os.dup(2)
        # Set stderr and stdout to null
        os.dup2(nulls[0], 1)
        os.dup2(nulls[1], 2)

    # Wrapping the function so that we can capture the output value.  t.join() won't give it to us
    output = {}

    def wrapped_func(function, out, *args, **kwargs):
        out['return_val'] = function(*args, **kwargs)
        return

    # Need to put the func and output as first arguments to wrapped_func
    new_args = [func, output]
    new_args.extend(args)  # Now put the old arguments afterwards

    t = threading.Thread(target=wrapped_func, args=new_args, kwargs=kwargs)
    t.start()
    t.join()

    if textSuppress:
        # Restore stderr and stdout to previous state
        os.dup2(old[0], 1)
        os.dup2(old[1], 2)
        # Close the os.devnulls
        os.close(nulls[0])
        os.close(nulls[1])

    return output['return_val']


@contextmanager
def easy_scope(name, *args, **kwargs):
    """
    Opens a variable and name scope so that things work as you think they should.

    variable_scopes automatically reuse names for variables, but Tensorflow does not
    normally reuse the name_scope that comes with, instead making a new one every time
    variable_scope is called, adding underscores if needed to make them unique.  This
    forces tensorflow's hand, making it reuse the name as well as variable scopes if
    the reuse keyword is not set to False.
    """

    # If we don't want to reuse, then don't massage the scopes.  Normal behavior suffices
    if "reuse" in kwargs:
        if kwargs["reuse"] is False:
            with tf.variable_scope(name, *args, **kwargs) as scope:
                yield scope

    # Not part of the public API, may be broken by an update
    try:
        curr_scope = tf.get_default_graph()._name_stack
    except AttributeError:
        raise AttributeError("tf.get_default_graph() no longer has attribute _name_stack. \
                This function has been broken by a tensorflow build after .12.  Either \
                fix this function, or revert to an older version of tensorflow")
    new_name = name
    if not name.endswith("/"):
        new_name = new_name + "/"

    if not curr_scope == "":
        new_name = curr_scope + "/" + new_name

    with tf.variable_scope(name, *args, **kwargs) as scope:
        with tf.name_scope(new_name):
            yield scope


def run_with_error_msg(command, errormsg):
    """ Runs the command, printing errormsg if it fails """
    try:
        subprocess.run("source ~/.bashrc;" + command, check=True,
                shell=True, executable="/bin/bash")
    except subprocess.CalledProcessError:
        eprint(errormsg)
        print("FALSE")
        return False
    print("TRUE")
    return True


def make_dir(directory):
    """ Makes directory if it does not already exist.  Returns true if directory made """
    if not os.path.isdir(directory):
        subprocess.run("mkdir " + directory, shell=True)
        return True
    return False


def prepare_COCO2017(restrict_to_COCO2007=True):
    """ Download the COCO2017 data set and return a dict of bboxes, None on failure """

    shell_load_train = "gsutil -m rsync gs://images.cocodataset.org/train2017 data/coco2017_train"
    shell_load_val = "gsutil -m rsync gs://images.cocodataset.org/val2017 data/coco2017_valid"
    shell_load_trainval_annotations = ("gsutil -m rsync gs://images.cocodataset.org/annotations"
                                       " data/")
    """
    if shutil.which("gsutil") is None:
        print("gsutil not installed, but necessary for downloading the COCO2017 data set.")
        inp = input("Install gsutil? y/n")
        if inp != 'yes' and inp != 'y':
            print("gsutil not installed.  prepare_COCO2017 aborted")

    run_with_error_msg("curl https://sdk.cloud.google.com | bash",
                       "gsutil download unsuccessful.  Manually install gsutil")

    """
    # Necessary so that gsutil becomes available for use

    # Move up to project's home directory.
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    make_dir("data/")

    if make_dir("data/coco2017_valid") or (
            os.path.exists("data/coco2017_valid") and os.listdir("data/coco2017_valid") == []):
        print("Downloading COCO2017 validation image set")
        if not run_with_error_msg(shell_load_val, "COCO2017 validation image set download failed"):
            return None

    if make_dir("data/coco2017_train") or (
            os.path.exists("data/coco2017_train") and os.listdir("data/coco2017_train") == []):
        print("Downloading COCO2017 training image set")
        if not run_with_error_msg(shell_load_train, "COCO2017 training image set download failed"):
            return None

    if make_dir("data/coco2017_trainval_annotations") or (
            os.path.exists("data/coco2017_trainval_annotations") and
            os.listdir("data/coco2017_trainval_annotations") == []):
        if not run_with_error_msg(shell_load_trainval_annotations,
                "COCO2017 trainval annotations download failed"):
            return None
        else:
            with zipfile.ZipFile("data/annotations_trainval2017.zip", "r") as zip_file:
                zip_file.extractall("data/coco2017_trainval_annotations/")

    annotations_json = None

    print("Loading annotations JSON file...")
    with open("data/coco2017_trainval_annotations/annotations/instances_train2017.json") as data:
        annotations_json = json.load(data)

    return process_COCO2017_JSON(restrict_to_COCO2007, annotations_json)


def process_COCO2017_JSON(restrict_to_COCO2007, annotations_json):
    """
    Process JSON file and produce dictionary with bounding boxes

    Returns dictionary with image numbers as keys, and values being lists of
    ground truth boxes in the following format: (x, y, w, h, class),
    where class is any one of the 90 classes present in COCO2017.
    If restrict_to_COCO2007 is True, then we restrict ourself only to classes
    belonging to the 20 (non-background) classes in COCO2007.
    """

    bounding_boxes = {}
    print("Processing bbox data from annotations file...")

    if restrict_to_COCO2007:
        # The zero is so that the allowed_classes get offset by one
        allowed_classes = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21,
                           67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        for annotation in annotations_json['annotations']:
            image_id = annotation['image_id']
            if annotation['category_id'] in allowed_classes:
                if image_id not in bounding_boxes:
                    # Start a list of bounding boxes for that image_id
                    bounding_boxes[image_id] = [annotation['bbox'][:]]
                else:
                    # Add to the already existing list of bounding boxes for that id
                    bounding_boxes[image_id].append(annotation['bbox'][:])

                # In either case, we need to append the classification category to the bbox
                # 4-tuple in order to get the desired format.
                # We want the classes to be numbered as they appear in allowed_classes to
                # ensure compatibility with COCO2007's class ordering.
                bounding_boxes[image_id][-1].append(
                    allowed_classes.index(annotation['category_id'])
                )

    else:
        # Not being restricted to COCO2007
        for annotation in annotations_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in bounding_boxes:
                bounding_boxes[image_id] = [annotation['bbox'][:]]
            else:
                bounding_boxes[image_id].append(annotation['bbox'][:])

            bounding_boxes[image_id][-1].append(annotation['category_id'][:])

    print("Bbox data processing complete!")
    return bounding_boxes
