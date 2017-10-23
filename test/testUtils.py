import tensorflow as tf
import numpy as np


def eval_cpu(operation, s=None):
    ''' Evaluate operation on default graph and with cpu

        If s is not None, then we use s as a tf.test.TestCase from which we then
        start a s.test_session() for performing operations
    '''
    if s is None:
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                return sess.run(operation)
    else:
        with s.test_session() as sess:
            with tf.device("/cpu:0"):
                return sess.run(operation)


def eval_gpu(operation, s=None):
    ''' Evaluate operation on default graph and with gpu

        If s is not None, then we use s as a tf.test.TestCase from which we then
        start a s.test_session() for performing operations
    '''
    if s is None:
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                return sess.run(operation)
    else:
        with s.test_session() as sess:
            with tf.device("/gpu:0"):
                return sess.run(operation)


def assertArrayAlmostEqual(self, A, B, places=6, msg=""):
    B = np.array(B)
    self.assertEqual(A.shape, B.shape,
            msg="There is a shape mismatch.  Expected shape %s but got shape %s"
                % (str(A.shape), str(B.shape))
    )
    # Under assumption of equal shape, the following runs just fine
    it = np.nditer(A, flags=['multi_index'])
    while not it.finished:
        ind = it.multi_index
        self.assertAlmostEqual(
            A[ind], B[ind], places=places,
            msg="At location %s, there is a mismatch.  We expected %d but got %d.\n "
                % (ind, B[ind], A[ind])
                + "Dumping entire comparison array: \n %s"
                % (str(A) + msg)
        )
        it.iternext()


def array_equality_assert(self, nparray, ref_nparray, tolerance=.01):
    """
    Looks up activation in list of activations to see whether it is as expected.

    Assumes that self is a subclass of unittest.TestCase

    Parameters:
    self        -- subclass of unittest.TestCase on which asserts will be run.
    nparray     -- numpy array that is undergoing testing
    ref_nparray -- reference numpy array that nparray is being compared to
    tolerance   -- maximum amount of difference between the entries of nparray
                        allowed before failing the test.
    """
    # Check for shape mismatches
    self.assertTrue(nparray.shape == ref_nparray.shape,
        msg="Unequal shapes.  Ref. shape is {}, array shape is {}".format(
            ref_nparray.shape, nparray.shape))
    greatest_diff = np.amax(np.absolute(ref_nparray - nparray))
    self.assertLessEqual(greatest_diff, tolerance,
        msg="Greatest difference was %f" % greatest_diff)
    return greatest_diff <= toleranc
