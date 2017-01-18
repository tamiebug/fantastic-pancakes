import tensorflow as tf

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
