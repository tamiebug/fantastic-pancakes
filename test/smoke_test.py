import util.utils
import util.frcnn_forward

util.frcnn_forward.demo("test/images/000456.png", threshold=0.05, gpu_fraction=.8)
