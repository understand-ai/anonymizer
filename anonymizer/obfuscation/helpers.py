import numpy as np
import tensorflow as tf


def kernel_initializer(kernels):
    """ Wrapper for an initializer of convolution weights.

    :return: Callable initializer object.
    """
    assert len(kernels.shape) == 3
    kernels = kernels.astype(np.float32)

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        """Initializer function which is called from tensorflow internally.

        :param shape: Runtime / Construction time shape of the tensor.
        :param dtype: Data type of the resulting tensor.
        :param partition_info: Placeholder for internal tf call.
        :return: 4D numpy array with weights [filter_height, filter_width, in_channels, out_channels].
        """
        if shape:
            # second last dimension is input, last dimension is output
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0

        assert fan_out == 1 and fan_in == kernels.shape[-1]

        # define weight matrix (set dtype always to float32)
        # weights = np.expand_dims(kernels, axis=2)
        weights = np.expand_dims(kernels, axis=-1)

        return weights

    return _initializer


def bilinear_filter(filter_size=(4, 4)):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    Also allows asymmetric kernels.

    :param filter_size: Tuple defining the filter size in width and height.

    :return: 2D numpy array containing bilinear weights.
    """
    assert isinstance(filter_size, (list, tuple)) and len(filter_size) == 2

    factor = [(size + 1) // 2 for size in filter_size]
    # define first center dimension
    if filter_size[0] % 2 == 1:
        center_x = factor[0] - 1
    else:
        center_x = factor[0] - 0.5
    # define second center dimension
    if filter_size[1] % 2 == 1:
        center_y = factor[1] - 1
    else:
        center_y = factor[1] - 0.5

    og = np.ogrid[:filter_size[0], :filter_size[1]]
    kernel = (1 - abs(og[0] - center_x) / float(factor[0])) * (1 - abs(og[1] - center_y) / float(factor[1]))

    return kernel


def get_default_session_config(memory_fraction=0.9):
    """ Returns default session configuration

    :param memory_fraction: percentage of the memory which should be kept free (growing is allowed).
    :return: tensorflow session configuration object
    """
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
