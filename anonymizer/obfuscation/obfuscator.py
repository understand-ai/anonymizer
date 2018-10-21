import math

import numpy as np
import scipy.stats as st
import tensorflow as tf

from anonymizer.obfuscation.helpers import kernel_initializer, bilinear_filter, get_default_session_config


class Obfuscator:
    """ This class is used to blur box regions within an image with gaussian blurring. """
    def __init__(self, kernel_size=21, sigma=2, channels=3, box_kernel_size=9, smooth_boxes=True):
        """
        :param kernel_size: Size of the blurring kernel.
        :param sigma: standard deviation of the blurring kernel. Higher values lead to sharper edges, less blurring.
        :param channels: Number of image channels this blurrer will be used for. This is fixed as blurring kernels will
            be created for each channel only once.
        :param box_kernel_size: This parameter is only used when smooth_boxes is True. In this case, a smoothing
            operation is applied on the bounding box mask to create smooth transitions from blurred to normal image at
            the bounding box borders.
        :param smooth_boxes: Flag defining if bounding box masks borders should be smoothed.
        """
        # Kernel must be uneven because of a simplified padding scheme
        assert kernel_size % 2 == 1

        self.kernel_size = kernel_size
        self.box_kernel_size = box_kernel_size
        self.sigma = sigma
        self.channels = channels
        self.smooth_boxes = smooth_boxes

        # create internal kernels (3D kernels with the channels in the last dimension)
        kernel = self._gaussian_kernel(kernel_size=self.kernel_size, sigma=self.sigma)  # kernel for blurring
        self.kernels = np.repeat(kernel, repeats=channels, axis=-1).reshape((kernel_size, kernel_size, channels))
        mean_kernel = bilinear_filter(filter_size=(box_kernel_size, box_kernel_size))  # kernel for smoothing
        self.mean_kernel = np.expand_dims(mean_kernel/np.sum(mean_kernel), axis=-1)

        # visualization
        # print(self.kernels.shape)
        # self._visualize_kernel(kernel=self.kernels[..., 0])
        # self._visualize_kernel(kernel=self.mean_kernel[..., 0])

        # wrap everything in a tf session which is always open
        sess = tf.Session(config=get_default_session_config(0.9))
        self._build_graph()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        self.sess = sess

    def _gaussian_kernel(self, kernel_size=30, sigma=5):
        """ Returns a 2D Gaussian kernel array.

        :param kernel_size: Size of the kernel, the resulting array will be kernel_size x kernel_size
        :param sigma: Standard deviation of the gaussian kernel.
        :return: 2D numpy array containing a gaussian kernel.
        """

        interval = (2 * sigma + 1.) / kernel_size
        x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()

        return kernel

    def _build_graph(self):
        """ Builds the tensorflow graph containing all necessary operations for the blurring procedure. """
        with tf.variable_scope('gaussian_blurring'):
            image = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channels], name='x_input')
            mask = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='x_input')

            # ---- mean smoothing
            if self.smooth_boxes:
                W_mean = tf.get_variable(name='mean_kernel',
                                         shape=[self.mean_kernel.shape[0], self.mean_kernel.shape[1], 1, 1],
                                         dtype=tf.float32,
                                         initializer=kernel_initializer(kernels=self.mean_kernel),
                                         trainable=False, validate_shape=True)

                smoothed_mask = tf.nn.conv2d(input=mask, filter=W_mean, strides=[1, 1, 1, 1], padding='SAME',
                                             use_cudnn_on_gpu=True, data_format='NHWC', name='smooth_mask')
            else:
                smoothed_mask = mask

            # ---- blurring the initial image
            W_blur = tf.get_variable(name='gaussian_kernels',
                                     shape=[self.kernels.shape[0], self.kernels.shape[1], self.kernels.shape[2], 1],
                                     dtype=tf.float32,
                                     initializer=kernel_initializer(kernels=self.kernels),
                                     trainable=False, validate_shape=True)

            # Use reflection padding in conjunction with convolutions without padding (no border effects)
            pad = (self.kernel_size - 1) / 2
            paddings = np.array([[0, 0], [pad, pad], [pad, pad], [0, 0]])
            img = tf.pad(image, paddings=paddings, mode='REFLECT')
            blurred_image = tf.nn.depthwise_conv2d_native(input=img, filter=W_blur, strides=[1, 1, 1, 1],
                                                          padding='VALID', data_format='NHWC', name='conv_spatial')

            # Combination of the blurred image and the original image with a bounding box mask
            anonymized_image = image * (1-smoothed_mask) + blurred_image * smoothed_mask

            # store internal variables
            self.image = image
            self.mask = mask
            self.anonymized_image = anonymized_image

    def _get_all_masks(self, bboxes, images):
        """ For a batch of boxes, returns heatmap encoded box images.

        :param bboxes: 3D np array containing a batch of box coordinates (see anonymize for more details).
        :param images: 4D np array with NHWC encoding containing a batch of images.
        :return: 4D np array in NHWC encoding. For each batch sample, there is a binary mask with one channel which
            encodes bounding box locations.
        """
        masks = np.zeros(shape=(images.shape[0], images.shape[1], images.shape[2], 1))
        image_size = (images.shape[1], images.shape[2])

        for n, boxes in enumerate(bboxes):
            masks[n, ...] = self._get_box_mask(box_array=boxes, image_size=image_size)

        return masks

    def _get_box_mask(self, box_array, image_size):
        """ For an array of boxes for a single image, return a binary mask which encodes box locations as heatmap.

        :param box_array: 2D numpy array with dimnesions: numer_bboxes x 4.
            Boxes are encoded as [x_min, y_min, x_max, y_max]
        :param image_size: tuple containing the image dimensions. This is used to create the binary mask layout.
        :return: 3D numpy array containing the binary mask (last dimension is always size 1).
        """
        # assert isinstance(box_array, np.ndarray) and len(box_array.shape) == 2
        mask = np.zeros(shape=(image_size[0], image_size[1], 1))

        # insert box masks into array
        for box in box_array:
            mask[box[1]:box[3], box[0]:box[2], :] = 1

        return mask

    def _obfuscate_numpy(self, images, bboxes):
        """ Anonymizes bounding box regions within a given region by applying gaussian blurring.

        :param images: 4D np array with NHWC encoding containing a batch of images.
            The number of channels must match self.num_channels.
        :param bboxes: 3D np array containing a batch of box coordinates. First dimension is the batch dimension.
            Second dimension are boxes within an image and third dimension are the box coordinates.
            np.array([[[10, 15, 30, 50], [500, 200, 850, 300]]]) contains one batch sample and two boxes for that
            sample. Box coordinates are in [x_min, y_min, x_max, y_max] notation.
        :return: 4D np array with NHWC encoding containing an anonymized batch of images.
        """
        # assert isinstance(images, np.ndarray) and len(images.shape) == 4
        # assert isinstance(bboxes, np.ndarray) and len(bboxes.shape) == 3 and bboxes.shape[-1] == 4
        bbox_masks = self._get_all_masks(bboxes=bboxes, images=images)

        anonymized_image = self.sess.run(fetches=self.anonymized_image,
                                         feed_dict={self.image: images, self.mask: bbox_masks})
        return anonymized_image

    def obfuscate(self, image, boxes):
        """
        Anonymize all bounding boxes in a given image.
        :param image: The image as np.ndarray with shape==(height, width, channels).
        :param boxes: A list of boxes.
        :return: The anonymized image.
        """
        if len(boxes) == 0:
            return np.copy(image)

        image_array = np.expand_dims(image, axis=0)
        box_array = []
        for box in boxes:
            x_min = int(math.floor(box.x_min))
            y_min = int(math.floor(box.y_min))
            x_max = int(math.ceil(box.x_max))
            y_max = int(math.ceil(box.y_max))
            box_array.append(np.array([x_min, y_min, x_max, y_max]))
        box_array = np.stack(box_array, axis=0)
        box_array = np.expand_dims(box_array, axis=0)

        anonymized_images = self._obfuscate_numpy(image_array, box_array)
        return anonymized_images[0]
