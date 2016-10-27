from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
import tensorflow as tf

from . import activations

from .utils import linear_weights
from .utils import linear_add_bias


class GlimpseNetwork(object):
    """Glimpse network.

    Take glimpse location input and output features for RNN.

    """

    def __init__(self, args):
        self.data_format = args.data_format if 'data_format' in args else 'NCHW'

        self.num_channels, self.original_height, self.original_width = args.observation_space

        self.depth = args.glimpse_depth
        self.patch_height, self.patch_width = args.glimpse_patch_size
        self.sensor_bandwidth = self.patch_height * self.patch_width * 1 * self.depth   # self.num_channels
        self.scale = args.glimpse_scale

        height = self.patch_height
        width = self.patch_width
        self.win_sizes = [np.array([height, width])]

        for _ in range(1, self.depth):
            height *= self.scale
            width *= self.scale
            self.win_sizes.append(np.array([height, width]))

        self.loc_dim = args.loc_dim
        self.location_hidden_size = args.location_hidden_size
        self.glimpse_hidden_size = args.glimpse_hidden_size
        self.glimpse_size = args.glimpse_size

        self.activation = activations.get(args.activation)

        # self.glimpse_images = []
        self.w = {}

        self.init_weights()

    def init_weights(self):
        """ Initialize all the trainable weights."""
        self.w['gs_l0_w'], self.w['gs_l0_b'] = linear_weights((self.sensor_bandwidth, self.glimpse_hidden_size),
                                                              stddev=0.01,
                                                              scope='glimpse_sensor/l0')
        self.w['gs_l1_w'], self.w['gs_l1_b'] = linear_weights((self.glimpse_hidden_size, self.glimpse_size),
                                                              stddev=0.01,
                                                              scope='glimpse_sensor/l1')

        self.w['ls_l0_w'], self.w['ls_l0_b'] = linear_weights((self.loc_dim, self.location_hidden_size),
                                                              stddev=0.01,
                                                              scope='location_sensor/l0')
        self.w['ls_l1_w'], self.w['ls_l1_b'] = linear_weights((self.location_hidden_size, self.glimpse_size),
                                                              stddev=0.01,
                                                              scope='location_sensor/l1')

    def glimpse_sensor(self, loc, input_):
        """Take glimpse_l1 on the original images."""
        shape = [tf.shape(input_)[0], self.num_channels, self.original_height, self.original_width]
        if self.data_format == 'NHWC':
            shape = [tf.shape(input_)[0], self.original_height, self.original_width, self.num_channels]
        imgs = tf.reshape(input_, shape)

        if self.data_format == 'NCHW':
            # extract_glimpse expect Tensor of shape [batch_size, height, width, channels].
            imgs = tf.transpose(imgs, perm=[0, 2, 3, 1])

        output = []
        for d in range(self.depth):
            zooms = tf.image.extract_glimpse(imgs, self.win_sizes[d], loc)
            if self.data_format == 'NCHW':
                zooms = tf.transpose(zooms, perm=[0, 3, 1, 2])
            if d > 0:
                zooms = tf.image.resize_bilinear(zooms, (self.patch_height, self.patch_width))
            output.append(zooms)

        output = tf.pack(output)    # (depth, batch_size, ...)
        axes = [1, 0] + list(range(2, len(output.get_shape())))
        output = tf.transpose(output, axes)  # (batch_size, depth, ...)

        # self.glimpse_images.append(output)

        return output

    def __call__(self, loc, input_):
        glimpse_input = self.glimpse_sensor(loc, input_)
        glimpse_input = tf.reshape(glimpse_input, (tf.shape(loc)[0], self.sensor_bandwidth))

        g = linear_add_bias(glimpse_input, self.w['gs_l0_w'], self.w['gs_l0_b'],
                            activation_fn=self.activation,
                            scope='glimpse_sensor_l1')
        g = linear_add_bias(g, self.w['gs_l1_w'], self.w['gs_l1_b'],
                            scope='glimpse_sensor_l2')

        l = linear_add_bias(loc, self.w['ls_l0_w'], self.w['ls_l0_b'],
                            activation_fn=self.activation,
                            scope='location_sensor_l1')
        l = linear_add_bias(l, self.w['ls_l1_w'], self.w['ls_l1_b'],
                            scope='location_sensor_l2')

        return self.activation(g + l, name='glimpse')


class LocationNetwork(object):
    """Location network.

    Take output from other network and produce and sample the next location.

    """

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        self._sampling = sampling

    def __init__(self, input_dim, loc_dim, loc_stddev):
        self.loc_dim = loc_dim
        self.input_dim = input_dim
        self.loc_stddev = loc_stddev
        self._sampling = True

        self.w = {}

        self.init_weights()

    def init_weights(self):
        self.w['l_l0_w'], self.w['l_l0_b'] = linear_weights((self.input_dim, self.loc_dim),
                                                            stddev=0.01,
                                                            scope='l0')

    def __call__(self, input_):
        mean = tf.clip_by_value(linear_add_bias(input_, self.w['l_l0_w'], self.w['l_l0_b'],
                                                scope='mean_location'), -1., 1.)
        mean = tf.stop_gradient(mean, name='mean')
        if self._sampling:
            loc = mean + tf.random_normal((tf.shape(input_)[0], self.loc_dim), stddev=self.loc_stddev)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc, name='loc')
        return loc, mean
