import tensorflow as tf
import numpy as np

from .utils import linear_weights
from .utils import activation_summaries
from .utils import variable_summaries


class FullyConnected(object):
    def __init__(self, shape, stddev=0.02, bias_start=0.0, activation_fn=None, scope=''):
        self.w = {}

        self.scope = scope
        self.activation_fn = activation_fn

        self.w[self.scope + '_w'], self.w[self.scope + '_b'] = linear_weights(shape, stddev=stddev,
                                                                              bias_start=bias_start,
                                                                              scope=scope)

    def __call__(self, input_):
        out = tf.nn.bias_add(tf.matmul(input_, self.w[self.scope + '_w']), self.w[self.scope + '_b'])

        if self.activation_fn is not None:
            out = self.activation_fn(out)
        activation_summaries(out)

        return out


class Conv2d(object):
    def __init__(self, in_channels, output_dim, kernel_size, strides,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 activation_fn=tf.nn.relu, data_format='NHWC', padding='VALID', scope=''):
        self.w = {}

        self.scope = scope
        self.activation_fn = activation_fn
        self.data_format = data_format
        self.padding = padding

        self.strides = [1, 1, strides[0], strides[1]]
        self.kernel_shape = [kernel_size[0], kernel_size[1], in_channels, output_dim]

        if data_format == 'NHWC':
            self.strides = [1, strides[0], strides[1], 1]
            self.kernel_shape = [kernel_size[0], kernel_size[1], in_channels, output_dim]

        with tf.variable_scope(self.scope):
            self.w[self.scope + '_w'] = tf.get_variable('Matrix', self.kernel_shape, tf.float32,
                                                        initializer=initializer)
            variable_summaries(self.w[self.scope + '_w'])

            self.w[self.scope + '_b'] = tf.get_variable('bias', [output_dim],
                                                        initializer=tf.constant_initializer(0.0))
            variable_summaries(self.w[self.scope + '_b'])

    def __call__(self, input_):
        conv = tf.nn.conv2d(input_, self.w[self.scope + '_w'], self.strides, self.padding,
                            data_format=self.data_format)
        out = tf.nn.bias_add(conv, self.w[self.scope + '_b'], self.data_format)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

    def num_features(self, in_height, in_width):
        if self.data_format == 'NHWC':
            out_height = np.ceil(float(in_height - self.kernel_shape[0] + 1) / float(self.strides[1]))
            out_width = np.ceil(float(in_width - self.kernel_shape[1] + 1) / float(self.strides[2]))
        else:
            out_height = np.ceil(float(in_height - self.kernel_shape[0] + 1) / float(self.strides[2]))
            out_width = np.ceil(float(in_width - self.kernel_shape[1] + 1) / float(self.strides[3]))

        num_features = self.kernel_shape[3] * out_height * out_width

        return num_features.astype('int32'), out_height, out_width
