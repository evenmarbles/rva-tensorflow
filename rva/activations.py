from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from collections import OrderedDict

from .utils import tf_max
from .utils import tf_sum
from .utils import get_from_module


def linear(x, name=None):
    return x


def tanh(x, name=None):
    """Element-wise tanh"""
    return tf.nn.tanh(x, name=name)


def hard_tanh(x, min_value=-1, max_value=1, name=None):
    """Segment-wise linear approximation of tanh.

    hard_tanh is defined as:
        f(x) = 1, if x > max_value
        f(x) = -1 if x < min_value
        f(x) = x, otherwise.
    """
    return tf.clip_by_value(x, min_value, max_value, name)


def sigmoid(x, name=None):
    """Element-wise sigmoid"""
    return tf.nn.sigmoid(x, name=name)


def hard_sigmoid(x, name=None):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid."""
    input_dtype = x.dtype.base_dtype

    x = (.2 * x) + .5
    return tf.clip_by_value(x, 0, 1, name=name)


def softmax(x, name=None):
    ndims = x.get_shape().ndims
    if ndims == 2:
        return tf.nn.softmax(x, name=name)

    if ndims == 3:
        e = tf.exp(x - tf_max(x, axis=-1, keep_dims=True))
        s = tf_sum(e, axis=-1, keep_dims=True)
        return tf.identity(e / s, name=name)

    raise Exception('Softmax only defined for 2D and 3D tensors: ndims=' + str(ndims))


def log_softmax(x, name=None):
    ndims = x.get_shape().ndims
    if ndims == 2:
        return tf.nn.log_softmax(x, name)

    raise Exception('Softmax only defined for 2D tensors: ndims=' + str(ndims))


def softplus(x, name=None):
    return tf.nn.softplus(x, name=name)


def softsign(x, name=None):
   return tf.nn.softsign(x, name=name)


def relu(x, alpha=0.0, max_value=None, name=None):
    x = tf.nn.relu(x, name=name)
    if max_value is not None:
        x = tf.clip_by_value(x, 0, max_value, name=name)
    if alpha != 0:
        alpha = tf.cast(tf.convert_to_tensor(alpha), x.dtype.base_dtype)
        x -= alpha * tf.nn.relu(-x)
    return x


def _get_defaults():
    return {
        'linear': OrderedDict([('name', None)]),
        'tanh': OrderedDict([('name', None)]),
        'hard_tanh': OrderedDict([('min_value', -1), ('max_value', 1), ('name', None)]),
        'sigmoid': OrderedDict([('name', None)]),
        'hard_sigmoid': OrderedDict([('name', None)]),
        'softmax': OrderedDict([('name', None)]),
        'softplus': OrderedDict([('name', None)]),
        'softsign': OrderedDict([('name', None)]),
        'relu': OrderedDict([('alpha', 0.0), ('max_value', None), ('name', None)]),
    }


def get(identifier):
    if identifier is None:
        return linear
    return get_from_module(identifier, globals(), 'activation', _get_defaults())
