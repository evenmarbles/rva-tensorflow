import six
from six import iteritems

import os
import sys
import time
import cPickle
import numpy as np
import tensorflow as tf
from collections import defaultdict

_UIDS = defaultdict(int)
_SESSION = None

__all__ = ['get_uid',
           'reset_uids',
           'tf_set_session',
           'tf_get_session',
           'tf_clear_session',
           'linear',
           'load_config',
           'dotdict']


def get_uid(prefix):
    _UIDS[prefix] += 1
    return _UIDS[prefix] - 1


def reset_uids():
    global _UIDS
    _UIDS = defaultdict(int)


def tf_set_session(session):
    """Sets the global TF session."""
    global _SESSION
    _SESSION = session


def tf_clear_session():
    global _SESSION
    global _TRAINING_PHASE
    tf.reset_default_graph()
    reset_uids()
    _SESSION = None


def tf_get_session():
    """Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global SmartMind session.

    If no global SmartMind session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `set_session(sess)`.
    """
    global _SESSION
    if tf.get_default_session() is not None:
        return tf.get_default_session()
    if _SESSION is None:
        if not os.environ.get('OMP_NUM_THREADS'):
            _SESSION = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
            _SESSION = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                                        allow_soft_placement=True))
    return _SESSION


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result

    return timed


def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


@timeit
def save_pkl(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print("  [*] save %s" % path)


@timeit
def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
        print("  [*] load %s" % path)
        return obj


@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print("  [*] save %s" % path)


@timeit
def load_npy(path):
    obj = np.load(path)
    print("  [*] load %s" % path)
    return obj


def _normalize_axis(axis, ndims):
    if axis is None:
        return

    axis = tolist(axis)
    for i, a in enumerate(axis):
        if axis is not None and a < 0:
            axis[i] = a % ndims
    if len(axis) == 1:
        axis = axis[0]
        if axis < 0:
            axis = axis % ndims
    return axis


def tf_max(x, axis=None, keep_dims=False):
    axis = _normalize_axis(axis, x.get_shape().ndims)
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keep_dims)


def tf_min(x, axis=None, keep_dims=False):
    axis = _normalize_axis(axis, x.get_shape().ndims)
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keep_dims)


def tf_sum(x, axis=None, keep_dims=False):
    axis = _normalize_axis(axis, x.get_shape().ndims)
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keep_dims)


def tolist(x, convert_tuple=True):
    """Returns the object as a list"""
    if x is None:
        return []
    typelist = (list,)
    if convert_tuple:
        typelist += (tuple,)
    if isinstance(x, typelist):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()

    return [x]


def load_config(filepath, agent):
    try:
        import json
        with open(filepath) as f:
            config = json.load(f)
    except NameError:
        config = {}
        with open(filepath) as f:
            code = compile(f.read(), filepath, 'exec')
            exec (code, None, config)

    return dotdict(config[agent])


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Progbar(object):
    def __init__(self, target, width=30, verbose=1, interval=0.01):
        """
            @param target: total number of steps expected
            @param interval: minimum visual progress update interval (in seconds)
        """
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        """
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
            @param force: force visual progress update
        """
        values = values if values is not None else []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if type(self.sum_values[k]) is list:
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        values = values if values is not None else []
        self.update(self.seen_so_far + n, values)


def variable_summaries(x, write_images=False):
    tensor_name = x.op.name

    mean = tf.reduce_mean(x)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(x - mean)))

    tf.scalar_summary(tensor_name + '/mean/' + x.name, mean)
    tf.scalar_summary(tensor_name + '/stddev/' + x.name, stddev)
    tf.scalar_summary(tensor_name + '/max/' + x.name, tf.reduce_max(x))
    tf.scalar_summary(tensor_name + '/min/' + x.name, tf.reduce_min(x))
    tf.histogram_summary(x.name, x)

    if write_images:
        w_img = tf.squeeze(x)

        shape = w_img.get_shape()
        if len(shape) > 1 and shape[0] > shape[1]:
            w_img = tf.transpose(w_img)

        if len(shape) == 1:
            w_img = tf.expand_dims(w_img, 0)

        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)

        tf.image_summary(x.name, w_img)


def activation_summaries(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, scope=None):
    shape = input_.get_shape().as_list()

    def _linear():
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            tf.truncated_normal_initializer(stddev=stddev))
        variable_summaries(w)

        b = tf.get_variable('bias', [output_size],
                            initializer=tf.constant_initializer(bias_start))
        variable_summaries(b)

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn is not None:
            out = activation_fn(out)
        activation_summaries(out)

        return out, w, b

    if scope is not None:
        with tf.variable_scope(scope):
            return _linear()
    return _linear()


def linear_weights(shape, stddev=0.02, bias_start=0.0, scope='linear'):
    with tf.variable_scope(scope):
        w = tf.get_variable('Matrix', shape, tf.float32,
                            tf.truncated_normal_initializer(stddev=stddev))
        variable_summaries(w)

        b = tf.get_variable('bias', [shape[1]],
                            initializer=tf.constant_initializer(bias_start))
        variable_summaries(b)

        return w, b


def linear_add_bias(input_, w, b, activation_fn=None, scope='linear'):
    with tf.variable_scope(scope):
        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn is not None:
            out = activation_fn(out)
        activation_summaries(out)

        return out


def conv2d(x, output_dim, kernel_size, strides, initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu, data_format='NHWC', padding='VALID', scope='conv2d'):
    def _conv2d():
        stride_ = [1, 1, strides[0], strides[1]]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]

        if data_format == 'NHWC':
            stride_ = [1, strides[0], strides[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('Matrix', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride_, padding, data_format=data_format)

        b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

        if activation_fn is not None:
            out = activation_fn(out)

        return out, w, b

    if scope is not None:
        with tf.variable_scope(scope):
            return _conv2d()
    return _conv2d()


def loglikelihood(means, sampled, sigma):
    # means = tf.pack(means)  # mu = [timesteps, batch_sz, loc_dim]
    # sampled = tf.pack(sampled)  # same shape as mu
    gaussian = tf.contrib.distributions.Normal(means, sigma)
    logll = gaussian.log_pdf(sampled)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    # logll = tf.transpose(logll)  # [batch_sz, timesteps]
    return logll


def get_from_module(identifier, modules, module_name, defaults,
                    instantiate=False, kwargs=None):
    name = None
    if isinstance(identifier, six.string_types):
        name = identifier
    elif isinstance(identifier, dict):
        name = identifier['name']
    elif isinstance(identifier, (list, tuple)):
        name = identifier[0]

    if name is not None:
        res = modules.get(name)
        if not res:
            raise Exception('Invalid {}: {}'.format(module_name, name))

        if instantiate:
            return res(**process_params(identifier, defaults, kwargs))
        return res

    return identifier


def process_params(identifier, defaults, kwargs):
    name = None
    kwargs = kwargs if kwargs is not None else {}
    args = ()

    if isinstance(identifier, six.string_types):
        name = identifier
    elif isinstance(identifier, dict):
        name = identifier.pop('name')
        kwargs = identifier
    elif isinstance(identifier, (list, tuple)):
        identifier = list(identifier)
        name = identifier.pop(0)
        args = identifier

    if not args and not kwargs:
        return {}

    defaults = defaults[name]

    p = defaults.copy()

    if kwargs:
        if not all(k in defaults for k in kwargs.keys()):
            raise Exception('{}: Parameter mismatch.'.format(identifier))
        p.update(kwargs)

    if args:
        if len(kwargs) > len(defaults):
            raise Exception('{}: Too many parameters given.'.format(identifier))

        p.update(dict(zip(defaults.keys(), args)))

    return p
