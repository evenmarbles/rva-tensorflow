import numpy as np
import tensorflow as tf

from .base import BaseModel
from ..glimpse import GlimpseNetwork
from ..glimpse import LocationNetwork
from ..layers import FullyConnected
from ..utils import loglikelihood


class RAMQNAgent(BaseModel):
    @property
    def name(self):
        return 'RAM-QN'

    def __init__(self, args, env, min_reward=-1.0, max_reward=1.0, sess=None):
        args['memory_size'] = 30 * args.scale                 # 300,000
        args['observation_space'] = env.observation_space

        self.initial_h = None
        self.initial_c = None

        self.prev_h = None
        self.prev_c = None

        super(RAMQNAgent, self).__init__(args, env, min_reward, max_reward, sess, True)

    def greedy(self, s_t):
        # reshape to account for history length = 1
        shape = [-1, 1, self.num_channels, self.screen_height, self.screen_width]
        if self.data_format == 'NHWC':
            shape = [-1, self.screen_height, self.screen_width, self.num_channels, 1]
        s_t = s_t.reshape(shape)

        if self.prev_h is None:
            self.prev_h, self.prev_c = self.sess.run([self.initial_state_h, self.initial_state_c],
                                                     feed_dict={self.s_t: s_t})

        action, self.prev_h, self.prev_c = self.sess.run([self.q_action, self.next_state_h, self.next_state_c],
                                                         feed_dict={
                                                             self.s_t: s_t,
                                                             self.lstm_h: self.prev_h,
                                                             self.lstm_c: self.prev_c,
                                                             self.training_phase: 0
                                                         })
        return action[0]

    def get_q_update(self, s_t, action, reward, s_t_plus_1, terminal, op_list, return_q2_max=False):
        # TODO: Monte-Carlo sampling could be implemented here by repeating
        #   s_t, action, s_t_plus_1, and terminal M times

        if self.initial_h is None:
            self.initial_h, self.initial_c = self.sess.run([self.initial_state_h, self.initial_state_c],
                                                           feed_dict={self.s_t: s_t})

        q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})  # [history_length, batch_size, num_actions]

        targets = []
        q2_max = []
        for i in range(self.history_length):
            terminal_ = np.array(terminal[:, i]) + 0.
            max_q_t_plus_1 = (1. - terminal_) * self.discount * np.max(q_t_plus_1[i], axis=1)
            q2_max.append(max_q_t_plus_1)
            targets.append(max_q_t_plus_1 + reward[:, i])

        result = self.sess.run(op_list, {   # self.delta_loss, self.baseline_mse, self.logllratio
                self.target_q_t: targets,
                self.action: action,
                self.s_t: s_t,
                self.learning_rate_step: self.step,
                self.lstm_h: self.initial_h,
                self.lstm_c: self.initial_c,
                self.training_phase: 1
            })

        if return_q2_max:
            result += [np.array(q2_max)]
        return result


    def _build(self, args):
        self.training_phase = tf.placeholder('uint8', name='training_phase')

        initial_q = tf.Variable(0, dtype='float32', trainable=False, validate_shape=False, name='initial_q')
        initial_base = tf.Variable(0, dtype='float32', trainable=False, validate_shape=False, name='initial_base')
        initial_mean = tf.Variable(0, dtype='float32', trainable=False, validate_shape=False, name='initial_mean')
        initial_sampled = tf.Variable(0, dtype='float32', trainable=False, validate_shape=False,
                                      name='initial_sampled')

        # training network
        with tf.variable_scope('prediction') as vs:
            if self.data_format == 'NHWC':
                self.s_t = tf.placeholder('float32',
                                          [None, self.screen_height, self.screen_width, self.num_channels, None],
                                          name='s_t')
                inpt = tf.transpose(self.s_t, perm=[4, 0, 1, 2, 3])  # [history_length, batch_size, ...]
            else:
                self.s_t = tf.placeholder('float32',
                                          [None, None, self.num_channels, self.screen_height, self.screen_width],
                                          name='s_t')
                inpt = tf.transpose(self.s_t, perm=[1, 0, 2, 3, 4])  # [history_length, batch_size, ...]

            shape = tf.shape(inpt)
            inpt = tf.reshape(inpt, [shape[0], shape[1], self.screen_height * self.screen_width * self.num_channels])

            self.cell = tf.nn.rnn_cell.LSTMCell(args.cell_size)
            self.lstm_h = tf.placeholder('float32', shape=[None, self.cell.output_size], name='lstm_h')
            self.lstm_c = tf.placeholder('float32', shape=[None, self.cell.output_size], name='lstm_s')

            with tf.variable_scope('location_network'):
                self.location_network = LocationNetwork(self.cell.output_size, args.loc_dim, args.loc_stddev)
                self.w.update(self.location_network.w)

            with tf.variable_scope('glimpse_network'):
                self.glimpse_network = GlimpseNetwork(args)
                self.w.update(self.glimpse_network.w)

            # build core network
            initial_state = self.cell.zero_state(shape[1], tf.float32)
            self.initial_state_h = initial_state.h
            self.initial_state_c = initial_state.c

            q_network = FullyConnected((self.cell.output_size, self.num_actions), stddev=0.01, scope='q')
            self.w.update(q_network.w)
            base_network = FullyConnected((self.cell.output_size, 1), stddev=0.01, scope='base')

            history_length = tf.cond(tf.cast(self.training_phase, 'bool'),
                                     lambda: tf.convert_to_tensor(self.history_length, dtype='int32'),
                                     lambda: tf.constant(1))

            def body(i_, z_prev, base, mean, sampled, lstm_c, lstm_h):
                _in = inpt[i_]
                loc, mean_loc = self.location_network(lstm_h)
                glimpse = self.glimpse_network(loc, _in)

                _out, _state = self.cell(glimpse, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))
                tf.get_variable_scope().reuse_variables()

                q_pred = q_network(_out)
                base_pred = tf.squeeze(base_network(_out))

                q_next = tf.cond(tf.equal(i_, 0),
                                 lambda: tf.assign(initial_q, tf.expand_dims(q_pred, 0),
                                                   validate_shape=False),
                                 lambda: tf.concat(0, [z_prev, tf.expand_dims(q_pred, 0)]))

                base_next, mean_next, sampled_next = tf.cond(tf.equal(i_, 0),
                                                             lambda: (
                                                                 tf.assign(initial_base,
                                                                           tf.expand_dims(base_pred, 0),
                                                                           validate_shape=False),
                                                                 tf.assign(initial_mean,
                                                                           tf.expand_dims(mean_loc, 0),
                                                                           validate_shape=False),
                                                                 tf.assign(initial_sampled, tf.expand_dims(loc, 0),
                                                                           validate_shape=False)),
                                                             lambda: (
                                                                 tf.concat(0, [base, tf.expand_dims(base_pred, 0)]),
                                                                 tf.concat(0, [mean, tf.expand_dims(mean_loc, 0)]),
                                                                 tf.concat(0, [sampled, tf.expand_dims(loc, 0)])))

                i_next = tf.add(i_, 1)
                return i_next, q_next, base_next, mean_next, sampled_next, _state.c, _state.h

            _, self.q, baselines, mean_locs, sampled_locs, self.next_state_c, self.next_state_h = tf.while_loop(
                lambda i_, pred, base, mean, sampled, lstm_c, lstm_h: tf.less(i_, history_length), body,
                [tf.constant(0), initial_q, initial_base, initial_mean, initial_sampled, self.lstm_c, self.lstm_h])

            for var in tf.all_variables():
                if var.name.startswith(vs.name + '/LSTMCell'):
                    key = 'lstm_' + var.name[20: var.name.index(':')].encode('ascii')
                    self.w[key] = var

            # baselines = tf.transpose(baselines)  # [batch_size, time_steps]
            self.baselines = tf.stop_gradient(baselines)

            self.q_action = tf.argmax(self.q[-1], dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q[-1], 0)
            for idx in range(self.num_actions):
                q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.merge_summary(q_summary, 'q_summary')

        # target network
        with tf.variable_scope('target') as vs:
            if self.data_format == 'NHWC':
                self.target_s_t = tf.placeholder('float32',
                                                 [None, self.screen_height, self.screen_width, self.num_channels,
                                                  self.history_length], name='s_t')
                inpt = tf.transpose(self.target_s_t, perm=[4, 0, 1, 2, 3])  # [history_length, batch_size, ...]
            else:
                self.target_s_t = tf.placeholder('float32',
                                                 [None, self.history_length, self.num_channels, self.screen_height,
                                                  self.screen_width], name='s_t')
                inpt = tf.transpose(self.target_s_t, perm=[1, 0, 2, 3, 4])  # [history_length, batch_size, ...]

            shape = tf.shape(inpt)
            inpt = tf.reshape(inpt, [shape[0], shape[1], self.screen_height * self.screen_width * self.num_channels])

            self.target_cell = tf.nn.rnn_cell.LSTMCell(args.cell_size)
            target_initial_state = self.target_cell.zero_state(shape[1], tf.float32)

            with tf.variable_scope('location_network'):
                self.target_location_network = LocationNetwork(self.target_cell.output_size, args.loc_dim,
                                                               args.loc_stddev)
                self.t_w.update(self.target_location_network.w)

            with tf.variable_scope('glimpse_network'):
                self.target_glimpse_network = GlimpseNetwork(args)
                self.t_w.update(self.target_glimpse_network.w)

            target_q_network = FullyConnected((self.target_cell.output_size, self.num_actions), stddev=0.01,
                                              scope='q')
            self.t_w.update(target_q_network.w)

            # build core network

            target_q = []

            def _get_next_rnn_input(frame, output_):
                loc, _ = self.target_location_network(output_)
                next_glimpse = self.target_glimpse_network(loc, frame)
                return next_glimpse

            state = target_initial_state
            prev = target_initial_state.h
            for i in range(self.history_length):
                with tf.variable_scope("loop_function", reuse=True):
                    in_ = _get_next_rnn_input(inpt[i], prev)
                output, state = self.target_cell(in_, state)
                tf.get_variable_scope().reuse_variables()
                target_q.append(target_q_network(output))
                prev = output

            for var in tf.all_variables():
                if var.name.startswith(vs.name + '/LSTMCell'):
                    key = 'lstm_' + var.name[16: var.name.index(':')].encode('ascii')
                    self.t_w[key] = var

            self.target_q = tf.pack(target_q)  # [time_steps, batch_size, num_actions]

        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [self.history_length, None], name='target_q_t')
            self.action = tf.placeholder('int64', [None, self.history_length], name='action')

            def optimizer_body(i_, z_prev):
                action_one_hot = tf.one_hot(self.action[:, i], self.num_actions, 1.0, 0.0, name='action_one_hot')
                q_acted = tf.reduce_sum(self.q[i_] * action_one_hot, reduction_indices=1, name='q_acted')

                delta = self.target_q_t - q_acted
                clipped_delta = tf.clip_by_value(delta, self.min_delta, self.max_delta, name='clipped_delta')

                z_prev = tf.cond(tf.equal(i_, 0),
                                 lambda: tf.assign(initial_q, tf.expand_dims(clipped_delta, 0),
                                                   validate_shape=False),
                                 lambda: tf.concat(0, [z_prev, tf.expand_dims(clipped_delta, 0)]))
                i_next = tf.add(i_, 1)
                return i_next, z_prev

            _, self.clipped_delta = tf.while_loop(
                lambda i_, delta: tf.less(i_, self.history_length), optimizer_body,
                [tf.constant(0), initial_q])

            self.delta_loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
            tf.add_to_collection('losses', self.delta_loss)

            advs = self.target_q_t - self.baselines

            logll = loglikelihood(mean_locs, sampled_locs, args.loc_stddev)
            self.logllratio = -tf.reduce_mean(logll * advs, name='logllratio')
            tf.add_to_collection('losses', self.logllratio)

            self.baseline_mse = tf.reduce_mean(tf.square(advs), name='base_mse')
            tf.add_to_collection('losses', self.baseline_mse)

            self.loss = tf.add_n(tf.get_collection('losses'))

            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.learning_rate_step,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))
            self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
