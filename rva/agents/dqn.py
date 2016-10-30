import numpy as np
import tensorflow as tf

from .. import activations

from .base import BaseModel
from ..history import History
from ..utils import linear, conv2d


class DQNAgent(BaseModel):
    @property
    def name(self):
        return 'DQN'

    def __init__(self, args, env, min_reward=-1.0, max_reward=1.0, sess=None):
        args['memory_size'] = 30 * args.scale      # 300,000
        args['observation_space'] = env.observation_space

        self.double_q = False
        self.dueling = False

        self.history = History(args)

        super(DQNAgent, self).__init__(args, env, min_reward, max_reward, sess)

    def greedy(self, s_t):
        return self.q_action.eval({self.s_t: [s_t]})[0]

    def get_q_update(self, s_t, action, reward, s_t_plus_1, terminal, op_list, return_q2_max=False):
        if self.double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})

            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.target_s_t: s_t_plus_1,
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            })
            max_q_t_plus_1 = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action
            target_q_t = max_q_t_plus_1 + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = (1. - terminal) * self.discount * np.max(q_t_plus_1, axis=1)
            target_q_t = max_q_t_plus_1 + reward

        result = self.sess.run(op_list, {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step,
        })

        if return_q2_max:
            result += [max_q_t_plus_1]
        return result

    def init_history(self, screen):
        for _ in range(self.history_length):
            self.history.add(screen)

    def update_history(self, screen):
        self.history.add(screen)

    def get_state(self, screen):
        return self.history.get()

    def _build(self, args):
        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = activations.get(args.activation)

        # training network
        with tf.variable_scope('prediction'):
            if self.data_format == 'NHWC':
                self.s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.num_channels,
                                                      self.history_length], name='s_t')
                inpt = tf.reshape(self.s_t,
                                  [-1, self.screen_height, self.screen_width, self.history_length * self.num_channels])
            else:
                self.s_t = tf.placeholder('float32', [None, self.history_length, self.num_channels, self.screen_height,
                                                      self.screen_width], name='s_t')
                inpt = tf.reshape(self.s_t,
                                  [-1, self.history_length * self.num_channels, self.screen_height, self.screen_width])

            layer_count = -1
            for output_dim, kernel, strides in zip(args.num_conv_units, args.kernel_size, args.kernel_strides):
                layer_count += 1
                scope = 'l' + str(layer_count)
                inpt, self.w[scope + '_w'], self.w[scope + '_b'] = conv2d(inpt, output_dim, kernel,
                                                                          strides, initializer,
                                                                          activation_fn, self.data_format,
                                                                          scope=scope)

            shape = inpt.get_shape().as_list()
            inpt = tf.reshape(inpt, [-1, np.prod(shape[1:])])  # flatten the layer

            if self.dueling:
                self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = linear(inpt, args.num_hidden[0],
                                                                                activation_fn=activation_fn,
                                                                                scope='value_hid')

                self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = linear(inpt, args.num_hidden[0],
                                                                              activation_fn=activation_fn,
                                                                              scope='adv_hid')

                self.value, self.w['val_w_out'], self.w['val_w_b'] = linear(self.value_hid, 1,
                                                                            scope='value_out')

                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = linear(self.adv_hid,
                                                                                self.num_actions,
                                                                                scope='adv_out')

                # Average Dueling
                self.q = self.value + (self.advantage -
                                       tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:
                for output_dim in args.num_hidden:
                    layer_count += 1
                    scope = 'l' + str(layer_count)
                    inpt, self.w[scope + '_w'], self.w[scope + '_b'] = linear(inpt, output_dim,
                                                                              activation_fn=activation_fn,
                                                                              scope=scope)

                self.q, self.w['q_w'], self.w['q_b'] = linear(inpt, self.num_actions, scope='q')

            self.q_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.num_actions):
                q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.merge_summary(q_summary, 'q_summary')

        # target network
        with tf.variable_scope('target'):
            if self.data_format == 'NHWC':
                self.target_s_t = tf.placeholder('float32',
                                                 [None, self.screen_height, self.screen_width, self.num_channels,
                                                  self.history_length], name='s_t')
                inpt = tf.reshape(self.target_s_t,
                                  [-1, self.screen_height, self.screen_width, self.history_length * self.num_channels])
            else:
                self.target_s_t = tf.placeholder('float32',
                                                 [None, self.history_length, self.num_channels, self.screen_height,
                                                  self.screen_width], name='s_t')
                inpt = tf.reshape(self.target_s_t,
                                  [-1, self.history_length * self.num_channels, self.screen_height, self.screen_width])

            layer_count = -1
            for output_dim, kernel, strides in zip(args.num_conv_units, args.kernel_size, args.kernel_strides):
                layer_count += 1
                scope = 'l' + str(layer_count)
                inpt, self.t_w[scope + '_w'], self.t_w[scope + '_b'] = conv2d(inpt, output_dim, kernel,
                                                                              strides, initializer,
                                                                              activation_fn, self.data_format,
                                                                              scope=scope)

            shape = inpt.get_shape().as_list()
            inpt = tf.reshape(inpt, [-1, np.prod(shape[1:])])  # flatten the layer

            if self.dueling:
                self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = linear(inpt, args.num_hidden[0],
                                                                                      activation_fn=activation_fn,
                                                                                      scope='value_hid')

                self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = linear(inpt, args.num_hidden[0],
                                                                                    activation_fn=activation_fn,
                                                                                    scope='adv_hid')

                self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = linear(self.t_value_hid, 1,
                                                                                  scope='value_out')

                self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = linear(self.t_adv_hid, self.num_actions,
                                                                                      scope='adv_out')

                # Average Dueling
                self.target_q = self.t_value + (self.t_advantage -
                                                tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
            else:
                for output_dim in args.num_hidden:
                    layer_count += 1
                    scope = 'l' + str(layer_count)
                    inpt, self.t_w[scope + '_w'], self.t_w[scope + '_b'] = linear(inpt, output_dim,
                                                                                  activation_fn=activation_fn,
                                                                                  scope=scope)

                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = linear(inpt, self.num_actions, scope='q')

            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted
            self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

            self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
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

    def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
        if test_ep is None:
            test_ep = self.ep_end

        test_history = History(self.config)

        if not render:
            self.env.start_monitor()

        from tqdm import tqdm

        best_reward, best_idx = 0, 0
        for idx in range(n_episode):
            screen, reward, action, terminal = self.env.new_random_game()
            current_reward = 0

            for _ in range(test_history.length):
                test_history.add(screen)

            for _ in tqdm(range(n_step), ncols=70):
                # 1. predict
                action = self.predict(test_history.get(), test_ep)
                # 2. act
                screen, reward, terminal = self.env.act(action, is_training=False)
                # 3. observe
                test_history.add(screen)

                current_reward += reward
                if terminal:
                    break

            if current_reward > best_reward:
                best_reward = current_reward
                best_idx = idx

            print("=" * 30)
            print(" [%d] Best reward : %d" % (best_idx, best_reward))
            print("=" * 30)

        if not render:
            self.env.stop_monitor()
