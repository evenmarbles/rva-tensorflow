import os
import time
import random
import pprint
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from ..replay_memory import ReplayMemory
from ..utils import save_pkl, load_pkl

pp = pprint.PrettyPrinter().pprint


class BaseModel(object):
    """Abstract object representing an Reader model."""

    def __init__(self, args, env=None, min_reward=-1.0, max_reward=1.0, sess=None, full_history=False):
        self._saver = None
        self.config = args
        pp(args)

        self.weight_dir = 'weights'

        self.step = 0

        self.sess = sess
        self.env = env

        self.history_length = args.history_length

        self.data_format = args.data_format
        self.num_channels, self.screen_height, self.screen_width = self.env.observation_space
        self.num_actions = self.env.action_size
        self.memory_size = args.memory_size

        self.discount = args.discount

        self.max_steps = 500 * args.scale  # 5,000,000
        self.learn_start = 2.5 * args.scale  # 25,000
        self.target_q_update_step = 1 * args.scale  # 10,000

        self.ep_start = args.ep_start
        self.ep_end = args.ep_end
        self.ep_end_t = args.memory_size

        self.learning_rate = args.learning_rate
        self.learning_rate_minimum = args.learning_rate_minimum
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_decay_step = 5 * args.scale  # 50,000

        self.min_delta = args.min_delta
        self.max_delta = args.max_delta

        self.min_reward = min_reward
        self.max_reward = max_reward

        self.train_frequency = args.train_frequency

        self.test_step = 5 * args.scale  # 50,000
        self.save_step = 10 * self.test_step  # 100,000

        self.memory = ReplayMemory(args, self.model_dir, full_history)

        # NEW NEW NEW
        self.eval_freq = args.eval_freq
        self.eval_steps = args.eval_steps

        self.last_screen = None

        self.valid_s_t = None
        self.valid_action = None
        self.valid_reward = None
        self.valid_s_t_plus_1 = None
        self.valid_terminal = None

        self.avg_reward_per_ep_history = np.zeros(int(np.ceil((self.max_steps - self.learn_start) / self.eval_freq)),
                                                  dtype=np.float16)
        self.v_history = []
        self.td_history = []
        self.reward_counts = []
        self.episode_counts = []
        self.train_time_history = []

        self.best = {}
        # END NEW

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.learning_rate_op = None
        self.learning_rate_step = None

        self.build(args)

    def train_and_eval(self):
        start_step = self.step_op.eval()

        self.total_loss = 0.
        self.total_q = 0.
        self.update_count = 0

        screen, reward, action, terminal = self.env.new_random_game()

        self.init_history(screen)

        eval_count = 0

        start_time = time.time()
        for self.step in tqdm(range(start_step, self.max_steps), ncols=70, initial=start_step):
            # 1. predict
            action = self.predict(self.get_state(screen))
            # 2. act
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()

            if self.step > self.learn_start:
                if self.step == self.learn_start + 1:
                    train_time = time.time() - start_time
                    self.sample_validation_data()
                    start_time = time.time() - train_time

                if self.step % self.eval_freq == 0:
                    self.train_time_history.append(time.time() - start_time)
                    self.evaluate(eval_count)

                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0

                    eval_count += 1
                    start_time = time.time()

    def evaluate(self, eval_count):
        total_reward = 0.
        avg_reward_per_ep = 0.
        n_rewards = 0
        n_episodes = 0
        ep_reward = 0.
        ep_rewards = []
        action_count = np.zeros(self.num_actions)

        screen, reward, action, terminal = self.env.new_random_game()

        for estep in range(self.eval_steps):
            action = self.predict(self.get_state(screen), test_ep=0.05)
            action_count[action] += 1

            self.update_history(screen)

            # play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal = self.env.act(action, is_training=False)

            ep_reward += reward
            total_reward += reward
            if reward != 0:
                n_rewards += 1

            if terminal:
                n_episodes += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.

                screen, reward, action, terminal = self.env.new_random_game()

        if n_episodes == 0:
            n_episodes += 1
        else:
            avg_reward_per_ep = total_reward / n_episodes  # average reward per episode

        if eval_count == 0 or avg_reward_per_ep > np.max(self.avg_reward_per_ep_history):
            w = self.sess.run(self.w.values())
            self.best = dict(zip(self.w.keys(), w))

        v_avg, tderr_avg = self.compute_validation_statistics()
        self.v_history.append(v_avg)
        self.td_history.append(tderr_avg)

        self.avg_reward_per_ep_history[eval_count] = avg_reward_per_ep
        self.reward_counts.append(n_rewards)
        self.episode_counts.append(n_episodes)

        avg_reward = total_reward / self.eval_steps
        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
        except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, '
              'max_ep_r: %.4f, min_ep_r: %.4f, avg_r_per_ep: %.4f, # game: %d' % (avg_reward, avg_loss, avg_q,
                                                                                  avg_ep_reward, max_ep_reward,
                                                                                  min_ep_reward, avg_reward_per_ep,
                                                                                  n_episodes))

        # save model
        self.step_assign_op.eval({self.step_input: self.step + 1})
        self.save_model(self.step + 1)

        self.inject_summary({
            'average.reward per episode': avg_reward_per_ep,
            'average.value': v_avg,
            'average.td error': tderr_avg,
            'average.reward': avg_reward,
            'average.loss': avg_loss,
            'average.q': avg_q,
            'episode.max reward': max_ep_reward,
            'episode.min reward': min_ep_reward,
            'episode.avg reward': avg_ep_reward,
            'episode.num of game': n_episodes,
            'episode.rewards': ep_rewards,
            'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
        }, self.step)

        self.save_eval_data_to_pkl()

    def sample_validation_data(self):
        self.valid_s_t, self.valid_action, self.valid_reward, self.valid_s_t_plus_1, \
        self.valid_terminal = self.memory.sample()

    def compute_validation_statistics(self):
        clipped_delta, q2_max = self.get_q_update(self.valid_s_t, self.valid_action, self.valid_reward,
                                                  self.valid_s_t_plus_1, self.valid_terminal,
                                                  [self.clipped_delta], return_q2_max=True)
        return q2_max.mean(), np.fabs(clipped_delta).mean()

    def train(self):
        start_step = self.step_op.eval()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, reward, action, terminal = self.env.new_random_game()

        self.init_history(screen)

        for self.step in tqdm(range(start_step, self.max_steps), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict(self.get_state(screen))
            # 2. act
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, '
                          'max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' % (avg_reward, avg_loss, avg_q, avg_ep_reward,
                                                                          max_ep_reward, min_ep_reward, num_game))

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        self.save_model(self.step + 1)

                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                    if self.step > 180:
                        self.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of game': num_game,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
                        }, self.step)

                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

    def predict(self, s_t, test_ep=None):
        ep = test_ep or (self.ep_end +
                         max(0., (self.ep_start - self.ep_end)
                             * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            return random.randrange(self.num_actions)

        return self.greedy(s_t)

    def observe(self, screen, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))

        self.update_history(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == 1:
                self.update_target_q_network()

    def q_learning_mini_batch(self):
        if self.memory.count < self.memory.history_length:
            return

        s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
        _, q_t, loss, summary_str = self.get_q_update(s_t, action, reward, s_t_plus_1, terminal,
                                                      [self.optim, self.q, self.loss, self.q_summary])

        self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def greedy(self, s_t):
        raise NotImplementedError

    def get_q_update(self, s_t, action, reward, s_t_plus_1, terminal, op_list, return_q2_max=False):
        raise NotImplementedError

    def init_history(self, screen):
        pass

    def update_history(self, screen):
        pass

    def get_state(self, screen):
        return screen

    def build(self, args):
        self.w = {}
        self.t_w = {}

        self._build(args)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward per episode', 'average.value', 'average.td error',
                                   'average.reward', 'average.loss', 'average.q',
                                   'episode.max reward', 'episode.min reward', 'episode.avg reward',
                                   'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.scalar_summary("%s-%s/%s" % (self.env.name, self.env.type, tag),
                                                          self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.histogram_summary(tag, self.summary_placeholders[tag])

            self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

        tf.initialize_all_variables().run()

        self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

        self.load_model()
        self.update_target_q_network()

    def _build(self, args):
        raise NotImplementedError

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
            })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step)

    def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
        if test_ep is None:
            test_ep = self.ep_end

        if not render:
            self.env.start_monitor()

        best_reward, best_idx = 0, 0
        for idx in range(n_episode):
            screen, reward, action, terminal = self.env.new_random_game()
            current_reward = 0

            for t in tqdm(range(n_step), ncols=70):
                # 1. predict
                action = self.predict(screen, test_ep)
                # 2. act
                screen, reward, terminal = self.env.act(action, is_training=False)

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

    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            print(" [*] Loading checkpoints...")

            if ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                fname = os.path.join(self.checkpoint_dir, ckpt_name)
                self.saver.restore(self.sess, fname)
                print(" [*] Load SUCCESS: %s" % fname)
                return True
            else:
                print(" [!] Load FAILED: %s" % self.checkpoint_dir)
                return False

    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self, cpu_mode=False):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}

            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

        self.update_target_q_network()

    def save_eval_data_to_pkl(self):
        data = {
            'avg_reward_per_ep_history': self.avg_reward_per_ep_history,
            'v_history': self.v_history,
            'td_history': self.td_history,
            'reward_counts': self.reward_counts,
            'episode_counts': self.episode_counts,
            'best': self.best,
            'train_time_history': self.train_time_history,
        }
        with open(self.checkpoint_dir + 'eval_data.pkl', 'wb') as f:
            pickle.dump(data, f)

    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)

    @property
    def model_dir(self):
        model_dir = ''
        if self.env is not None:
            model_dir = self.env.name
        if hasattr(self, 'name'):
            model_dir += '/' + self.name
        # for k, v in self.config.items():
        #     if not k.startswith('_') and k not in ['display']:
        #         model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v]) if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
