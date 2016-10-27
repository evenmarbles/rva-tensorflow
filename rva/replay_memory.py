"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import numpy as np

from utils import save_npy, load_npy


class ReplayMemory:
    def __init__(self, config, model_dir, full_history=False):
        self.model_dir = model_dir
        self.full_history = full_history

        num_channels, screen_height, screen_width = config.observation_space
        self.data_format = config.data_format

        self.dims = (num_channels, screen_height, screen_width)
        if self.data_format == 'NHWC':
            self.dims = (screen_height, screen_width, num_channels)

        self.cnn_format = config.cnn_format
        self.memory_size = config.memory_size

        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.integer)

        if self.data_format == 'NCHW':
            self.screens = np.empty((self.memory_size, num_channels, screen_height, screen_width), dtype=np.float16)
        else:
            self.screens = np.empty((self.memory_size, screen_height, screen_width, num_channels), dtype=np.float16)

        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = config.history_length
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)

        if self.full_history:
            self.actionshist = np.empty((self.batch_size, self.history_length), dtype=np.uint8)
            self.rewardshist = np.empty((self.batch_size, self.history_length), dtype=np.integer)
            self.terminalshist = np.empty((self.batch_size, self.history_length), dtype=np.bool)

    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    # def getState(self, index):
    #     assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    #     # normalize index to expected range, allows negative indexes
    #     index %= self.count
    #     # if is not in the beginning of matrix
    #     if index >= self.history_length - 1:
    #         # use faster slicing
    #         return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    #     else:
    #         # otherwise normalize indexes and use slower list based access
    #         indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
    #         return self.screens[indexes, ...]

    def get(self, index, container_type='state'):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
        container = {
            'screens': self.screens,
            'actions': self.actions,
            'rewards': self.rewards,
            'terminals': self.terminals
        }
        # normalize index to expected range, allows negative indexes
        index %= self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return container[container_type][(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return container[container_type][indexes, ...]

    def sample(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size

        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current > index - self.history_length:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.get(index - 1, 'screens')
            self.poststates[len(indexes), ...] = self.get(index, 'screens')

            if self.full_history:
                self.actionshist[len(indexes), ...] = self.get(index - 1, 'actions')
                self.rewardshist[len(indexes), ...] = self.get(index - 1, 'rewards')
                self.terminalshist[len(indexes), ...] = self.get(index, 'terminals')

            indexes.append(index - 1)

        prestates = self.prestates
        poststates = self.poststates
        if self.cnn_format == 'NHWC':
            prestates = np.transpose(self.prestates, (0, 3, 4, 2, 1))
            poststates = np.transpose(self.poststates, (0, 3, 4, 2, 1))

        if not self.full_history:
            return prestates, self.actions[indexes], self.rewards[indexes], poststates, self.terminals[
                np.array(indexes) + 1]

        return prestates, self.actionshist, self.rewardshist, poststates, self.terminalshist

    def save(self):
        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates', 'actionshist',
                     'rewardshist', 'terminalshist'],
                    [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates,
                     self.actionshist, self.rewardshist, self.terminalshist])):
            save_npy(array, os.path.join(self.model_dir, name))

    def load(self):
        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates', 'actionshist',
                     'rewardshist', 'terminalshist'],
                    [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates,
                     self.actionshist, self.rewardshist, self.terminalshist])):
            array = load_npy(os.path.join(self.model_dir, name))
