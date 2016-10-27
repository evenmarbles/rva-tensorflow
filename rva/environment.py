# noinspection PyUnresolvedReferences
from six.moves import range

import sys
import cv2
import gym
import random
import numpy as np

from utils import get_time


class Environment(object):
    def __init__(self, env_name, env_type, screen_width=-1, screen_height=-1,
                 gray_scale=True, action_repeat=1, random_start=30, display=False):
        self.env = gym.make(env_name)
        self.type = env_type

        self.gray_scale = gray_scale
        self.action_repeat = action_repeat
        self.random_start = random_start

        self.display = display
        self.dims = (screen_width, screen_height)

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @property
    def name(self):
        return self.env.spec.id

    @property
    def screen(self):
        img = self._screen
        if self.dims != (-1, -1):
            img = cv2.resize(img, self.dims)
        if self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.
            img = np.expand_dims(img, 2)
        img = np.transpose(img, [2, 0, 1])
        return img
        # return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

    @property
    def observation_space(self):
        shape = self.env.observation_space.shape
        num_channels = shape[2]
        if self.gray_scale:
            num_channels = 1

        if self.dims == (-1, -1):
            return (num_channels,) + shape[:2]
        return num_channels, self.dims[1], self.dims[0]

    @property
    def action_size(self):
        if 'gym.envs.doom' in sys.modules.keys():
            if self.env.spec.id == 'DoomHealthGathering-v0':
                return 3
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        self.render()

    def start_monitor(self):
        gym_dir = '/tmp/%s-%s' % (self.name, get_time())
        self.env.monitor.start(gym_dir)

    def stop_monitor(self):
        self.env.monitor.close()
        # gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')


class GymEnvironment(Environment):
    def __init__(self, env_name, env_type, screen_width, screen_height, gray_scale=True,
                 action_repeat=1, random_start=30, display=False):
        super(GymEnvironment, self).__init__(env_name, env_type, screen_width, screen_height,
                                             gray_scale, action_repeat, random_start, display)

    def act(self, action, is_training=True):
        cumulated_reward = 0
        start_lives = self.lives

        for _ in range(self.action_repeat):
            self._step(action)
            cumulated_reward = cumulated_reward + self.reward

            if is_training and start_lives > self.lives:
                cumulated_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = cumulated_reward

        self.after_act(action)
        return self.state


class SimpleGymEnvironment(Environment):
    def __init__(self, env_name, env_type, screen_width, screen_height, gray_scale=True,
                 action_repeat=1, random_start=30, display=False):
        super(SimpleGymEnvironment, self).__init__(env_name, env_type, screen_width, screen_height,
                                                   gray_scale, action_repeat, random_start, display)

    def act(self, action, is_training=True):
        self._step(action)

        self.after_act(action)
        return self.state
