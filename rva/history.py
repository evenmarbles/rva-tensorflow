import numpy as np


class History:
    def __init__(self, config):
        self.cnn_format = config.cnn_format

        self.length = config.history_length

        num_channels, screen_height, screen_width = config.observation_space

        self.history = np.zeros(
            [self.length, num_channels, screen_height, screen_width], dtype=np.float32)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
            return np.transpose(self.history, (2, 3, 1, 0))
        return self.history
