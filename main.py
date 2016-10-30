from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import traceback
import tensorflow as tf

from rva.utils import load_config
from rva.agents.dqn import DQNAgent
from rva.agents.drqn import DRQNAgent
from rva.agents.ramqn import RAMQNAgent
from rva.environment import SimpleGymEnvironment
from rva.environment import GymEnvironment

_agents = {
    'dqn': DQNAgent,
    'drqn': DRQNAgent,
    # 'draqn': DRAQNAgent,
    'ramqn': RAMQNAgent,
    # 'dramqn': DRAMQNAgent,
}


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_data_paths', type=str, action='append')
    # parser.add_argument('--eval_data_paths', type=str, action='append')
    # parser.add_argument('--metadata_path', type=str)
    # parser.add_argument('--output_path', type=str)
    # parser.add_argument('--max_steps', type=int, default=5000)
    # parser.add_argument('--layer1_size', type=int, default=20)
    # parser.add_argument('--layer2_size', type=int, default=10)
    # parser.add_argument('--learning_rate', type=float, default=0.01)
    # parser.add_argument('--epsilon', type=float, default=0.0005)
    # parser.add_argument('--batch_size', type=int, default=30)
    # parser.add_argument('--eval_batch_size', type=int, default=30)
    parser.add_argument('--is_train', action='store_false', help='If specified, the agent performs.')
    parser.add_argument('--config', type=str, default='rva/configs/breakout_agent.cfg')
    parser.add_argument('--agent', type=str, default='dqn')
    parser.add_argument('--env_type', type=str, default='detail', choices=['simple', 'detail'])
    parser.add_argument('--env_name', type=str, default='Breakout-v0', help='The environment name.')
    parser.add_argument('--screen_height', type=int, default=-1, help='The environment screen_height.')
    parser.add_argument('--screen_width', type=int, default=-1, help='The environment screen width.')
    parser.add_argument('--max_reward', type=float, default=1., help='The maximum reward.')
    parser.add_argument('--min_reward', type=float, default=-1, help='The minimum reward.')
    parser.add_argument('--random_start', type=int, default=30, help='The number of random starts.')
    parser.add_argument('--action_repeat', action='store_false', help='Whether to repeat the action or not.')
    parser.add_argument('--gray_scale', action='store_false', help='Whether the screen image is converted to grayscale.')
    parser.add_argument('--display', action='store_true', help='Whether to display the environment or not.')
    parser.add_argument('--gpu_fraction', type=str, default='1/1', help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
    return parser.parse_args(args=argv[1:])


def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction


def main(argv=None):
    args = parse_arguments(sys.argv if argv is None else argv)

    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.config)
    config = load_config(filepath, args.agent)

    if args.env_type == 'simple':
        env = SimpleGymEnvironment(args.env_name, args.env_type, args.screen_width, args.screen_height,
                                   args.gray_scale, args.action_repeat, args.random_start, args.display)
    else:
        env = GymEnvironment(args.env_name, args.env_type, args.screen_width, args.screen_height,
                             args.gray_scale, args.action_repeat, args.random_start, args.display)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(args.gpu_fraction))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        agent = _agents[args.agent](config, env, args.min_reward, args.max_reward, sess)
        if args.is_train:
            agent.train_and_eval()
        else:
            agent.play()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    try:
        main()
    except Exception as e:
        traceback.print_exc()
