import traceback
import argparse
import cv2
import tensorflow as tf
# from tqdm import tqdm

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


def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction


def main(args):
    config = load_config(args.config, args.agent)
    # setattr(config, 'learning_rate_decay_step', mnist.train.num_examples // config.batch_size)

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
    ap = argparse.ArgumentParser(description='Train a Recurrent Model for Visual Attention')
    ap.add_argument('--config', type=str, default='configs/breakout_agent.cfg', help='Path to configuration file.')
    ap.add_argument('--agent', type=str, default='dqn', help='The name of the agent to run.')
    ap.add_argument('--is_train', action='store_false', help='If specified, the agent performs.')
    ap.add_argument('--env_type', type=str, default='detail', choices=['simple', 'detail'],
                    help='The environment type: `simple` and `detail`.')
    ap.add_argument('--env_name', type=str, default='Breakout-v0', help='The environment name.')
    ap.add_argument('--screen_height', type=int, default=-1, help='The environment screen_height.')
    ap.add_argument('--screen_width', type=int, default=-1, help='The environment screen width.')
    ap.add_argument('--max_reward', type=float, default=1., help='The maximum reward.')
    ap.add_argument('--min_reward', type=float, default=-1, help='The minimum reward.')
    ap.add_argument('--random_start', type=int, default=30, help='The number of random starts.')
    ap.add_argument('--action_repeat', action='store_false', help='Whether to repeat the action or not.')
    ap.add_argument('--gray_scale', action='store_false', help='Whether the screen image is converted to grayscale.')
    ap.add_argument('--display', action='store_true', help='Whether to display the environment or not.')
    ap.add_argument('--gpu_fraction', type=str, default='1/1', help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
    # ap.add_argument('--config', type=str, default='configs/mnist.py', help='Path to configuration file.')
    # ap.add_argument('--batch_size', type=int, default=32, help='Number of examples per batch.')
    # ap.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate at t=0.')
    # ap.add_argument('--learning_rate_decay_factor', type=float, default=0.97,
    #                 help='The factor at which the learning rate decays.')
    # ap.add_argument('--num_epochs_per_decay', type=int, help='Epochs after which learning rate decays.')
    # ap.add_argument('--min_learning_rate', default=1e-4, type=float, help='Minimum learning rate.')
    # # ap.add_argument('--saturate_epoch', type=int,
    # #                 help='Epoch at which linear decayed learning rate will reach minimum learning rate.')
    # ap.add_argument('--max_grad_norm', type=float, help='Maximum gradient normal.')
    # ap.add_argument('--max_epoch', type=int, default=2000, help='Maximum number of epochs to run.')
    # ap.add_argument('--max_tries', type=int, default=100,
    #                 help='Maximum number of epochs to try to find a better local minima for early-stopping.')
    # ap.add_argument('--activation', type=str, default='relu', help='Activation function.')
    # ap.add_argument('--monte_carlo_samples', type=int, default=10, help='Number of samples for monte carlo.')
    # ap.add_argument('--verbosity', type=int, default=1, help='Increase output verbosity.')
    #
    # # glimpse layer
    # ap.add_argument('--glimpse_patch_size', type=int, default=8,
    #                 help='Size of glimpse patch at highest res (height = width).')
    # ap.add_argument('--glimpse_depth', type=int, default=3, help='Number of concatenated downscaled patches.')
    # ap.add_argument('--glimpse_scale', type=int, default=2,
    #                 help='Scale of successive patches w.r.t. original input image.')
    # ap.add_argument('--glimpse_hidden_size', type=int, default=128, help='Size of glimpse hidden layer.')
    # ap.add_argument('--locator_hidden_size', type=int, default=128, help='Size of locator hidden layer.')
    # ap.add_argument('--image_hidden_size', type=int, default=256,
    #                 help='Size of hidden layer combining glimpse and locator hidden layers.')
    #
    # # reinforce
    # ap.add_argument('--reward_scale', type=int, default=1, help='Scale of positive reward (negative is 0).')
    # ap.add_argument('--unit_pixels', type=int, default=13,
    #                 help='The locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13).')
    # ap.add_argument('--locator_std', type=float, default=0.11,
    #                 help='Stddev of gaussian location sampler (between 0 and 1) (low values may cause NaNs).')
    # ap.add_argument('--stochastic', action='store_true',
    #                 help='Reinforce modules forward inputs stochastically during evaluation.')
    #
    # # recurrent layer
    # ap.add_argument('--rho', type=int, default=7, help='Back-propagate through time (BPTT) for rho time-steps.')
    # ap.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units used in Simple RNN.')
    # ap.add_argument('--use_lstm', action='store_true', help='Use LSTM instead of linear layer.')
    #
    # # data
    # ap.add_argument('--dataset', type=str, default='Mnist',
    #                 help='Which dataset to use : Mnist | TranslatedMnist | etc.')
    # ap.add_argument('--train_epoch_size', type=int, default=-1,
    #                 help='Number of train examples seen between each epoch.')
    # ap.add_argument('--valid_epoch_size', type=int, default=-1,
    #                 help='Number of valid examples used for early stopping and cross-validation.')

    try:
        main(ap.parse_args())
    except Exception as e:
        traceback.print_exc()
