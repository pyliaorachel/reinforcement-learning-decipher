import logging
import argparse

import numpy as np
import gym

from .dqn import DQN


def parse_args():
    parser = argparse.ArgumentParser(description='Reinforcement learning to decipher')
    parser.add_argument('--cipher-type', metavar='CT', type=str, default='Caesar',
                    help='Cipher type')
    parser.add_argument('-v', metavar='V', type=int, default=0,
                    help='Version of environment')
    parser.add_argument('--no-hint', dest='use_hint', default=True, action='store_false',
                    help='Hint mode or not')
    parser.add_argument('--n-states', metavar='N', type=int, default=10,
                    help='number of dimension in encoded hidden state') 
    parser.add_argument('--lr', metavar='LR', type=float, default=0.01,
                    help='Q-learning rate')
    parser.add_argument('--gamma', metavar='G', type=float, default=0.9,
                    help='Reward discount')
    parser.add_argument('--epsilon', metavar='E', type=float, default=0.8,
                    help='Epsilon under epsilon greedy policy (probability to act greedily)')
    parser.add_argument('--batch-size', metavar='BS', type=int, default=32,
                    help='Batch size')
    parser.add_argument('--memory-capacity', metavar='M', type=int, default=500,
                    help='Replay memory capacity')
    parser.add_argument('--target-replace-iter', metavar='N', type=int, default=50,
                    help='Number of learning iterations before updating target Q-network')
    parser.add_argument('--n-episode', metavar='N', type=int, default=2000,
                    help='number of episodes')
    parser.add_argument('--save-interval', metavar='N', type=int, default=100,
                    help='interal (number of episodes) for saving model')
    parser.add_argument('--output-model', metavar='F', type=str, default='model.bin',
                    help='Output model file path')
    parser.add_argument('--input-model', metavar='F', type=str, default=None,
                    help='Input model file path; keep training or for evaluation')
    parser.add_argument('--log-file', metavar='F', type=str, default='log.txt',
                    help='Log file name')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='Evaluation mode')

    return parser.parse_args()

def eval(env, args):
    if args.input_model is None:
        logging.error('No model provided for evaluation.')
        return

    env_a_shape = [space.n for space in env.action_space.spaces]
    env_s_shape = [space.n for space in env.observation_space.spaces]
    n_actions = sum(env_a_shape)
    n_states = args.n_states

    dqn = DQN(env.base, n_states, n_actions, env_s_shape, env_a_shape, args.use_hint, args.lr, args.gamma, args.epsilon, args.batch_size, args.memory_capacity, args.target_replace_iter)
    dqn.load_state_dict(args.input_model)
    dqn.eval()

    cnt_success = 0
    for i_episode in range(args.n_episode):
        s = env.reset()
        timestep = 0
        ep_r = 0
        acc_loss = 0
        while True:
            env.render()

            a = dqn.choose_action(s)
            next_s, r, done, info = env.step(a)
            ep_r += r

            finished = info['finished']
            if done:
                env.render()
                logging.info(f'Ep: {i_episode}\tRewards: {round(ep_r, 2)}\tSuccess: {finished}')
                cnt_success += 1 if finished else 0
                break

            s = next_s
            timestep += 1
    logging.info('Success rate: {}'.format(round(cnt_success / args.n_episode, 2)))

def run(env, args):
    env_a_shape = [space.n for space in env.action_space.spaces]    
    env_s_shape = [space.n for space in env.observation_space.spaces]    
    n_actions = sum(env_a_shape)
    n_states = args.n_states 

    dqn = DQN(env.base, n_states, n_actions, env_s_shape, env_a_shape, args.use_hint, args.lr, args.gamma, args.epsilon, args.batch_size, args.memory_capacity, args.target_replace_iter)
    if args.input_model is not None:
        dqn.load_state_dict(args.input_model)

    for i_episode in range(args.n_episode):
        s = env.reset()
        timestep = 1
        ep_r = 0
        acc_loss = 0
        while True:
            env.render()

            # Choose an action based on policy
            a = dqn.choose_action(s)

            # Take action
            next_s, r, done, info = env.step(a)

            # Keep transition in memory
            dqn.store_transition(s, a, r, next_s, done)

            # Accumulate reward
            ep_r += r

            # Internally, only learn after having enough experience
            loss = dqn.learn()
            acc_loss += 0 if loss is None else loss

            finished = info['finished']
            if done:
                env.render()
                logging.info(f'Ep: {i_episode}\tRewards: {round(ep_r, 2)}\tLoss: {"None" if loss is None else round(loss, 2)}\tAverage loss: {round(acc_loss/timestep, 2)}\tSuccess: {finished}')
                break

            s = next_s
            timestep += 1

        if i_episode % args.save_interval == 0:
            dqn.save_state_dict(args.output_model)

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(format='%(levelname)s [%(asctime)s] %(message)s', level=logging.INFO, filename=args.log_file)

    cipher_type = args.cipher_type
    version = args.v
    use_hint = args.use_hint
    env = gym.make('{}{}Cipher-v{}'.format('Hint' if use_hint else '', cipher_type, version))
    env = env.unwrapped
    
    if args.eval:
        eval(env, args)
    else:
        run(env, args)

    env.close()
