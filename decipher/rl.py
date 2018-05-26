import logging
import argparse

import numpy as np
import gym

from .dqn import DQN
from .utils import symbol_repr_total_size 


def parse_args():
    parser = argparse.ArgumentParser(description='Reinforcement learning to decipher')
    parser.add_argument('--cipher-type', metavar='CT', type=str, default='Caesar',
                        help='Cipher type (default: Caesar)')
    parser.add_argument('-v', metavar='V', type=int, default=0,
                        help='Version of environment (default: 0)')
    parser.add_argument('--no-hint', dest='use_hint', default=True, action='store_false',
                        help='Hint mode or not (default: True)')
    parser.add_argument('--n-states', metavar='N', type=int, default=10,
                        help='number of dimension in encoded hidden state (default: 10)') 
    parser.add_argument('--symbol-repr-method', metavar='M', type=str, default='one_hot',
                        help='Symbol representation method; one of one_hot, ordinal_vec, ordinal_num (default: one_hot)')
    parser.add_argument('--lr', metavar='LR', type=float, default=0.01,
                        help='Q-learning rate (default:  0.01)')
    parser.add_argument('--gamma', metavar='G', type=float, default=0.9,
                        help='Reward discount (default: 0.9)')
    parser.add_argument('--epsilon', metavar='E', type=float, default=0.8,
                        help='Epsilon under epsilon greedy policy (probability to act greedily) (default: 0.8)')
    parser.add_argument('--batch-size', metavar='BS', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--memory-capacity', metavar='M', type=int, default=500,
                        help='Replay memory capacity (default: 500)')
    parser.add_argument('--target-replace-iter', metavar='N', type=int, default=50,
                        help='Number of learning iterations before updating target Q-network (default: 50)')
    parser.add_argument('--hidden-dim', metavar='N', type=int, default=None,
                        help='Hidden layer dimension within Q-network (default: None, i.e. no hidden layer)')
    parser.add_argument('--n-episode', metavar='N', type=int, default=2000,
                        help='number of episodes (default: 2000')
    parser.add_argument('--start-episode', metavar='N', type=int, default=0,
                        help='starting episode number (for continued training) (default: 0)')
    parser.add_argument('--save-interval', metavar='N', type=int, default=100,
                        help='interal (number of episodes) for saving model (default: 100)')
    parser.add_argument('--output-model', metavar='F', type=str, default='model.bin',
                        help='Output model file path (default: model.bin)')
    parser.add_argument('--input-model', metavar='F', type=str, default=None,
                        help='Input model file path; keep training or for evaluation (default: None)')
    parser.add_argument('--log-file', metavar='F', type=str, default='log.txt',
                        help='Log file name (default: log.txt)')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                        help='Evaluation mode (default: False)')

    return parser.parse_args()

def get_env_nn_shapes(env, args):
    env_a_shape = [space.n for space in env.action_space.spaces]
    env_s_shape = [space.n for space in env.observation_space.spaces]
    n_actions = sum(env_a_shape)
    n_states = args.n_states

    return env_a_shape, env_s_shape, n_actions, n_states

def eval(env, args):
    if args.input_model is None:
        logging.error('No model provided for evaluation.')
        return

    env_a_shape, env_s_shape, n_actions, n_states = get_env_nn_shapes(env, args) 

    dqn = DQN(env.base, n_states, n_actions, env_s_shape, env_a_shape, args.symbol_repr_method, args.lr, args.gamma, args.epsilon, args.batch_size, args.memory_capacity, args.target_replace_iter, args.hidden_dim)
    dqn.load_state_dict(args.input_model)
    dqn.eval()

    cnt_success = 0
    for i_episode in range(args.start_episode, args.start_episode + args.n_episode):
        s = env.reset()
        timestep = 1
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
    env_a_shape, env_s_shape, n_actions, n_states = get_env_nn_shapes(env, args) 

    dqn = DQN(env.base, n_states, n_actions, env_s_shape, env_a_shape, args.symbol_repr_method, args.lr, args.gamma, args.epsilon, args.batch_size, args.memory_capacity, args.target_replace_iter, args.hidden_dim)
    if args.input_model is not None: # Load pretrained model if provided
        dqn.load_state_dict(args.input_model)

    for i_episode in range(args.start_episode, args.start_episode + args.n_episode):
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
                logging.info(f'Ep: {i_episode}\tRewards: {round(ep_r, 2)}\tLoss: {"None" if loss is None else round(loss, 2)}\tAverage loss: {round(acc_loss/timestep, 2)}\tLength: {env.target_width}\tSuccess: {finished}')
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
    env = gym.make(f'{"Hint" if use_hint else ""}{cipher_type}Cipher-v{version}')
    env = env.unwrapped
    
    if args.eval:
        eval(env, args)
    else:
        run(env, args)

    env.close()
