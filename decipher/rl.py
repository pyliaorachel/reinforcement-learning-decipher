import logging
import argparse

import numpy as np
import gym

from .dqn import DQN


def parse_args():
    parser = argparse.ArgumentParser(description='Reinforcement learning to decipher')
    parser.add_argument('--lr', metavar='LR', type=float, default=0.01,
                    help='Learning rate')
    parser.add_argument('--gamma', metavar='G', type=float, default=0.9,
                    help='Reward discount')
    parser.add_argument('--epsilon', metavar='E', type=float, default=0.9,
                    help='Epsilon under epsilon greedy policy (probability to act greedily)')
    parser.add_argument('--batch-size', metavar='BS', type=int, default=32,
                    help='Batch size')
    parser.add_argument('--memory-capacity', metavar='M', type=int, default=2000,
                    help='Replay memory capacity')
    parser.add_argument('--target-replace-iter', metavar='N', type=int, default=100,
                    help='Number of learning iterations before updating target Q-network')
    parser.add_argument('--n-episode', metavar='N', type=int, default=400,
                    help='Number of episodes') 

    return parser.parse_args()

def run(env, args):
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    env_a_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

    dqn = DQN(n_states, n_actions, env_a_shape, args.lr, args.gamma, args.epsilon, args.batch_size, args.memory_capacity, args.target_replace_iter)

    for i_episode in range(args.n_episode):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()

            # Choose an action based on policy
            a = dqn.choose_action(s)

            # Take action
            next_s, r, done, info = env.step(a)

            # Modify the reward
            x, x_dot, theta, theta_dot = next_s
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            # Keep transition in memory
            dqn.store_transition(s, a, r, next_s)

            # Accumulate reward
            ep_r += r

            # Internally, only learn after having enough experience
            dqn.learn()

            if done:
                logging.info('Ep: {}\tRewards: {}'.format(i_episode, round(ep_r, 2)))
                break

            s = next_s

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s [%(asctime)s] %(message)s', level=logging.INFO)
    args = parse_args()

    env = gym.make('CaesarCipher-v0')
    env = env.unwrapped
    
    run(env, args)

    env.close()
