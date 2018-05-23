import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from .utils import one_hot_hstack, split_chunks, pad_zeros, to_idx_list


class StateEncoder(nn.Module):
    def __init__(self, base, n_states, use_hint=False):
        super().__init__()
        self.one_hot_sizes = [base] if not use_hint else [base, base+1]
        self.input_dim = base if not use_hint else base * 2 + 1 # base for cipher char, (base + 1) for hint char with '_'
        self.hidden_dim = n_states
        self.output_dim = base

        self.one_hot = lambda x: np.apply_along_axis(lambda a: one_hot_hstack(a, self.one_hot_sizes), 2, x)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        x = Variable(torch.Tensor(self.one_hot(x)))
        lstm_out, (h, c) = self.lstm(x)

        return lstm_out 

class QNet(nn.Module):
    def __init__(self, base, n_states, n_actions, use_hint=False):
        super().__init__()
        self.state_encoder = StateEncoder(base, n_states, use_hint=use_hint)
        
        self.n_states = n_states
        self.n_actions = n_actions

        self.out = nn.Linear(n_states, n_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.state_encoder(x)

        x = F.relu(x)
        actions_value = self.out(x)

        return actions_value

    def get_hidden_state(self, x):
        return self.state_encoder(x).detach()[0].data.numpy()

class DQN(object):
    def __init__(self, base, n_states, n_actions, env_s_shape, env_a_shape, use_hint=False, lr=0.01, gamma=0.9, epsilon=0.8, batch_size=32, memory_capacity=500, target_replace_iter=50):
        self.eval_net, self.target_net = QNet(base, n_states, n_actions, use_hint=use_hint), QNet(base, n_states, n_actions, use_hint=use_hint)

        self.base = base
        self.n_states = n_states
        self.n_actions = n_actions
        self.env_s_shape = env_s_shape
        self.env_a_shape = env_a_shape # Shapes may be a discrete value or box-like

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter
        self.learn_step_counter = 0 # For target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        # Experience replay memory 
        self.memory = np.zeros((memory_capacity, len(env_s_shape) * 2 + len(env_a_shape) + 1)) # [state, action, reward, next_state]
        self.memory_counter = 0
        self.memory_ep_end_flags = [] # Memories of each episode is kept as one single replay memory

        # Evaluation mode
        self.eval_mode = False

    def choose_action(self, s):
        s = [[s]] # Unsqueeze timestep, batch_size

        # epsilon-greedy policy
        if not self.eval_mode or np.random.uniform() < self.epsilon: # Choose action greedily
            actions_value = self.eval_net(s).detach()[0][0]
            action = tuple([np.argmax(subact_v) for subact_v in split_chunks(actions_value.data.numpy(), self.env_a_shape)])
        else: # Choose action randomly
            action = tuple([np.random.randint(0, a) for a in self.env_a_shape])

        return action

    def store_transition(self, s, a, r, next_s, done):
        a_idx = to_idx_list(a, self.env_a_shape) # Turn action into indices for ease of use in learning
        transition = np.hstack((s, a_idx, [r], next_s))

        # Memory is a circular buffer
        i = self.memory_counter % self.memory_capacity 
        self.memory[i, :] = transition
        self.memory_counter += 1

        # Mark end of episode flag
        if done:
            if len(self.memory_ep_end_flags) > 0 and i <= self.memory_ep_end_flags[-1]: # Memory overflow
                l = self.memory_ep_end_flags
                old_mem = next(x[0] for x in enumerate(l) if x[1] > i) # Find first elem larger than i
                self.memory_ep_end_flags = self.memory_ep_end_flags[old_mem:]
            self.memory_ep_end_flags.append(i)

    def learn(self):
        # Only learn after having enough experience
        if self.memory_counter < self.memory_capacity:
            return

        # Update target Q-network every several iterations (replace with eval Q-network's parameters)
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Sample batch transitions from memory
        l_states = len(self.env_s_shape) 
        l_actions = len(self.env_a_shape)
        m = self.memory
        flags = self.memory_ep_end_flags
        a_shape = self.env_a_shape

        ## Random select series of memories, pad them to equal sequence length
        sample_idxs = np.random.choice(range(len(flags)-1), self.batch_size)

        memory_ranges = zip(flags[:-1], flags[1:])
        memory_series = [m[s:e] if s <= e else (np.concatenate((m[s:], m[:e]))) for s, e in memory_ranges]
        seq_len = max([len(ms) for ms in memory_series])
        memory_series = pad_zeros(memory_series, seq_len) # Pad 0s to make the seq len the same
        b_memory = memory_series[sample_idxs]

        ## Sample batch memory series
        b_s = b_memory[:, :, :l_states]
        b_a = Variable(torch.LongTensor(b_memory[:, :, l_states:l_states+l_actions].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, :, l_states+l_actions:l_states+l_actions+1]))
        b_next_s = b_memory[:, :, -l_states:]

        # Update eval Q-network from loss against target Q-network
        q_eval = self.eval_net(b_s).gather(2, b_a)                              # Shape (batch, seq_len, l_actions)
        q_next = self.target_net(b_next_s).detach()                             # Detach from graph, don't backpropagate
        argmax_q_next = np.apply_along_axis(lambda x: to_idx_list([np.argmax(subact_v) for subact_v in split_chunks(x, a_shape)], a_shape), \
                                            2, q_next.data.numpy())
        max_q_next = q_next.gather(2, Variable(torch.LongTensor(argmax_q_next)))
        q_target = b_r + self.gamma * max_q_next 
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval(self):
        self.eval_mode = True

    def save_state_dict(self, file_path):
        model = { 'eval_net': self.eval_net.state_dict(), 'target_net': self.target_net.state_dict() }
        with open(file_path, 'wb') as fout:
            pickle.dump(model, fout)

    def load_state_dict(self, file_path):
        with open(file_path, 'rb') as fout:
            model = pickle.load(fout)
        self.eval_net.load_state_dict(model['eval_net'])
        self.target_net.load_state_dict(model['target_net'])
