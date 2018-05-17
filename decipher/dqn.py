import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(n_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)

        return actions_value

class DQN(object):
    def __init__(self, n_states, n_actions, env_a_shape, lr=0.01, gamma=0.9, epsilon=0.9, batch_size=32, memory_capacity=2000, target_replace_iter=100):
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.n_states = n_states
        self.n_actions = n_actions
        self.env_a_shape = env_a_shape # Action shape may be a discrete value or box-like

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
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # [state, action, reward, next_state]
        self.memory_counter = 0

    def choose_action(self, s):
        s = Variable(torch.unsqueeze(torch.FloatTensor(s), 0))

        # epsilon-greedy policy
        if np.random.uniform() < self.epsilon: # Choose action greedily
            actions_value = self.eval_net.forward(s)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else: # Choose action randomly
            action = np.random.randint(0, self.n_actions)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)

        return action

    def store_transition(self, s, a, r, next_s):
        transition = np.hstack((s, [a, r], next_s))

        # Memory is a circular buffer
        i = self.memory_counter % self.memory_capacity 
        self.memory[i, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Only learn after having enough experience
        if self.memory_counter < self.memory_capacity:
            return

        # Update target Q-network every several iterations (replace with eval Q-network's parameters)
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Sample batch transitions from memory
        n_states = self.n_states

        sample_idxs = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_idxs, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :n_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, n_states:n_states+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, n_states+1:n_states+2]))
        b_next_s = Variable(torch.FloatTensor(b_memory[:, -n_states:]))

        # Update eval Q-network from loss against target Q-network
        q_eval = self.eval_net(b_s).gather(1, b_a)                              # Shape (batch, 1)
        q_next = self.target_net(b_next_s).detach()                             # Detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # Shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
