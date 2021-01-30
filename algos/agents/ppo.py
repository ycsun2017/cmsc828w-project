import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import ActorCritic, ContActorCritic
from .updates import ppo_update
from torch.distributions import Categorical
from .gaussian_model import hard_update


class PPO(nn.Module):
    def __init__(self, state_space, action_space, K_epochs=4, eps_clip=0.2, hidden_sizes=(64, 64),
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", action_std=0.5):
        super(PPO, self).__init__()

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        # deal with 1d state input
        self.state_dim = state_space.shape[0]
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.action_std = action_std
        self.activation = activation
        self.lr = learning_rate

        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.policy = ActorCritic(self.state_dim, self.action_dim, hidden_sizes, activation).to(self.device)

        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
            self.policy = ContActorCritic(self.state_dim, self.action_dim, hidden_sizes, activation, action_std,
                                          self.device).to(self.device)

        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        return self.policy.act(state, self.device)

    def update_policy(self, memory):

        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        ppo_update(self.policy, self.optimizer, old_logprobs, memory.rewards,
                   memory, self.gamma, self.K_epochs, self.eps_clip, self.loss_fn, self.device)

    def update_policy_m(self, memory):

        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()

        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_reward.insert(0, Gt)

        discounted_reward = torch.tensor(discounted_reward).to(self.device)
        # Optimize self.policy for 1 epochs:
        # Evaluating old actions and values :
        new_logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(new_logprobs - old_logprobs.detach())

        # Finding Surrogate Loss:
        advantages = discounted_reward - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        loss = -torch.min(surr1, surr2) + 0.5 * self.loss_fn(state_values.float(),
                                                             discounted_reward.float()) - 0.01 * dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()

        if isinstance(self.action_space, Discrete):
            policy_m = ActorCritic(self.state_dim, self.action_dim, self.hidden_sizes, self.activation, with_clone=True,
                                   prior=self.policy).to(self.device)

        elif isinstance(self.action_space, Box):
            policy_m = ContActorCritic(self.state_dim, self.action_dim, self.hidden_sizes, self.activation,
                                       self.action_std, self.device, with_clone=True, prior=self.policy).to(self.device)

        return policy_m

    def get_state_dict(self):
        return self.policy.state_dict(), self.optimizer.state_dict()

    def set_state_dict(self, state_dict, optim):
        self.policy.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optim)

    def set_params(self, sample_policy):
        for layer, sample_layer in zip(self.policy.action_layer, sample_policy.action_layer):
            # print(type(layer), type(sample_layer))
            if type(layer) == nn.Linear:
                # print("layer.weight", layer.weight)
                # print("sample_layer.weight", sample_layer.weight)
                hard_update(layer.weight, sample_layer.weight)
                hard_update(layer.bias, sample_layer.bias)
                # print("layer.weight", layer.weight)
                # print("sample_layer.weight", sample_layer.weight)
