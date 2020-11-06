import sys
import torch  
import math
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import Actor, ContActor
from .gaussian_model import PolicyHub
from .updates import vpg_update
from torch.distributions import Categorical

class GaussianVPG(nn.Module):
    """
    A meta policy (maintaining a gaussian distribution of policies)
    """
    def __init__(self, state_space, action_space, sample_size, hidden_sizes=(4,4), 
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", 
                 action_std=0.5, delta=0.1, coeff=1.0, tau=0.5):
        super(GaussianVPG, self).__init__()
        
        # deal with 1d state input
        state_dim = state_space.shape[0]
        
        self.gamma = gamma
        self.device = device
        self.learning_rate = learning_rate
        self.coeff = coeff
        self.N = sample_size
        self.delta = delta
        self.log_term = math.log(2*math.sqrt(self.N)/self.delta)
        self.update_prior_every = 10
        self.update = 0
        self.cont_action = False
        self.action_std = action_std
        self.device = device

        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.policy_hub = PolicyHub(state_dim, self.action_dim, hidden_sizes, activation, tau, self.device)
            
        elif isinstance(action_space, Box):
            self.cont_action = True
            self.action_dim = action_space.shape[0]
            self.policy_hub = PolicyHub(state_dim, self.action_dim, hidden_sizes, activation, tau, self.device)
        # print(self.policy_hub.get_parameters())
        self.meta_optimizer = optim.SGD(self.policy_hub.get_parameters(), lr=self.learning_rate)

        

    def sample_policy(self):
        if self.cont_action:
            self.cur_policy = self.policy_hub.sample_cont_policy(self.action_std, self.device)
        else:
            self.cur_policy = self.policy_hub.sample_policy(self.device)
        # print(self.cur_policy.get_parameters())
        # self.cur_optimizer = optim.Adam(self.cur_policy.get_parameters(), lr=self.learning_rate)
        return self.cur_policy
    
    
    # def policy_update(self, memory):
    #     print("policy update", memory.logprobs)
    #     discounted_reward = []
    #     Gt = 0
    #     for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
    #         if is_terminal:
    #             Gt = 0
    #         Gt = reward + (self.gamma * Gt)
    #         discounted_reward.insert(0, Gt)
    #     policy_gradient = []
    #     for log_prob, Gt in zip(memory.logprobs, discounted_reward):
    #         policy_gradient.append(-log_prob * Gt)
        
    #     loss = torch.stack(policy_gradient).sum()
    #     loss.backward()
    #     print("gradient of policy sample network")
    #     for l in self.cur_policy.get_parameters():
    #         print(l.grad)
        
    def meta_update(self, memory):
        print("meta update", len(memory.rewards))

        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_reward.insert(0, Gt)
        
        # Normalizing the rewards:
        # discounted_reward = torch.tensor(discounted_reward).to(self.device)
        # discounted_reward = (discounted_reward - discounted_reward.mean()) / (discounted_reward.std() + 1e-5)
        # print("old param", self.policy_hub.get_parameters())
        policy_gradient = []
        for log_prob, Gt in zip(memory.logprobs, discounted_reward):
            policy_gradient.append(-log_prob * Gt)
        # print("policy gradient", len(policy_gradient))
        
        # regularizer
        # regularize_loss = []
        # ori_params = self.policy_hub.get_parameters().detach().clone()
        # cur_params = self.policy_hub.get_parameters()
        # for p, c in zip(ori_params, cur_params):
        #     regularize_loss.append(torch.sum(p-c))
        # print(regularize_loss)
        regularize_loss = self.policy_hub.regularize_loss()
        print("reg loss", regularize_loss)
        regularize_loss = torch.sqrt((regularize_loss+self.log_term)/(2*self.N))
        print("reg loss", regularize_loss)
        self.meta_optimizer.zero_grad()
        loss = self.coeff * torch.stack(policy_gradient).sum() + regularize_loss
        print("loss", loss)
        loss.backward()
        # for param in self.policy_hub.get_parameters():
        #     print("grad", param.grad)
        self.meta_optimizer.step()
        # print("new param", self.policy_hub.get_parameters())
        # print("new mu", self.policy_hub.gaussian_policy_layers[0].weight.mu)

        # self.update += 1
        # if self.update % self.update_prior_every:
        self.policy_hub.update_prior()
    
    def get_state_dict(self):
        return self.policy.state_dict(), self.optimizer.state_dict()
    
    
    def set_state_dict(self, state_dict, optim):
        self.policy.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optim)
        
        