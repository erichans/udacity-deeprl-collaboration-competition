import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        
        self.l1 = nn.Linear(observation_size, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_size)
        
    def forward(self, observation):
        x = F.relu(self.l1(observation))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))
        
class TwinCritic(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        
        #Q1 architecture
        self.l1 = nn.Linear(observation_size + action_size, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(observation_size + action_size, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        
    def forward(self, observation_action):
        return self.Q1(observation_action), self.Q2(observation_action)
     
    def Q1(self, observation_action):
        q1 = F.relu(self.l1(observation_action))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)
        
    def Q2(self, observation_action):
        q2 = F.relu(self.l4(observation_action))
        q2 = F.relu(self.l5(q2))
        return self.l6(q2)
    