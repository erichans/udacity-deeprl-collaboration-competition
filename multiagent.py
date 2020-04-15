import torch
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np

from agent import TD3Agent

EXPLORATION_NOISE = 0.1
WARMUP_TIMESTEPS = 1024 #at least the size of the batch
UPDATES_PER_STEP = 4

class MultiAgentTD3:
    def __init__(self, total_agents, state_size, action_size, writer):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.total_agents = total_agents
        self.action_size = action_size
        self.agents = [TD3Agent(agent_number, state_size, action_size, self.device, writer) for agent_number in range(total_agents)]
        print('Device used: {}'.format(self.device)) 
        self.writer = writer
        self.t_step = 0
    
    def act(self, observations_full):
        actions = []
        if self.t_step == WARMUP_TIMESTEPS:
            print('\n\n *** WARMUP ENDED - Starting Training! *** \n')
            
        if self.t_step < WARMUP_TIMESTEPS:
            actions = np.random.normal(size=(self.total_agents,self.action_size)).clip(-1, 1)
        else:
            for agent, observation in zip(self.agents, observations_full):
                noise = np.random.normal(0, EXPLORATION_NOISE, size=self.action_size)
                action = agent.act(torch.from_numpy(observation).float().to(self.device)) + noise
                actions.append(action.clip(-1.0, 1.0))
        
        return actions
        
    def step(self, observations, actions, rewards, next_observations, dones):
        self.t_step += 1
        for agent, observation, action, reward, next_observation, done in zip(self.agents, observations, actions, rewards, next_observations, dones):
            agent.experience(observation, action, reward, next_observation, done)
            
        self._learn()
        
        
    def _learn(self):
        if self.t_step < WARMUP_TIMESTEPS:
            return
        
        for agent in self.agents:
            for _ in range(UPDATES_PER_STEP):
                agent.learn()
        