import torch
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import copy

from model import Actor, TwinCritic
from buffer import PrioritizedReplayBuffer


BUFFER_SIZE = int(1e6)
BATCH_SIZE = 1024

TAU = 5e-2

LR_ACTOR = 1e-3
LR_TWIN_CRITIC = 1e-3

POLICY_NOISE = 0.2
NOISE_CLIP = 0.5

GAMMA = .99

POLICY_UPDATE_FREQUENCY = 2

class TD3Agent:
    def __init__(self, agent_number, observation_size, action_size, device, writer):
        self.device = device
        
        self.actor = Actor(observation_size, action_size).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.twin_critic = TwinCritic(observation_size, action_size).to(self.device)
        self.twin_critic_target = copy.deepcopy(self.twin_critic)
        self.twin_critic_optimizer = optim.Adam(self.twin_critic.parameters(), lr=LR_TWIN_CRITIC)
        
        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, self.device)
        self.agent_number = agent_number
        self.t_step = 0
        
        print('Agent {} - Actor Local DDPG ->'.format(self.agent_number), self.actor)
        print('Agent {} - Actor Target DDPG ->'.format(self.agent_number), self.actor_target)
        
        print('Agent {} - Twin Critic Local DDPG ->'.format(self.agent_number), self.twin_critic)
        print('Agent {} - Twin Critic Target DDPG ->'.format(self.agent_number), self.twin_critic_target)

        self.writer = writer
        
    def experience(self, observation, action, reward, next_observation, done):
        self.replay_buffer.add(observation, action, reward, next_observation, done)
    
    def act(self, observations):
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(observations).data.cpu().numpy()
        self.actor.train()
        return actions
        
    def square_error(self, predicted, target):
        return (predicted - target) ** 2
     
    def learn(self):
        self.t_step += 1 
        
        samples, indices, weights = self.replay_buffer.sample(BATCH_SIZE)
        observations, actions, rewards, next_observations, dones = self.replay_buffer.decode_samples(samples)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = (self.actor_target(next_observations) + noise).clamp(-1.0, 1.0)
            target_Q1_next, target_Q2_next = self.twin_critic_target(torch.cat([next_observations, next_actions], dim=1))
            target_Q_next = torch.min(target_Q1_next, target_Q2_next).view(-1)
            target_Q = rewards + GAMMA * target_Q_next * (1 - dones)
            
        current_Q1, current_Q2 = self.twin_critic(torch.cat([observations, actions], dim=1))
        twin_critic_loss = weights * (self.square_error(current_Q1.view(-1), target_Q) + self.square_error(current_Q2.view(-1), target_Q))
        self.replay_buffer.update_priorities(indices, twin_critic_loss.data.cpu().numpy())
        twin_critic_loss = twin_critic_loss.mean()
        self.twin_critic_optimizer.zero_grad()
        twin_critic_loss.backward()
        self.twin_critic_optimizer.step()
        
        self.writer.add_scalar('Agent_{}_Critic_Loss'.format(self.agent_number), twin_critic_loss.data.cpu().numpy(), self.t_step)

        if self.t_step % POLICY_UPDATE_FREQUENCY == 0:
            actor_loss = -self.twin_critic.Q1(torch.cat([observations, self.actor(observations)], dim=1)).mean() # not using weights as it results in poor performance o_O
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self._soft_update(self.twin_critic, self.twin_critic_target)
            self._soft_update(self.actor, self.actor_target)
            
            self.writer.add_scalar('Agent_{}_Actor_Loss'.format(self.agent_number), actor_loss.data.cpu().numpy(), self.t_step)
        
    def _soft_update(self, local_model, target_model):
        for local_parameter, target_parameter in zip(local_model.parameters(), target_model.parameters()):
            target_parameter.data.copy_((1.0-TAU)*target_parameter+(TAU*local_parameter))