import numpy as np
import torch

EPSILON = 1e-5

class PrioritizedReplayBuffer:
    """Implementation based on Deep Reinforcement Learning Hands on book - https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/05_dqn_prio_replay.py"""
    def __init__(self, buf_size, device, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)
        self.device = device
        self.beta=0.4
        #self.beta_increase = (1.0 - self.beta) / 50000

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0
        sample = (self.to_torch(state), self.to_torch(action), self.to_torch(reward), self.to_torch(next_state), self.to_torch(done))
        
        if len(self.memory) < self.capacity:
            self.memory.append(sample)
        else:
            self.memory[self.pos] = sample
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False) # no replace to increase diversity
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        #self.beta = min(self.beta + self.beta_increase, 1) # not increasing as it results in poor performance
        return samples, indices, self.to_torch(np.array(weights, dtype=np.float32))

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities + EPSILON):
            self.priorities[idx] = prio
            
    def decode_samples(self, samples):
        observations, actions, rewards, next_observations, dones = [], [], [], [], []
        
        for sample in samples:
            observation, action, reward, next_observation, done = sample
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            dones.append(done)
                    
        return torch.stack(observations), torch.stack(actions), torch.stack(rewards), torch.stack(next_observations), torch.stack(dones)
        
        
    def to_torch(self, element):
        return torch.tensor(element).float().to(self.device)