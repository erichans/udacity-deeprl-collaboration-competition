# Learning Algorithm
The Reinforcement Learning algorithm used to **solve** the environment was [Twin Delayed Deep Deterministic policy gradient (TD3)](https://arxiv.org/abs/1802.09477)

![](images/td3-algorithm.PNG)

To achieve better results, the algorithm was customized to address the problem and introduced the use of [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952)

![](images/per.png)

* The Prioritized Experience Replay (PER) is inside the DQN algorithm as in the original paper. The PER algorithm parts are in __Green__ and __Yellow__.
* The __Green__ part was used as is without needing any customization.
* The __Yellow__ part was applied only to the errors of the Twin Critics and not toe the Policy errors because it worsen its results.

## Model
```
Agent 0 - Actor Local TD3 -> Actor(
  (l1): Linear(in_features=24, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=2, bias=True)
)
Agent 0 - Actor Target TD3 -> Actor(
  (l1): Linear(in_features=24, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=2, bias=True)
)
Agent 0 - Twin Critic Local TD3 -> TwinCritic(
  (l1): Linear(in_features=26, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=1, bias=True)
  (l4): Linear(in_features=26, out_features=400, bias=True)
  (l5): Linear(in_features=400, out_features=300, bias=True)
  (l6): Linear(in_features=300, out_features=1, bias=True)
)
Agent 0 - Twin Critic Target TD3 -> TwinCritic(
  (l1): Linear(in_features=26, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=1, bias=True)
  (l4): Linear(in_features=26, out_features=400, bias=True)
  (l5): Linear(in_features=400, out_features=300, bias=True)
  (l6): Linear(in_features=300, out_features=1, bias=True)
)
Agent 1 - Actor Local TD3 -> Actor(
  (l1): Linear(in_features=24, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=2, bias=True)
)
Agent 1 - Actor Target TD3 -> Actor(
  (l1): Linear(in_features=24, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=2, bias=True)
)
Agent 1 - Twin Critic Local TD3 -> TwinCritic(
  (l1): Linear(in_features=26, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=1, bias=True)
  (l4): Linear(in_features=26, out_features=400, bias=True)
  (l5): Linear(in_features=400, out_features=300, bias=True)
  (l6): Linear(in_features=300, out_features=1, bias=True)
)
Agent 1 - Twin Critic Target TD3 -> TwinCritic(
  (l1): Linear(in_features=26, out_features=400, bias=True)
  (l2): Linear(in_features=400, out_features=300, bias=True)
  (l3): Linear(in_features=300, out_features=1, bias=True)
  (l4): Linear(in_features=26, out_features=400, bias=True)
  (l5): Linear(in_features=400, out_features=300, bias=True)
  (l6): Linear(in_features=300, out_features=1, bias=True)
)
```

## Hyperparameters used for training

* TOTAL_EPISODES: 500

* EXPLORATION_NOISE: 0.1 (Noise Scale added to each predicted action from the agents)
* WARMUP_TIMESTEPS: 1000 or 1024 (Number of steps before training starts. For faster convergence and more stable results after solving: __Ubuntu 16.04__: 1000. __Windows 10__: 1024)
* UPDATES_PER_STEP: 4 (Number of updates of the neural networks per timestep)

* BUFFER_SIZE: 1.000.000
* BATCH_SIZE: 1000 or 1024 (for faster convergence and more stable results after solving: __Ubuntu 16.04__: 1000. __Windows 10__: 1024)
* GAMMA: .99 (discount factor)
* TAU: 5e-2 (soft update from local actor and critic network parameters to their respective target network parameters)
* LR_ACTOR: 1e-3 (Actor local learning rate)
* LR_CRITIC: 1e-3 (Critic local learning rate)
* POLICY_UPDATE_FREQUENCY: 2 (Policies are updated __UPDATES_PER_STEP/POLICY_UPDATE_FREQUENCY__ times for each timestep and the Twin Critics are updated are updated __UPDATES_PER_STEP__ times for each timestep)
* POLICY_NOISE: 0.2 (Noise Scale added to the learning process of the agents)
* NOISE_CLIP: 0.5 (Noise clipping to the POLICY_NOISE)

* prob_alpha: 0.6 (Alpha determines how much prioritization is used)
* beta: 0.4 (Importance-sampling correction exponent)
* EPSILON: 1e-5 (Small positive constant that prevents transitions not being revisited once their error is zero in the Prioritized Experience Replay)


# Plot of Rewards

## Warmup before training (No Learning - Random outputs to explore)
```
Score: 0.00/-0.01 -> Ep. 1/500 - Avg Max Global Score: 0.00
Score: -0.01/0.00 -> Ep. 10/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 20/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 30/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 40/500 - Avg Max Global Score: 0.02
Score: 0.00/0.09 -> Ep. 50/500 - Avg Max Global Score: 0.02
Score: 0.09/0.10 -> Ep. 51/500 - Avg Max Global Score: 0.02
Score: -0.01/0.00 -> Ep. 52/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 53/500 - Avg Max Global Score: 0.02
Score: 0.10/-0.01 -> Ep. 54/500 - Avg Max Global Score: 0.02
```
 
 ## Training
 
```
Score: -0.01/0.10 -> Ep. 55/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 60/500 - Avg Max Global Score: 0.02
Score: -0.01/0.00 -> Ep. 70/500 - Avg Max Global Score: 0.02
Score: -0.01/0.00 -> Ep. 80/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 90/500 - Avg Max Global Score: 0.02
Score: -0.01/0.00 -> Ep. 100/500 - Avg Max Global Score: 0.01
Score: 0.00/-0.01 -> Ep. 110/500 - Avg Max Global Score: 0.01
Score: 0.00/-0.01 -> Ep. 120/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 130/500 - Avg Max Global Score: 0.02
Score: 0.00/-0.01 -> Ep. 140/500 - Avg Max Global Score: 0.03
Score: 0.10/-0.01 -> Ep. 150/500 - Avg Max Global Score: 0.03
Score: 0.00/0.09 -> Ep. 160/500 - Avg Max Global Score: 0.04
Score: 0.10/-0.01 -> Ep. 170/500 - Avg Max Global Score: 0.05
Score: 0.00/0.09 -> Ep. 180/500 - Avg Max Global Score: 0.05
Score: 0.19/0.30 -> Ep. 190/500 - Avg Max Global Score: 0.06
Score: 0.10/0.09 -> Ep. 200/500 - Avg Max Global Score: 0.07
Score: 0.10/0.09 -> Ep. 210/500 - Avg Max Global Score: 0.08
Score: -0.01/0.10 -> Ep. 220/500 - Avg Max Global Score: 0.08
Score: 0.20/0.09 -> Ep. 230/500 - Avg Max Global Score: 0.09
Score: 0.20/0.09 -> Ep. 240/500 - Avg Max Global Score: 0.10
Score: 0.29/0.30 -> Ep. 250/500 - Avg Max Global Score: 0.11
Score: 0.10/-0.01 -> Ep. 260/500 - Avg Max Global Score: 0.11
Score: 0.10/-0.01 -> Ep. 270/500 - Avg Max Global Score: 0.12
Score: 0.69/0.70 -> Ep. 280/500 - Avg Max Global Score: 0.13
Score: 0.10/-0.01 -> Ep. 290/500 - Avg Max Global Score: 0.18
Score: 0.10/-0.01 -> Ep. 300/500 - Avg Max Global Score: 0.22
Score: 2.60/2.60 -> Ep. 310/500 - Avg Max Global Score: 0.29
Score: 2.60/2.60 -> Ep. 320/500 - Avg Max Global Score: 0.47
Score: 2.60/2.60 -> Ep. 323/500 - Avg Max Global Score: 0.50
```
Environment solved (mean of 0.5 for 100 episodes) in **323** episodes!	Average Max Score: **0.50**


## Still Training to see if the Agents keep improving over time

```
Score: -0.01/0.10 -> Ep. 324/500 - Avg Max Global Score: 0.50
Score: 2.60/2.60 -> Ep. 331/500 - Avg Max Global Score: 0.57
Score: 2.60/2.60 -> Ep. 340/500 - Avg Max Global Score: 0.65
Score: 2.60/2.60 -> Ep. 350/500 - Avg Max Global Score: 0.79
Score: 0.00/0.09 -> Ep. 360/500 - Avg Max Global Score: 0.90
Score: 0.09/0.10 -> Ep. 370/500 - Avg Max Global Score: 1.01
Score: 2.60/2.60 -> Ep. 380/500 - Avg Max Global Score: 1.18
Score: 2.60/2.50 -> Ep. 390/500 - Avg Max Global Score: 1.24
Score: 1.00/0.99 -> Ep. 400/500 - Avg Max Global Score: 1.33
Score: 2.70/2.60 -> Ep. 410/500 - Avg Max Global Score: 1.40
Score: 0.19/0.30 -> Ep. 420/500 - Avg Max Global Score: 1.37
Score: 0.10/-0.01 -> Ep. 430/500 - Avg Max Global Score: 1.47
Score: 0.60/0.59 -> Ep. 440/500 - Avg Max Global Score: 1.49
Score: 2.60/2.60 -> Ep. 450/500 - Avg Max Global Score: 1.45
Score: 0.00/-0.01 -> Ep. 460/500 - Avg Max Global Score: 1.46
Score: 2.70/2.60 -> Ep. 470/500 - Avg Max Global Score: 1.57
Score: 0.89/0.90 -> Ep. 480/500 - Avg Max Global Score: 1.60
Score: 2.60/2.70 -> Ep. 490/500 - Avg Max Global Score: 1.74
Score: 2.60/2.60 -> Ep. 500/500 - Avg Max Global Score: 1.86
```

Avergage Max Score achieved: at Episode **500/500**: **1.86**


### Detailed execution logs: [here](/results/results-linux.txt)

The agents kept learning until the end of the training with no sign of reaching the plateau as shown in the graph below:

<h3 align="center">Score evolution</h3>
<p align="center">
  <img src="/images/score-evolution-linux.png" />
</p>


# Ideas for Future Work

1. Hyperparameter tuning:
    * Tuning alpha and beta from [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) to explore the trade off aggressiveness x robustness
    * Find the optimal the Neural Network sizes for faster convergence and improved stability)
2. Implement [Multi Agent DDPG(MADDPG)](https://arxiv.org/abs/1706.02275) algorithm to compare performance with the TD3
3. Implement [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) algorithm to compare performance with the TD3
4. Implement [Alpha Zero in Continuous Action Space](https://arxiv.org/abs/1805.09613) algorithm (the continuous output version of original [Alpha Zero](https://arxiv.org/abs/1712.01815) algorithm) to compare performance with the TD3
5. Implement [Multi Agent TD3(MATD3)](https://arxiv.org/abs/1910.01465) algorithm to compare performance with the TD3
6. Stack a sequence of 4 states for 1 action so the network can learn sequence of events like in [Reward learning from human preferences and
demonstrations in Atari](https://arxiv.org/pdf/1811.06521.pdf) paper
7. Switch the agent to learn from pixels
