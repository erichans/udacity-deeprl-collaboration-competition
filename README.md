# udacity-deeprl-collaboration-competition
Collaboration and Competition problem using Reinforcement Learning

# Project Details

![](/images/unity-wide.png)

## Unity ML-Agents

**Unity Machine Learning Agents (ML-Agents)** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

For game developers, these trained agents can be used for multiple purposes, including controlling [NPC](https://en.wikipedia.org/wiki/Non-player_character) behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

In this course, you will use Unity's rich environments to design, train, and evaluate your own deep reinforcement learning algorithms. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

## The Environment

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.


<p align="center">
  <img src="/images/tennis.gif" />
</p>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Getting Started

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Windows__:
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6 
	source activate drlnd
	```

2. Install pytorch >= 1.4.0

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/erichans/udacity-deeprl-collaboration-competition.git
cd udacity-deeprl-collaboration-competition/python
pip install .
```

# Instructions

## Train the Agent
```bash
python train.py
```
You can tune the model by changing the following hyperparameters in following files (default values below):

### train.py
* TOTAL_EPISODES = 500
* SEED = 0

### multiagent.py
* EXPLORATION_NOISE = 0.1 (Noise Scale added to each predicted action from the agents)
* WARMUP_TIMESTEPS = 1000 or 1024 (Number of steps before training starts. For faster convergence and more stable results after solving: __Linux__: 1000. __Windows 10__: 1024 )
* UPDATES_PER_STEP = 4 (Number of updates of the neural networks per timestep)

### agent.py
* BUFFER_SIZE = 1.000.000
* BATCH_SIZE = 1000 or 1024 (for faster convergence and more stable results after solving: __Linux__: 1000. __Windows 10__: 1024 )
* GAMMA = .99 (discount factor)
* TAU = 5e-2 (soft update from local actor and critic network parameters to their respective target network parameters)
* LR_ACTOR = 1e-3 (Actor local learning rate)
* LR_CRITIC = 1e-3 (Critic local learning rate)
* POLICY_UPDATE_FREQUENCY = 2 (Policies are updated __UPDATES_PER_STEP/POLICY_UPDATE_FREQUENCY__ times for each timestep and the Twin Critics are updated are updated __UPDATES_PER_STEP__ times for each timestep)
* POLICY_NOISE = 0.2 (Noise Scale added to the learning process of the agents)
* NOISE_CLIP = 0.5 (Noise clipping to the POLICY_NOISE)

### buffer.py
* prob_alpha = 0.6 (Alpha determines how much prioritization is used)
* beta = 0.4 (Importance-sampling correction exponent)
* EPSILON = 1e-5 (Small positive constant that prevents transitions not being revisited once their error is zero in the Prioritized Experience Replay)

## Report
To see more details like:
* Learning Algorithm 
* Plot of Rewards
* Ideas for Future Work

Check the [Report](/Report.md)
