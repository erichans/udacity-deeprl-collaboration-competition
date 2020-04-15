from multiagent import MultiAgentTD3

from collections import deque

from unityagents import UnityEnvironment

import torch
import numpy as np

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import datetime

def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def start_env():
    env = UnityEnvironment(file_name='Tennis_Windows_x86_64/Tennis.exe')

    # get the default brain
    brain_name = get_brain_name(env)
    brain = get_brain(env)

    env_info = reset_env_info(env)

    print('Number of agents:', get_total_agents(env_info))

    action_size = get_action_size(env)
    print('Number of actions:', action_size)

    state = env_info.vector_observations[0]
    print('States look like:', state)
    print('States have length:', get_state_size(env_info))
    
    return env

def get_brain_name(env):
    return env.brain_names[0]
    
def get_brain(env):
    return env.brains[get_brain_name(env)]
    
def get_state_size(env_info):
    return len(env_info.vector_observations[0])
    
def get_action_size(env):
    return get_brain(env).vector_action_space_size
    
def reset_env_info(env):
    return env.reset(train_mode=True)[get_brain_name(env)]
    
def env_step(env, action):
    return env.step(action)[get_brain_name(env)]

def get_total_agents(env_info):
    return len(env_info.agents)

def multi_agent_td3_run(episodes=1000, seed=42):
    seeding(seed)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter('tensorboard/' + current_time)
        
    env = start_env()
    env_info = reset_env_info(env)
    
    state_size = get_state_size(env_info)
    action_size = get_action_size(env)
    
    print('Seed used:', seed)
    total_agents = get_total_agents(env_info)
    multiagent = MultiAgentTD3(total_agents, state_size, action_size, writer)
    
    scores = []
    scores_window = deque(maxlen=MOVING_AVG_WINDOW)
    solved = False
    for episode in range(1, episodes+1):
        init_time = datetime.datetime.now()
        
        env_info = reset_env_info(env)
        score = np.zeros(total_agents)
        dones = np.zeros(total_agents)
        while not np.all(dones):
            observations = env_info.vector_observations
            actions = multiagent.act(observations)
            env_info = env_step(env, actions)
            next_observations = env_info.vector_observations
            rewards = np.round(env_info.rewards, decimals=2)
            dones = env_info.local_done
            score = np.round(score + rewards, decimals=2)
            print('\rScore: {:.2f}/{:.2f}'.format(score[0], score[1]), end=' ')
            
            multiagent.step(observations, actions, rewards, next_observations, dones)

        max_score = np.max(score)
        scores_window.append(max_score)
        scores.append(max_score)
        moving_avg = np.mean(scores_window).round(decimals=2)
        
        print('-> Ep. {}/{} - Avg Max Global Score: {:.2f} - time: {}'.format(episode, episodes, np.mean(scores_window), datetime.datetime.now() - init_time))
        writer.add_scalar('Score_Player_0', score[0], episode)
        writer.add_scalar('Score_Player_1', score[1], episode)
        writer.add_scalar('Moving_Average_{}'.format(MOVING_AVG_WINDOW), moving_avg, episode)

        if not solved and moving_avg >= SOLVING_SCORE:
            solved = True
            print('\n*** Environment solved (mean of {:.2} for {:d} episodes) in {:d} episodes!\tAverage Score: {:.2f} ***\n'.format(SOLVING_SCORE, MOVING_AVG_WINDOW, episode, np.mean(scores_window)))
            for agent_number, agent in enumerate(multiagent.agents):
                torch.save(agent.actor.state_dict(), 'agent{}-actor_checkpoint.pth'.format(agent_number))
                torch.save(agent.actor_target.state_dict(), 'agent{}-actor_target_checkpoint.pth'.format(agent_number))
            
                torch.save(agent.twin_critic.state_dict(), 'agent{}-twin_critic_checkpoint.pth'.format(agent_number))
                torch.save(agent.twin_critic_target.state_dict(), 'agent{}-twin_critic_target_checkpoint.pth'.format(agent_number))
                
    for agent_number, agent in enumerate(multiagent.agents):
        torch.save(agent.actor.state_dict(), 'agent{}-actor_final_checkpoint.pth'.format(agent_number))
        torch.save(agent.actor_target.state_dict(), 'agent{}-actor_target_final_checkpoint.pth'.format(agent_number))

        torch.save(agent.twin_critic.state_dict(), 'agent{}-twin_critic_final_checkpoint.pth'.format(agent_number))
        torch.save(agent.twin_critic_target.state_dict(), 'agent{}-twin_critic_target_final_checkpoint.pth'.format(agent_number))
    
    writer.flush()
    writer.close()
    print('\n *** End of Training ***\n')
    
    env.close()
    return scores
    
def save_scores(scores):
    moving_average = np.asarray([np.mean(scores[max(index-MOVING_AVG_WINDOW+1, 0):index+1]) for index in range(len(scores))]).round(decimals=2)
    index_solved = np.where(moving_average >= SOLVING_SCORE)[0].min()
    
    fig = plt.figure(figsize=(20, 10))
    
    plt.plot(np.arange(len(scores)), scores, label='Max Player Score')
    plt.plot(np.arange(len(moving_average)), moving_average, label='Moving Average {}'.format(MOVING_AVG_WINDOW))
    plt.plot(index_solved, moving_average[index_solved], 'o--', label='Solved')
    
    plt.legend()
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    plt.savefig('score-evolution.png')
    #plt.show()

TOTAL_EPISODES = 500
SEED = 0

SOLVING_SCORE = 0.5
MOVING_AVG_WINDOW = 100

if __name__ == '__main__':
    scores = multi_agent_td3_run(TOTAL_EPISODES, SEED)
    save_scores(scores)