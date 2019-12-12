from Replay_Memory import *
from TD3_agent import *
from Render import *

import gym
import tqdm
import math
import pickle

"""
Implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) using pytorch 

Reference 1: https://github.com/sfujim/TD3/blob/master/main.py 
Reference 2: https://github.com/sfujim/TD3
Reference 3: DQN programming assignment

Environments
The following two environments chosen have a continuous state-space and continuous action-space:
Hopper: Make a two-dimensional one-legged robot hop forward as fast as possible (https://gym.openai.com/envs/Hopper-v2/).
HalfCheetah: (https://gym.openai.com/envs/HalfCheetah-v2/).
"""

Env_Hopper = gym.make('Hopper-v3')
Env_HalfCheetah = gym.make('HalfCheetah-v2')

def train_td3(env, replay_size, replay_populate_steps, total_timesteps, exploration_noise, gamma, reward_threshold, \
              sample_batch_size, tau, policy_noise, noise_clip, policy_update_frequency, path, visulize_num_episodes):
    """ Training a Td3 (agent) policy network model. 
    :param env: Openai Gym environment.
    :param replay_size: Maximum size of the buffer.
    :param replay_populate_steps: Number of steps for which to populate the replay memory.
    :param total_timesteps: Total number of timesteps for which to repeat policy training.
    :param exploration_noise: Exploration noise to be added to action.
    :param gamma:The discount factor.
    :param reward_threshold: The threshold for reward after which stop training.
    :param sample_batch_size: Number of experiences in a batch.
    :param tau: Soft update for main networks to target networks.
    :param policy_noise: amount of noise to be added to actions.
    :param noise_clip: decides the range for clipping the noise.
    :param: policy_update_frequency: after how many timesteps the policy network needs to be updated.
    :param path: directory to store data.
    :param visulize_num_episodes: number of episodes to visualize the trained agent for.
    """

    # get the state_size, action_size, and max_action from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) 

    # initialize the replay memory and prepopulate it
    memory = ReplayMemory(state_size, action_size, replay_size)
    memory.populate(env, replay_populate_steps)
    
    # initialize the TD3 agent (policy) network model
    policy_network = TD3(env, state_size, action_size, max_action) 
    
    #initialize first state of episode
    state = env.reset()
    
    #initializing episode length and best average reward (average of rewards achieved in last 100 timestepssteps)
    ep_length = 0
    best_average_reward = -2000
    
    # initiate lists to store returns, current episode rewards, rewards of all the timsteps lengths and 
    #average rewards (average of rewards achieved in last 100 timestepssteps)
    returns = []
    ep_rewards = []
    all_rewards = []
    average_rewards = []
    
    # iterate for a total number of timesteps
    pbar = tqdm.tnrange(total_timesteps, ncols='100%')
    for step in pbar:
        
         
        #Sample a clipped action with exploration noise from the TD3 policy  and use the action to advance 
        #the environment by one step. Store the transition into the replay memory.
        action = policy_network.select_clipped_action(state, exploration_noise)        
        next_state, reward, done, _ = env.step(action)       
        memory.add(state, action, reward, next_state, done)
        
        ep_rewards.append(reward)
        all_rewards.append(reward)
        
        state = next_state
        ep_length += 1
        
        #maximum length of an episode can be 200
        if ep_length + 1 == 200:
            done = True
        
        if done:
            #calculating episode return
            G = sum([math.pow(gamma, i) * ep_rewards[i] for i in range(len(ep_rewards))])
            returns.append(G)
            
            #calculating average reward (average of rewards achieved in last 100 timestepssteps)
            average_reward = np.mean(all_rewards[-100:])
            average_rewards.append(average_reward)
            
            pbar.set_description(
                f'Episode Length: {ep_length} | Return: {G:5.2f} | Average Reward: {average_reward:4.2f}'
            )
            
            #if average_reward > best_average_reward till now, new best_average_reward = average_reward 
            #and save the hence acheived best policy
            if best_average_reward < average_reward:
                best_average_reward = average_reward
                policy_network.custom_save(path)
                
            #if average_reward > reward_threshold, stop training the policy
            if average_reward >= reward_threshold:
                break
                
            #train and update the policy network
            policy_network.train_network(memory, ep_length, sample_batch_size, gamma, tau, \
                      policy_noise, noise_clip, policy_update_frequency)
            
            ep_length = 0
            ep_rewards = []
            #reset the environment state
            state = env.reset()
            
    #saving the average rewards list
    with open(path + 'average_rewards.txt','wb') as fp:
        pickle.dump(average_rewards, fp)
        
    #saving the episode rewards list
    with open(path + 'returns.txt','wb') as fp:
        pickle.dump(returns, fp)
        
    #visualizing the performance of the trained agent/ learnt policy
    render(env, path, policy_network, visulize_num_episodes)