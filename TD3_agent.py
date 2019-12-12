from Actor import *
from Critic import *
import torch
import numpy as np

class TD3(object):
    def __init__(self, env, state_size, action_size, max_action):
        """ TD3 PyTorch Td3 (agent) policy network model. 

        Args:
            - env: Openai Gym environment
            - state_size:  Dimensionality of states.
            - action_size: Dimensionality of actions.
            - max_action: Maximum action value.
        """
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action

        self.actor = Actor(self.state_size, self.action_size, self.max_action)
        self.actor_target = Actor(self.state_size, self.action_size, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
    def select_clipped_action(self, state, exploration_noise = 0.1):
        """Observe the state and select a clipped action (within the action range) with exploration noise
        using the agent policy.
        
        :param state: current state
        :param exploration_noise: amount of exploration noise to be added to actions
        :returns: clipped action with noise regularization     
        """
            
        action = self.actor(torch.FloatTensor(state.reshape(1, -1))).data.numpy().flatten()
        if exploration_noise != 0:
            action += np.random.normal(0, exploration_noise, size = self.action_size)
        action = action.clip(self.env.action_space.low, self.env.action_space.high)
        
        return action
        
    def train_network(self, replay_buffer, train_steps, batch_size, gamma, tau, policy_noise, \
                      noise_clip, policy_update_frequency):
        """ Train the policy network for given number of timesteps and update it.
        
        :param replay_buffer: The replay buffer.
        :param train_steps: Total timesteps to be used for training.
        :param batch_size: Number of experiences in a batch.
        :param gamma: The discount factor.
        :param tau: Soft update for main networks to target networks.
        :param policy_noise: amount of noise to be added to actions.
        :param noise_clip: decides the range for clipping the noise.
        :param: policy_update_frequency: after how many timesteps the policy network needs to be updated.
        """
        
        for step in range(train_steps):
            
            #randomly sample a batch of transitions from replay buffer of size = batch_size
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            #compute target action
            noise = actions.data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            target_action = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            #compute target for both the critic network by taking the minimum Q value from the two critic networks
            target_q1, target_q2 = self.critic_target(next_states, actions)
            target_q = rewards + (dones.type(torch.FloatTensor) * gamma * torch.min(target_q1, target_q2)).detach()
            
            #compute critics loss 
            current_q1, current_q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            #update Q-funnctions by one-step gradient descent
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            #Delaying update of policy network
            if step % policy_update_frequency == 0:
                
                #compute actor loss 
                actor_loss = -self.critic.forward(states, self.actor(states), Q1=True).mean()
                
                #update policy by one-step gradient ascent
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                #update target networks
                for parameter, target_parameter in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_parameter.data.copy_(tau * parameter.data + (1-tau) * target_parameter.data)
                
                for parameter, target_parameter in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_parameter.data.copy_(tau * parameter.data + (1-tau) * target_parameter.data)
                    
    def custom_save(self, path):
        """Cloning and storing actor and critic network data.
        
        :param path: path of directory where data will be stored.
        """
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic.state_dict(), path + 'critic.pth')
        
    def custom_load(self, path):
        """Retrieving stored actor and critic network data.
        
        :param path: path of directory where data is stored.
        """
        self.actor.load_state_dict(torch.load(path+'actor.pth'))
        self.critic.load_state_dict(torch.load(path+'critic.pth'))