import torch
import numpy as np

class ReplayMemory:
    def __init__(self, state_size, action_size, max_size):
        """
        Implementing the Replay Memory, the data-structure to store previous experiences 
        so that can re-sample and train on them. Replay memory implemented as a circular buffer.
        Experiences will be removed in a FIFO manner after reaching maximum buffer size.

        Args:
            - state_size: Size of the state-space features for the environment.
            - action_size: Size of the action-space for the environment.
            - max_size: Maximum size of the buffer.
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.max_size = max_size

        # preallocating all the required memory, for speed concerns
        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, action_size))
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)

        # pointer to the current location in the circular buffer
        self.idx = 0
        # indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        :param state:       np.ndarray of state-features.
        :param action:      np.ndarray of action-features.
        :param reward:      float reward.
        :param next_state:  np.ndarray of state-features.
        :param done:        boolean value indicating the end of an episode.
        """
        
        # store the input values into the appropriate attributes, using the current buffer position index
        self.states[self.idx] = torch.from_numpy(state)
        self.actions[self.idx] = torch.from_numpy(action)
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = torch.from_numpy(next_state)
        self.dones[self.idx] = done
                
        # circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample a batch of experiences. If the buffer contains less than batch size transitions, sample all
        of them.

        :param batch_size:  Number of transitions to sample.
        :returns:           A tuple (states, actions, rewards, next_states, dones)
        """
        
        sample_indices = np.random.choice(self.size, min(self.size, batch_size), replace=False)
        
        states = self.states[sample_indices]
        actions = self.actions[sample_indices]
        rewards = self.rewards[sample_indices]
        next_states = self.next_states[sample_indices]
        dones = self.dones[sample_indices]

        return states, actions, rewards, next_states, dones

    def populate(self, env, num_steps):
        """Populate this replay memory with num_steps from the random policy.

        :param env:        Openai Gym environment
        :param num_steps:  Number of steps for which to populate the replay memory with the resulting transitions.
        """
        
        state = env.reset()
        for i in range(num_steps):
            random_action = env.action_space.sample()
            
            next_state, reward, done, info = env.step(random_action)
            
            self.add(state, random_action, reward, next_state, done)
            
            if done:
                state = env.reset()
            else:
                state = next_state