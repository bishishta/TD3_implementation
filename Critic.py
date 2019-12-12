import torch.nn.functional as F
import torch.nn as nn
import torch

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers = 3, hidden_dim = 400):
        """ TD3 PyTorch Critic Network models. There two critic networks in TD3 

        Args:
            - state_dim:  Dimensionality of states.
            - action_dim: Dimensionality of actions.
            - num_layers: Number of total linear layers.
            - hidden_dim: Number of neurons in the hidden layers.
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        #defining the network layers for first critic network
        self.layers_q1 = nn.ModuleList([nn.Linear(state_dim + action_dim, hidden_dim)])
        self.layers_q1.extend([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers-2)])
        self.layers_q1.append(nn.Linear(hidden_dim, 1))
        
        #defining the network layers for second critic network
        self.layers_q2 = nn.ModuleList([nn.Linear(state_dim + action_dim, hidden_dim)])
        self.layers_q2.extend([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers-2)])
        self.layers_q2.append(nn.Linear(hidden_dim, 1))
        
        #defining the activation functions for both the critic networks - all layers have activation function 
        #ReLU except last layer 
        self.activations_q1 = nn.ModuleList([nn.ReLU(hidden_dim) for i in range(num_layers-1)])
        self.activations_q2 = nn.ModuleList([nn.ReLU(hidden_dim) for i in range(num_layers-1)])

        
    def forward(self, states, actions, Q1 = False):
        """
        Q function mapping from state-action pairs to corresponding action-values. Using the defined layers and 
        activations for both critic networks to compute the action-values tensor associated with the input state-action pairs.

        :param states:  (*, S) torch.Tensor where * is any number of additional.
                        dimensions, and S is the dimensionality of state-space.
        :param actions: (*, A) torch.Tensor where * is the same number of additional.
                        dimensions as the `states`, and A is the dimensionality of the action-space.
        :param Q1:      if True, the action-values need to be calvulated just for first critic network.
        :rtype:         (*,1) torch.Tensor where * is the same number of additional.
                        dimensions as the `states`, This represents the Q values Q(s, a).
        """
        
        state_action_pairs = torch.cat([states, actions], 1)
        
        x1 = self.layers_q1[0](state_action_pairs)
        x1 = self.activations_q1[0](x1) 
                         
        for layer_num in range(1, len(self.layers_q1) - 1):
            x1 = self.layers_q1[layer_num](x1)
            x1 = self.activations_q1[layer_num](x1)
        
        output_q1 = self.layers_q1[-1](x1)
        
        if not Q1:
            x2 = self.layers_q2[0](state_action_pairs)
            x2 = self.activations_q2[0](x2) 

            for layer_num in range(1, len(self.layers_q2) - 1):
                x2 = self.layers_q2[layer_num](x2)
                x2 = self.activations_q2[layer_num](x2)

            output_q2 = self.layers_q2[-1](x2)
        
            return output_q1, output_q2
        
        return output_q1