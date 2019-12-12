import torch.nn.functional as F
import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, *, num_layers = 3, hidden_dim = 400):
        """ TD3 PyTorch Actor Network model.

        Args:
            - state_dim:  Dimensionality of state.
            - action_dim: Dimensionality of actions.
            - max_action: Maximum action value.
            - num_layers: Number of total linear layers.
            - hidden_dim: Number of neurons in the hidden layers.
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        #defining the network layers
        self.layers = nn.ModuleList([nn.Linear(state_dim, hidden_dim)])
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers-2)])
        self.layers.append(nn.Linear(hidden_dim, action_dim))
        
        #defining the activation functions for the network - all layers have activation function ReLU except last
        #layer has activation function TanH
        self.activations = nn.ModuleList([nn.ReLU(hidden_dim) for i in range(num_layers-1)])
        
    def forward(self, states):
        """
        Q function mapping from states to action-values. Using the defined layers and activations for actor network 
        to compute the action-values tensor associated with the input states.

        :param states: (*, S) torch.Tensor where * is any number of additional.
                       dimensions, and S is the dimensionality of state-space.
        :rtype:        (*, A) torch.Tensor where * is the same number of additional.
                       dimensions as the `states`, and A is the dimensionality of the.
                       action-space.  This represents the Q values Q(s, .).
        """
        x = self.layers[0](states)
        x = self.activations[0](x) 
                         
        for layer_num in range(1, len(self.layers) - 1):
            x = self.layers[layer_num](x)
            x = self.activations[layer_num](x)
        
        output = torch.tanh(self.layers[-1](x))
        
        return self.max_action * output