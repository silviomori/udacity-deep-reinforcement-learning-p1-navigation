import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_advantage=[512, 512], hidden_state_value=[512,512]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): List containing the hidden layer sizes
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Include state_size as the first parameter to create the layers
        hidden_layers = [state_size] + hidden_advantage
        
        ## Create the advantage network
        # Create an OrderedDict to store the network layers
        advantage_layers = OrderedDict()

        # Iterate over the parameters to create the advantage network
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1],hidden_layers[1:])):
            # Add a linear layer
            advantage_layers['adv_fc_'+str(idx)] = nn.Linear(hl_in, hl_out)
            # Add an activation function
            advantage_layers['adv_activation_'+str(idx)] = nn.ReLU()
        
        # Create the output layer for the advantage network
        advantage_layers['adv_output'] = nn.Linear(hidden_layers[-1], action_size)
        
        # Create the advantage network
        self.network_advantage = nn.Sequential(advantage_layers)
        
        
        ## Create the value network
        # Create an OrderedDict to store the network layers
        value_layers = OrderedDict()
        hidden_layers = [state_size] + hidden_state_value

        # Iterate over the parameters to create the value network
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1],hidden_layers[1:])):
            # Add a linear layer
            value_layers['val_fc_'+str(idx)] = nn.Linear(hl_in, hl_out)
            # Add an activation function
            value_layers['val_activation_'+str(idx)] = nn.ReLU()
        
        # Create the output layer for the value network
        value_layers['val_output'] = nn.Linear(hidden_layers[-1], 1)
        
        # Create the value network
        self.network_value = nn.Sequential(value_layers)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # Perform a feed-forward pass through the networks
        advantage = self.network_advantage(state)
        value = self.network_value(state)

        # Return the aggregated modules
        return advantage.sub_(advantage.mean()).add_(value)