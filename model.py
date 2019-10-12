import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[512, 512]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): List containing the hidden layer sizes
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Create an OrderedDict to store the network layers
        layers = OrderedDict()
        
        # Include state_size and action_size as layers
        hidden_layers = [state_size] + hidden_layers + [action_size]
        
        # Iterate over the parameters to create layers
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1],hidden_layers[1:])):
            # Add a linear layer
            layers['fc'+str(idx)] = nn.Linear(hl_in, hl_out)
            # Add an activation function
            layers['activation'+str(idx)] = nn.ReLU()
        
        # Remove the last activation layer
        layers.popitem()
        
        # Create the network
        self.network = nn.Sequential(layers)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # Perform a feed-forward pass through the network
        return self.network(state)