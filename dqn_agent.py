import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_DOUBLE_DQN = True

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        print("Running on: "+str(device))
        
        # Q-Network
        hidden_layer_1 = 128
        hidden_layer_2 = 32
        self.qnetwork_local = QNetwork(state_size, action_size, seed,
                                       hidden_layer_1=hidden_layer_1, hidden_layer_2=hidden_layer_2).to(device)
        
        self.qnetwork_target = QNetwork(state_size, action_size, seed,
                                       hidden_layer_1=hidden_layer_1, hidden_layer_2=hidden_layer_2).to(device)
        self.qnetwork_target.eval()
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy())
        
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss

        with torch.no_grad():
            ### Use of Double DQN method
            if USE_DOUBLE_DQN:
                ## Select the greedy actions using the QNetwork Local
                # calculate the pair action/reward for each of the next_states
                next_action_rewards_local = self.qnetwork_local(next_states)
                # select the action with the maximum reward for each of the next actions
                greedy_actions_local = next_action_rewards_local.max(dim=1, keepdim=True)[1]

                ## Get the rewards for the greedy actions using the QNetwork Target
                # calculate the pair action/reward for each of the next_states
                next_action_rewards_target = self.qnetwork_target(next_states)
                # get the target reward for each of the greedy actions selected following the local network
                target_rewards = next_action_rewards_target.gather(1, greedy_actions_local)
                
            ### Use of Fixed Q-Target
            else:
                # calculate the pair action/reward for each of the next_states
                next_action_rewards = self.qnetwork_target(next_states)
                # select the maximum reward for each of the next actions
                target_rewards = next_action_rewards.max(dim=1, keepdim=True)[0]
                
            
            ## Calculate the discounted target rewards
            target_rewards = rewards + (gamma * target_rewards * (1 - dones))
            
            
        # calculate the pair action/rewards for each of the states
        expected_action_rewards = self.qnetwork_local(states) # shape: [batch_size, action_size]
        # get the reward for each of the actions
        expected_rewards = expected_action_rewards.gather(1, actions) # shape: [batch_size, 1]
        
        # calculate the loss
        loss = F.mse_loss(expected_rewards, target_rewards)
        
        # perform the back-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)