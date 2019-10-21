import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from dueling_model import DuelingQNetwork
from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 4.8e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_DOUBLE_DQN = True
USE_PRIORITIZED_REPLAY = False
USE_DUELING_NETWORK = True

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, lr_decay=0.9999):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            lr_decay (float): multiplicative factor of learning rate decay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        print("Running on: "+str(device))
        
        # Q-Network
        hidden_layers = [128, 32]
        
        if USE_DUELING_NETWORK:
            hidden_state_value = [64, 32]
            
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, hidden_layers, hidden_state_value).to(device)

            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, hidden_layers, hidden_state_value).to(device)
            self.qnetwork_target.eval()
            
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layers).to(device)

            self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layers).to(device)
            self.qnetwork_target.eval()
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)


        # Replay memory
        if USE_PRIORITIZED_REPLAY:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device,
                                                  alpha=0.6, beta=0.4, beta_scheduler=1.0)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
        
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
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, w) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, w = experiences

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

        if USE_PRIORITIZED_REPLAY:
            target_rewards.sub_(expected_rewards)
            target_rewards.squeeze_()
            target_rewards.pow_(2)
            
            with torch.no_grad():
                td_error = target_rewards.detach()
                td_error.pow_(0.5)
                self.memory.update_priorities(td_error)
            
            target_rewards.mul_(w)
            loss = target_rewards.mean()
        else:
            # calculate the loss
            loss = F.mse_loss(expected_rewards, target_rewards)

        # perform the back-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

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

