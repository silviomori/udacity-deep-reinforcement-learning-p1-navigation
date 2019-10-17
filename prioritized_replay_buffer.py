import numpy as np
import torch

class PrioritizedReplayBuffer:
    """Fixed-size prioritized buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, device, alpha=0., beta=1., beta_scheduler=1.):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): determines how much prioritization is used; α = 0 corresponding to the uniform case
            beta (float): amount of importance-sampling correction; β = 1 fully compensates for the non-uniform probabilities
            beta_scheduler (float): multiplicative factor (per sample) for increasing beta (should be >= 1.0)
        """
        
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_scheduler = beta_scheduler
        
        # Create a Numpy Array to store tuples of experience
        self.memory = np.empty(buffer_size, dtype=[
            ("state", np.ndarray),
            ("action", np.int),
            ("reward", np.float),
            ("next_state", np.ndarray),
            ("done", np.bool),
            ('prob', np.double)])
        # Variable to control the memory buffer as being a circular list
        self.memory_idx_ctrl = 0
        
        # Variable to control the selected samples
        self.memory_samples_idx = np.empty(batch_size)
        # Numpy Array to store selected samples
        # Those samples could be controlled only by the index,
        # however keeping an allocated space in memory improves performance.
        # (Here we have a tradeoff between memory space and cumputacional processing)
        self.memory_samples = np.empty(batch_size, dtype=type(self.memory))

        # Each new experience is added to the memory with
        # the maximum probability of being choosen
        self.max_prob = 0.0001
        
        # Value to a non-zero probability
        self.nonzero_probability = 0.00001
        
        # Numpy Arrays to store probabilities and weights
        # (tradeoff between memory space and cumputacional processing)
        self.p = np.empty(buffer_size, dtype=np.double)
        self.w = np.empty(buffer_size, dtype=np.double)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        # Add the experienced parameters to the memory
        self.memory[self.memory_idx_ctrl]['state'] = state
        self.memory[self.memory_idx_ctrl]['action'] = action
        self.memory[self.memory_idx_ctrl]['reward'] = reward
        self.memory[self.memory_idx_ctrl]['next_state'] = next_state
        self.memory[self.memory_idx_ctrl]['done'] = done
        self.memory[self.memory_idx_ctrl]['prob'] = self.max_prob
        
        # Control memory as a circular list
        self.memory_idx_ctrl = (self.memory_idx_ctrl + 1) % self.buffer_size
    
    def sample(self):
        """Sample a batch of prioritized experiences from memory."""
        
        # Normalize the probability of being chosen for each one of the memory registers
        np.divide(self.memory['prob'], self.memory['prob'].sum(), out=self.p)
        # Choose "batch_size" sample index following the defined probability
        self.memory_samples_idx = np.random.choice(self.buffer_size, self.batch_size, replace=False, p=self.p)
        # Get the samples from memory
        self.memory_samples = self.memory[self.memory_samples_idx]
        
        # Compute importance-sampling weights for each one of the memory registers
        # w = ((N * P) ^ -β) / max(w)
        np.multiply(self.memory['prob'], self.buffer_size, out=self.w)
        np.power(self.w, -self.beta, out=self.w, where=self.w!=0) # condition to avoid division by zero
        np.divide(self.w, self.w.max(), out=self.w) # normalize the weights
        
        self.beta = min(1, self.beta*self.beta_scheduler)
        
        # Split data into new variables
        states = torch.from_numpy(np.vstack(self.memory_samples['state'])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(self.memory_samples['action'])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(self.memory_samples['reward'])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(self.memory_samples['next_state'])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(self.memory_samples['done'])).float().to(self.device)
        weights = torch.from_numpy(self.w[self.memory_samples_idx]).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones, weights)
    
    def update_priorities(self, td_error):
        # Balance the prioritization using the alpha value
        td_error.pow_(self.alpha)

        # Guarantee a non-zero probability
        td_error.add_(self.nonzero_probability)
        
        # Update the probabilities in memory
        self.memory_samples['prob'] = td_error
        self.memory[self.memory_samples_idx] = self.memory_samples
        
        # Update the maximum probability value
        self.max_prob = self.memory['prob'].max()
        
       
    def __len__(self):
        """Return the current size of internal memory."""
        return self.buffer_size if self.memory_idx_ctrl // self.buffer_size > 0 else self.memory_idx_ctrl