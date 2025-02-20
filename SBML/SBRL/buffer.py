import torch

class Buffer: 
    """ 
    Buffer: Replay Buffer for RL Agent
    ===================================

    This class implements the replay buffer for the RL agent and vizualization.
    The buffer stores (s,a,r,s',d) tuples for the agent to learn from.

    Attributes:
    -----------
    - buffer_size: Size of the replay buffer
    - device: gpu or cpu
    - state_dim: Dimension of the state space
    - action_dim: Dimension of the action space
    - num_generators: Number of generators of the zonotope (default: None)
    """
    
    def __init__(self, buffer_size, device, state_dim, action_dim, num_generators = None):
        """
        Initializes the replay buffer

        Parameters:
        -----------
        - buffer_size: Size of the replay buffer
        - device: gpu or cpu
        - state_dim: Dimension of the state space
        - action_dim: Dimension of the action space
        - num_generators: Number of generators of the zonotope (default: None)
        """

        self.device = device
        self.buffer_size = buffer_size
        self.buffer = {}

        self.buffer['state'] = torch.zeros((buffer_size, state_dim), dtype=torch.float32).to(device)

        if num_generators:
            self.buffer['action'] = torch.zeros((buffer_size, action_dim, num_generators+1), dtype=torch.float32).to(device)
        else:
            self.buffer['action'] = torch.zeros((buffer_size, action_dim), dtype=torch.float32).to(device)

        self.buffer['reward'] = torch.zeros((buffer_size, 1), dtype=torch.float32).to(device)
        self.buffer['next_state'] = torch.zeros((buffer_size, state_dim), dtype=torch.float32).to(device)
        self.buffer['done'] = torch.zeros((buffer_size, 1), dtype=torch.float32).to(device)

        self.indx = 0
        self.full = False
        

    def reset(self):
        """Resets the replay buffer"""
        self.indx = 0
        self.full = False

    def add(self, state, action, reward, next_state, done):
        """
        Adds a transition to the replay buffer

        Parameters:
        -----------
        - state: Current state
        - action: Action taken
        - reward: Reward received
        - next_state: Next state
        - done: Boolean to check if the episode is done
        """

        self.buffer['state'][self.indx] = state
        self.buffer['action'][self.indx] = action
        self.buffer['reward'][self.indx] = reward
        self.buffer['next_state'][self.indx] = next_state
        self.buffer['done'][self.indx] = done

        self.indx += 1
        if self.indx == self.buffer_size:
            self.full = True
            self.indx = 0


    def sample(self, batch_size):
        """
        Samples a batch from the replay buffer

        Parameters:
        -----------
        - batch_size: Size of the batch

        Returns:
        --------
        - state: Current state
        - action: Action taken
        - reward: Reward received
        - next_state: Next state
        - done: Boolean to check if the episode is done
        """

        if self.full:
            indx = torch.randint(0, self.buffer_size, (batch_size,))
        else:
            if self.indx < batch_size:
                raise ValueError("Not enough samples in the buffer")
            else:
                indx = torch.randint(0, self.indx, (batch_size,))

        return self.buffer['state'][indx], self.buffer['action'][indx], self.buffer['reward'][indx], self.buffer['next_state'][indx], self.buffer['done'][indx]


    