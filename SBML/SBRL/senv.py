import torch
from ..ZonoTorch.set import Interval

class SetEnvironmnent:
    """
    SetEnvironment: Environment for Set-Based Reinforcement Learning
    ================================================================

    The SetEnvironment is a class that defines the environment for set-based reinforcement learning.

    Attributes:
    -----------
    - init_state: Initial state of the environment (Interval)
    - options: Dictionary with options for the environment
        - ct: 0.1 (default) - Control time step
        - dt: 0.01 (default) - Simulation time step
        - max_steps: 100 (default) - Maximum number of steps
        - initial_ops: 'uniform' (default) - Initial state option for the environment ('uniform', 'symmetric', 'center')
    - dynamics: Dynamics of the environment
    - rewardfun: Reward function of the environment
    - collisioncheck: None (default) - Function for collision checking
    - device: cuda (gpu) or cpu
    - state: Current state of the environment
    - stepNum: Number of steps taken in the environment
    """

    def __init__(self,init_state,options,dynamics,rewardfun,collisioncheck=None,device='cpu'):
        self.device = device
        self.options = self.__validateOptions(options)
        assert isinstance(init_state, Interval) and init_state._batchSize < 2
        if init_state._batchSize == 0:
            init_state._tensor = init_state._tensor.unsqueeze(2)
        self.init_state = init_state
        self.dynamic = dynamics
        self.rewardfun = rewardfun
        self.collisioncheck = collisioncheck
        self.state = None
        self.isDone = False
        self.stepNum = 0
    
    def __validateOptions(self,options):
        default_options = {
            'ct': 0.1,
            'dt': 0.01,
            'max_steps': 100,
            'initial_ops': 'uniform',
        }

        for key in default_options.keys():
            if key not in options:
                options[key] = default_options[key]
        return options
    
    def reset(self):
        """Resets the environment"""
        self.stepNum = 0
        self.isDone = False
        if self.options['initial_ops'] == 'uniform':
            randn = torch.rand((self.init_state._dim,1)).to(self.device)
            state = self.init_state.getLower() + (self.init_state.getUpper()-self.init_state.getLower())*randn
            self.state = state.permute(1,0)
            return self.state
        elif self.options['initial_ops'] == 'symmetric':
            randi = torch.randint(0,1,(self.init_state._dim,1)).to(self.device)
            state = self.init_state.getLower() + (self.init_state.getUpper()-self.init_state.getLower())*randi
            self.state = state.permute(1,0)
            return self.state
        elif self.options['initial_ops'] == 'center':
            self.state = 1/2*(self.init_state.getLower()+self.init_state.getUpper()).permute(1,0)
            return self.state
        
    def step(self,action):
        assert self.state is not None and not self.isDone and isinstance(action,torch.Tensor)
        self.stepNum += 1
        next_state = self.step_dynamics(self.state,action,self.options['ct'],self.options['dt'])
        reward = self.rewardfun(self.state,action,next_state)
        self.state = next_state
        if self.collisioncheck is not None:
            self.isDone = self.stepNum >= self.options['max_steps'] or self.collisioncheck(self.state)
        else:
            self.isDone = self.stepNum >= self.options['max_steps']

        return next_state, reward, self.isDone, {}
    
    def step_dynamics(self,state,action,ct,dt):
        assert ct >= dt 
        num_steps = int(ct/dt)
        dtt = ct/num_steps

        interm_state = state
        for _ in range(num_steps):
            interm_state = interm_state + dtt*self.dynamic(interm_state,action)
            
        return interm_state
        

    

        