import torch
import gym
from gym.utils.save_video import save_video
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
        """
        Initializes the SetEnvironment

        Parameters:
        -----------
        - init_state: Initial state of the environment (Interval)
        - options: Dictionary with options for the environment
        - dynamics: Dynamics of the environment
        - rewardfun: Reward function of the environment
        - collisioncheck: Function for collision checking
        - device: cuda (gpu) or cpu
        """

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
        """
        Validates the options for the environment

        Parameters:
        -----------
        - options: Dictionary with options for the environment

        Returns:
        --------
        - options: Dictionary with validated options for the environment
        """

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
        """
        Resets the environment
        
        Returns:
        --------
        - state: Initial state of the environment
        """

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
        """
        Takes a step in the environment

        Parameters:
        -----------
        - action: Action to take in the environment

        Returns:
        --------
        - next_state: Next state of the environment
        - reward: Reward for the step
        - isDone: Boolean to check if the episode is done
        - {}: Empty dictionary
        """

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
        """
        Steps the dynamics of the environment
        
        Parameters:
        -----------
        - state: Current state of the environment
        - action: Action to take in the environment
        - ct: Control time step
        - dt: Simulation time step

        Returns:
        --------
        - final_state: Final state of the environment
        """

        assert ct >= dt 
        num_steps = int(ct/dt)
        dtt = ct/num_steps

        interm_state = state
        for _ in range(num_steps):
            interm_state = interm_state + dtt*self.dynamic(interm_state,action)

        final_state = interm_state
        return final_state
        

class GymEnvironment:
    """
    GymEnvironment: Environment for Gym Reinforcement Learning
    ==========================================================

    The GymEnvironment is a class that defines the environment for gym-based reinforcement learning.

    Attributes:
    -----------
    - env: Gym environment
    - options: Dictionary with options for the environment
        - max_steps: 1000 (default) - Maximum number of steps
        - reset_noise_scale: 1e-3 (default) - Reset noise scale
        - video_folder: './videos' (default) - Folder to save videos
    - device: cuda (gpu) or cpu
    - isDone: Boolean to check if the episode is done
    - stepNum: Number of steps taken in the environment
    - max_steps: Maximum number of steps
    - frames: Frames of the environment for video
    - episode: Number of episodes
    - eval_run: Boolean to check if the run is evaluation
    """

    def __init__(self, gym_env_name, options, device='cpu'):
        """
        Initializes the GymEnvironment
        
        Parameters:
        -----------
        - gym_env_name: Name of the gym environment
        - options: Dictionary with options for the environment
        - device: cuda (gpu) or cpu
        """

        self.options = self.__validateOptions(options)
        self.device = device
        self.env = gym.make(gym_env_name, render_mode="rgb_array", reset_noise_scale=self.options['reset_noise_scale'])
        self.isDone = False
        self.stepNum = 0
        self.max_steps = self.options['max_steps']
        self.frames = []
        self.episode = 0
        self.eval_run = False


    def __validateOptions(self,options):
        """
        Validates the options for the environment

        Parameters:
        -----------
        - options: Dictionary with options for the environment

        Returns:
        --------
        - options: Dictionary with validated options for the environment
        """

        default_options = {
            'max_steps': 1000,
            'reset_noise_scale': 1e-3,
            'video_folder': './videos',
        }

        for key in default_options.keys():
            if key not in options:
                options[key] = default_options[key]

        return options
    
    
    def reset(self,eval_run=False):
        """
        Resets the environment
        
        Parameters:
        -----------
        - eval_run: Boolean to check if the run is evaluation

        Returns:
        --------
        - observation: Initial observation of the environment
        """

        self.stepNum = 0
        self.isDone = False

        if self.frames != [] and self.eval_run:
            save_video(self.frames, video_folder=self.options['video_folder'], video_length=len(self.frames), fps=30, name_prefix=f"rl-video-{self.episode}")

        self.eval_run = eval_run

        self.episode += 1
        self.frames = []
        observation = torch.as_tensor(self.env.reset()[0],dtype=torch.float32).to(self.device).unsqueeze(0)
        return observation
    
    def step(self,action):
        """
        Takes a step in the environment

        Parameters:
        -----------
        - action: Action to take in the environment

        Returns:
        --------
        - next_state: Next state of the environment
        - reward: Reward for the step
        - isDone: Boolean to check if the episode is done
        - info: Information about the environment
        """

        assert not self.isDone
        self.stepNum += 1
        a = action.squeeze().detach().cpu().numpy()
        next_state, reward, done, info, _ = self.env.step(a)
        next_state = torch.as_tensor(next_state,dtype=torch.float32).to(self.device).unsqueeze(0)
        reward = torch.as_tensor(reward,dtype=torch.float32).to(self.device)
        done = torch.as_tensor(done,dtype=torch.bool).to(self.device)
        frame = self.env.render()
        self.frames.append(frame)
        self.isDone = done or torch.as_tensor(self.stepNum >= self.max_steps,dtype=torch.bool).to(self.device)
        return next_state, reward, self.isDone, info