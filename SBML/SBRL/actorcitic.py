from abc import ABC, abstractmethod
from time import time
import torch

from .buffer import Buffer
from ..ZonoTorch.set import Zonotope
from ..ZonoTorch.losses import ZonotopeRegressionLoss
from .senv import GymEnvironment


class ActorCritic(ABC):
    """
    ActorCritic: Abstract class for ActorCritic algorithm
    =====================================================

    Actor critic reinforcment leaning agent. The actor parameterizes the policy, while the critic estimates the value function.

    Attributes:
    -----------
    - actor: The actor network (torch.nn.Sequential)
    - critic: The critic network (torch.nn.Sequential)
    - options: Dictionary with options for the agent
        - gamma: .99 (default) - Discount factor
        - buffer_size: 10000 (default) - Size of the replay buffer
        - batch_size: 64 (default) - Batch size for training
        - actor_lr: 0.001 (default) - Learning rate of the actor
        - actor_l2: 0.01 (default) - L2 regularization of the actor
        - critic_lr: 0.001 (default) - Learning rate of the critic
        - critic_l2: 0.0 (default) - L2 regularization of the critic
        - critic_train_mode: 'set' (default) - Training mode of the critic ('set' or 'point')
        - criic_eta: 0.01 (default) - Eta weighting factor for set training
        - actor_train_mode: 'set' (default) - Training mode of the actor ('set', 'point', 'adv_grad', 'adv_naive')
        - actor_eta: 0.01 (default) - Eta weighting factor for set training
        - actor_omega: .5 (default) - Omega weighting factor for set training
        - noise: 0.1 (default) - Noise for set training
    - device: cuda (gpu) or cpu
    - buffer: Replay buffer for the agent
    - state_dim: Dimension of the state space
    - action_dim: Dimension of the action space
    """

    def __init__(self,actor,critic,options,device):
        """
        Initializes the actor-critic agent

        Parameters:
        -----------
        - actor: The actor network (torch.nn.Sequential)
        - critic: The critic network (torch.nn.Sequential)
        - options: Dictionary with options for the agent
        - device: cuda (gpu) or cpu
        """

        self.device = device
        self.options = self.__validateOptions(options)

        assert isinstance(actor, torch.nn.Sequential)
        self.__xavierInit(actor)
        self.actor = actor.to(device)

        assert isinstance(critic, torch.nn.Sequential)
        self.__xavierInit(actor)
        self.critic = critic.to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=options['actor_lr'], weight_decay=options['actor_l2'])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=options['critic_lr'],weight_decay=options['critic_l2'])

        if self.options['critic_train_mode'] == 'set':
            self.critic_loss = ZonotopeRegressionLoss(options['critic_eta'],options['noise'])
        elif self.options['critic_train_mode'] == 'point':
            self.critic_loss = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid critic training mode")

        num_generators = self.__getNumActionGens()

        if isinstance(self.actor[-1], torch.nn.Linear):
            self.action_dim = self.actor[-1].out_features
        else:
            self.action_dim = self.actor[-2].out_features

        self.state_dim = self.actor[0].in_features

        self.buffer = Buffer(int(options['buffer_size']),device,self.state_dim,self.action_dim,num_generators)

        self.learn_hist = {'reward':[]}

    
    def train(self,env,steps=1000,verbose=True):
        """
        Trains the agent in the environment

        Parameters:
        -----------
        - env: The environment (GymEnvironment or SetEnvironmnent)
        - steps: Number of training steps (simulatin steps of the environment)
        - verbose: If True, prints training information
        """

        self.buffer.reset()
        
        if self.options['actor_train_mode'] == 'set':
            self.noise = torch.eye(self.actor[0].in_features).to(self.device) * self.options['noise']

        self.__printOpionsInfo(steps)
        t_start = time()
        step = 0

        while step < steps:
            state = env.reset()
            done = False
            total_reward = torch.zeros(1, device=self.device)
            total_critic_loss = torch.zeros(1, device=self.device)
            total_actor_loss = torch.zeros(1, device=self.device)
            num_step = 0

            while not done:
                action = self.act(state)

                if isinstance(action, Zonotope):    
                    a = action.getCenter().reshape(-1,self.action_dim)
                else:
                    a = action

                if num_step == 0:
                    q_val = self.eval(state,a)
                
                next_state, reward, done, _ = env.step(a)

                if isinstance(action, Zonotope):
                    action = action._tensor.reshape(1,self.action_dim,-1)

                self.buffer.add(state,action,reward,next_state,done)

                state = next_state
                total_reward += reward

                if self.buffer.full or self.buffer.indx > self.options['batch_size']:
                    critic_loss, actor_loss = self.train_step()
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss

                num_step += 1
                step += 1

                if verbose and (step -1) % self.options['print_freq'] == 0:
                    self.__printTrainingInfo(time()-t_start,step,total_reward,q_val,total_critic_loss/num_step, total_actor_loss/num_step)

                if step % self.options['eval_freq'] == 0:
                    self.eval_agent(env)
                    break

    
    def eval_agent(self,env):
        """
        Evaluates the agent in the environment without exploration noise

        Parameters:
        -----------
        - env: The environment (GymEnvironment or SetEnvironmnent)
        """
        
        if isinstance(env, GymEnvironment):
            total_reward = torch.zeros(1, device=self.device)
            for i in range(10):
                if i == 0:
                    state = env.reset(eval_run=True)
                else: 
                    state = env.reset()

                done = False

                while not done:
                    action = self.actor(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward

            print("Evaluation Reward: {}".format(float(total_reward.cpu())/10.0))
            self.learn_hist['reward'].append(float(total_reward.cpu())/10.0)
            
            if total_reward > env.options['reward_save_threshold']:
                torch.save(self.actor.state_dict(), f"{env.options['agent_folder']}/actor{env.episode}.pth")

        else:
            state = env.reset()

            total_reward = torch.zeros(1, device=self.device)

            done = False

            while not done:
                action = self.actor(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

            self.learn_hist['reward'].append(float(total_reward.cpu()))


    def train_critic(self,state,action,target):
        """
        Trains the critic network

        Parameters:
        -----------
        - state: The state tensor
        - action: The action tensor
        - target: The target tensor

        Returns:
        --------
        - loss: The critic loss
        """

        self.critic_optim.zero_grad()

        if self.options['critic_train_mode'] == 'set':
            z_state = Zonotope(self.augmentState(state).permute(1,2,0))
            z_action = Zonotope(action.permute(1,2,0))
            z_input = torch.cartesian_prod(z_state,z_action)
            z_target = Zonotope(target.unsqueeze(1).permute(1,2,0))
            q_val = self.critic(z_input)
            loss = self.critic_loss(q_val,z_target)
        elif self.options['critic_train_mode'] == 'point':
            q_val = self.critic(torch.cat([state,action],dim=1))
            loss = self.critic_loss(q_val,target)
        else:
            raise ValueError("Invalid critic training mode")
        
        loss.backward()
        self.critic_optim.step()
        return loss.item()
    
    
    def eval(self,state,action):
        """
        Evaluates the critic network

        Parameters:
        -----------
        - state: The state tensor
        - action: The action tensor

        Returns:
        --------
        - q_val: The Q-value
        """

        q_val = self.critic(torch.cat([state,action],dim=1))
        return q_val
    
    
    def augmentState(self,state):
        """
        Augments the state tensor with noise for set training

        Parameters:
        -----------
        - state: The state tensor

        Returns:
        --------
        - augmented state tensor
        """

        assert state.dim() == 2

        noise = self.noise.unsqueeze(0)
        noise_batch = noise.repeat(state.size(0),1,1)
        return torch.cat([state.unsqueeze(2),noise_batch],dim=2)
    
        
    @abstractmethod
    def train_step(self):
        """Abstract function for training step specific to the algorithm"""
        pass

    @abstractmethod
    def act(self,state):
        """Abstract functio for actor evaluation and policy roll out"""
        pass

    def __getNumActionGens(self):
        """
        Returns the number of generators for the action space

        Returns:
        --------
        - num_generators: The number of generators
        """

        if self.options['critic_train_mode'] == 'set':
            num_activations = sum(self.actor[i-1].out_features for i in range(1,len(self.actor)) if isinstance(self.actor[i],(torch.nn.ReLU, torch.nn.Tanh)))
            num_generators = self.actor[0].in_features + num_activations
        else:
            num_generators = None
        return num_generators
    
    def __xavierInit(self,network):
        """
        Xavier initialization of the network, except of last layers:
        The final layer weights and biases of both the actor and critic
        were initialized from a uniform distribution [-3 x 10-3, 3 x 10-3]
        
        Parameters:
        -----------
        - network: The network to be initialized
        """

        for layer in network[:-2]:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        if isinstance(network[-2], torch.nn.Linear):
            network[-2].weight.data.uniform_(-3e-3,3e-3)
            network[-2].bias.data.uniform_(-3e-3,3e-3)
        
        if isinstance(network[-1], torch.nn.Linear):
            network[-1].weight.data.uniform_(-3e-3,3e-3)
            network[-1].bias.data.uniform_(-3e-3,3e-3)
            
    
    def __validateOptions(self,options):
        """
        Validates the option dictionary and sets default values
        
        Parameters:
        -----------
        - options: The options dictionary

        Returns:
        --------
        - options: The validated options dictionary
        """

        default_options = {
            'gamma': .99,
            'buffer_size': 10000,
            'batch_size': 64,
            'actor_lr': 0.001,
            'actor_l2': 0.01,
            'critic_lr': 0.001,
            'critic_l2': 0.0,
            'critic_train_mode': 'set',
            'critic_eta': 0.01,
            'actor_train_mode': 'set',
            'actor_eta': 0.01,
            'actor_omega': .5,
            'noise': 0.1,
            'eval_freq': 10000,
            'print_freq': 1000,
        }

        for key in default_options:
            if key not in options:
                options[key] = default_options[key]

        return options
    
        
    def __printOpionsInfo(self,steps):
        """
        Prints options of the agent
        
        Parameters:
        -----------
        - steps: Number of training steps
        """

        print("Reinforcment Learning Parameters:")
        print("=================================")
        print("Standard-RL Options:")
        print("--------------------")
        print("Discount Factor (gamma): {}".format(self.options['gamma']))
        print("Buffer Size: {}".format(self.options['buffer_size']))
        print("Batch Size: {}".format(self.options['batch_size']))
        print("Steps: {}".format(steps))
        print("Device: {}".format(self.device))

        print("\nActor Options:")
        print("--------------")
        print("Learning Rate: {}".format(self.options['actor_lr']))
        print("Training Mode: {}".format(self.options['actor_train_mode']))
        if self.options['actor_train_mode'] == 'set':
            print("Eta: {}".format(self.options['actor_eta']))
            print("Omega: {}".format(self.options['actor_omega']))
            print("Noise: {}".format(self.options['noise']))

        print("\nCritic Options:")
        print("---------------")
        print("Learning Rate: {}".format(self.options['critic_lr']))
        print("Training Mode: {}".format(self.options['critic_train_mode']))
        if self.options['critic_train_mode'] == 'set':
            print("Eta: {}".format(self.options['critic_eta']))
        print("=================================")
        print("")
        print("")

        
    def __printTrainingInfo(self,time, step,reward,q_val,critic_loss,actor_loss):
        """
        Prints the training information in each time step
        
        Parameters:
        -----------
        - time: Time of the training step
        - step: The episode number
        - reward: The total reward
        - q_val: The Q-value
        - critic_loss: The critic loss
        - actor_loss: The actor loss
        """

        if step == 1:
            print("Training Information:")
            print("=====================")
            print("|Step           |Time   |Reward         |Q-Value        |Critic-Loss    |Actor-Loss     |")
            print("|---------------|-------|---------------|---------------|---------------|---------------|")
        print("|{:.2e}\t|{:.1f}\t|{:.2e}\t|{:.2e}\t|{:.2e}\t|{:.2e}\t|".format(step,time/60,reward.item(),q_val.item(),critic_loss.item(),actor_loss.item()))