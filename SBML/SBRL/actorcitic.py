from abc import ABC, abstractmethod
from time import time
import torch

from .buffer import Buffer
from ..ZonoTorch.set import Zonotope
from ..ZonoTorch.losses import ZonotopeRegressionLoss


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
        self.device = device
        self.options = self.__validateOptions(options)

        assert isinstance(actor, torch.nn.Sequential)
        #self.__xavierInit(actor)
        self.actor = actor.to(device)

        assert isinstance(critic, torch.nn.Sequential)
        #self.__xavierInit(critic)
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

        self.buffer = Buffer(options['buffer_size'],device,self.state_dim,self.action_dim,num_generators)
    
    def train(self,env,episodes=1000,verbose=True):
        """Trains the actor-critic agent"""
        self.buffer.reset()
        
        if self.options['actor_train_mode'] == 'set':
            self.noise = torch.eye(self.actor[0].in_features).to(self.device) * self.options['noise']

        self.__printOpionsInfo(episodes)

        t_start = time()

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            total_critic_loss = 0
            total_actor_loss = 0
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

                num_step += 1

                if self.buffer.full or self.buffer.indx > self.options['batch_size']:
                    critic_loss, actor_loss = self.train_step()
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss

            if verbose:
                self.__printTrainingInfo(time()-t_start,episode,total_reward,q_val,total_critic_loss/num_step, total_actor_loss/num_step)

    def train_critic(self,state,action,target):
        """Trains the critic network"""
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
        """Evaluates the critic for the first timestep"""
        q_val = self.critic(torch.cat([state,action],dim=1))
        return q_val
    
    def augmentState(self,state):
        """Augments the state with noise for set training"""
        assert state.dim() == 2
        noise = self.noise.unsqueeze(0)
        noise_batch = noise.repeat(state.size(0),1,1)
        return torch.cat([state.unsqueeze(2),noise_batch],dim=2)
        
    @abstractmethod
    def train_step(self):
        """Abstract wrapper for training step specific to the algorithm"""
        pass

    @abstractmethod
    def act(self,state):
        """Abstract wrapper for actor evaluation and policy roll out"""
        pass

    def __xavierInit(self,model):
        """Xavier initialization of the model"""
        for layer in model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def __getNumActionGens(self):
        """Returns the number of generators of the action zonotope"""
        if self.options['critic_train_mode'] == 'set':
            num_activations = sum(self.actor[i-1].out_features for i in range(1,len(self.actor)) if isinstance(self.actor[i],(torch.nn.ReLU, torch.nn.Tanh)))
            num_generators = self.actor[0].in_features + num_activations
        else:
            num_generators = None
        return num_generators
    
    def __validateOptions(self,options):
        """Validates the option dictionary and sets default values"""
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
            'noise': 0.1
        }
        for key in default_options:
            if key not in options:
                options[key] = default_options[key]
        return options
        
    def __printOpionsInfo(self,episodes):
        """Prints the options of the agent"""
        print("Reinforcment Learning Parameters:")
        print("=================================")
        print("Standard-RL Options:")
        print("--------------------")
        print("Discount Factor (gamma): {}".format(self.options['gamma']))
        print("Buffer Size: {}".format(self.options['buffer_size']))
        print("Batch Size: {}".format(self.options['batch_size']))
        print("Episodes: {}".format(episodes))
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
        
    def __printTrainingInfo(self,time, episode,reward,q_val,critic_loss,actor_loss):
        """Prints the training information"""
        if episode == 0:
            print("Training Information:")
            print("=====================")
            print("|Episode\t|Elapsed Time\t|Reward\t|Q-Value\t|Critic-Loss\t|Actor-Loss")
            print("|-------\t|----------\t|----------\t|-----------\t|-----------\t|----------")
        print("|{}\t|{}\t|{}\t|{}\t|{}\t|{}".format(episode,time,reward,q_val.item(),critic_loss,actor_loss))