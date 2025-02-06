import torch
from copy import deepcopy
from ..actorcitic import ActorCritic
from ...ZonoTorch.set import Zonotope
from ...ZonoTorch.losses import ZonotopePolicyGradient

class DDPG(ActorCritic):
    """
    DDPG: Deep Deterministic Policy Gradient
    ========================================
    
    DDPG is an actor-critic algorithm that uses a deterministic policy.
    
    Attributes:
    -----------
    - actor: The actor network (torch.nn.Sequential)
    - critic: The critic network (torch.nn.Sequential)
    - target_actor: The target actor network (torch.nn.Sequential)
    - target_critic: The target critic network (torch.nn.Sequential)
    - options: Dictionary with options for the agent
        - gamma: .99 (default) - Discount factor
        - buffer_size: 10000 (default) - Size of the replay buffer
        - batch_size: 64 (default) - Batch size for training
        - exp_noise: 0.2 (default) - Noise for exploration
        - exp_noise_type: 'ou' (default) - Noise type for exploration
        - exp_noise_mu: 0 (default) - Mu for OU noise 
        - exp_noise_theta: 0.15 (default) - Theta for OU noise ('ou' or 'gaussian')
        - action_ub: 1 (default) - Upper bound for the action
        - action_lb: -1 (default) - Lower bound for the action
        - actor_lr: 0.001 (default) - Learning rate of the actor
        - actor_l2: 0.01 (default) - L2 regularization of the actor
        - critic_lr: 0.001 (default) - Learning rate of the critic
        - critic_l2: 0.0 (default) - L2 regularization of the critic
        - exp_noise: 0.2 (default) - Noise for exploration
        - critic_train_mode: 'set' (default) - Training mode of the critic ('set' or 'point')
        - criic_eta: 0.01 (default) - Eta weighting factor for set training
        - actor_train_mode: 'set' (default) - Training mode of the actor ('set', 'point', 'adv_grad', 'adv_naive')
        - actor_eta: 0.01 (default) - Eta weighting factor for set training
        - actor_omega: .5 (default) - Omega weighting factor for set training
        - noise: 0.1 (default) - Noise for set training
    - device: cuda (gpu) or cpu
    - buffer: Replay buffer for the agent
    """
    
    def __init__(self,actor,critic,options,device):
        super().__init__(actor,critic,options,device)
        self.options = self.__validateDDPGOptions(options)
        self.exploration = torch.zeros((self.action_dim,1)).to(device)
        self.target_actor = deepcopy(self.actor).to(device)
        self.target_critic = deepcopy(self.critic).to(device)
        self.actor_loss = ZonotopePolicyGradient(self.options['actor_eta'],self.options['actor_omega'],self.options['noise'])

    def act(self,state):
        """Returns the action for a given state acting with the environment and gaining experience"""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if self.options['critic_train_mode'] == 'set':
            z_state = self.augmentState(state).permute(1,2,0)
            eval_state = Zonotope(z_state)

        elif self.options['critic_train_mode'] == 'adv_naive':
            raise NotImplementedError("Advantage Naive Training not implemented yet.")
        elif self.options['critic_train_mode'] == 'adv_grad':
            raise NotImplementedError("Advantage Gradient Training not implemented yet.")
        else:
            eval_state = state

        if self.options['exp_noise_type'] == 'ou':
            self.exploration = self.exploration + self.options['exp_noise'] * (self.options['exp_noise_mu'] - self.exploration) + self.options['exp_noise_theta'] * torch.randn_like(self.exploration) 
        elif self.options['exp_noise_type'] == 'gaussian':
            self.exploration = torch.randn_like(self.exploration) * self.options['exp_noise']
        else:
            raise ValueError("Invalid Exploration Noise Type. Choose 'ou' or 'gaussian'.")

        with torch.no_grad():
            action = self.actor(eval_state)
        if self.options['critic_train_mode'] == 'set':
            action._tensor[:,0,...] = action._tensor[:,0,...] + self.exploration
        else:
            action = action + self.exploration
        return torch.clamp(action,self.options['action_lb'],self.options['action_ub'])
    
    def eval(self, state, action):
        return super().eval(state, action)
    

    def train_step(self):
        """Performs train step of the actor-critic DDPG agent"""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.options['batch_size'])

        with torch.no_grad():   
            next_target_action = self.target_actor(next_states)
            target_q = self.target_critic(torch.cat([next_states,next_target_action],dim=1))
            if target_q.isnan().any():
                raise ValueError("NaN in target Q values.")
            target = rewards + self.options['gamma'] * target_q * (1 - dones)

        critic_loss = self.train_critic(states,actions,target)

        if self.options['actor_train_mode'] == 'set':
            z_states = self.augmentState(states.clone().detach()).permute(1,2,0)
            eval_states = Zonotope(z_states)
        else:
            eval_states = states.clone().detach()

        actor_loss = self.train_actor(eval_states)

        self.soft_update(self.actor,self.target_actor)
        self.soft_update(self.critic,self.target_critic)

        return critic_loss, actor_loss

        
    def train_actor(self,states):
        """Trains the actor network"""
        self.actor_optim.zero_grad()
        actions = self.actor(states)
        if self.options['actor_train_mode'] == 'set':
            assert isinstance(states,Zonotope)
            if self.options['critic_train_mode'] == 'set':
                q_val = self.critic(torch.functional.cartesian_prod(states,actions))
            elif self.options['critic_train_mode'] == 'point':
                action_center = actions.getCenter().reshape(-1,actions._dim)
                states_center = states.getCenter().reshape(-1,states._dim)
                q_val = self.critic(torch.cat([states_center,action_center],dim=1))
            else:
                raise ValueError('Critic training only implemented point or set.')
            
            loss = self.actor_loss(q_val)

        elif self.options['actor_train_mode'] in ['point','adv_naive','adv_grad']:
            q_val = self.critic(torch.cat([states,actions],dim=1))
            loss = -q_val.mean()
        else:
            raise ValueError("Invalid critic training mode")
                             
        loss.backward()
        self.actor_optim.step()
        
        return loss.item()
    
    def soft_update(self,net,net_target):
        """Soft update of the target network"""
        for param, target_param in zip(net.parameters(),net_target.parameters()):
            target_param.data = target_param.data * (1 - self.options['tau']) + param.data * self.options['tau']
        
    
    def __validateDDPGOptions(self,options):
        """Validates the options for the DDPG agent"""
        default_options = {
            'exp_noise': 0.2,
            'exp_noise_type': 'ou',
            'exp_noise_mu': 0,
            'exp_noise_theta': 0.15,
            'action_ub': 1,
            'action_lb': -1,
            'actor_adv_num_samples': 10,
            'actor_adv_alpha': 4,
            'actor_adv_beta': 4
        }

        for key in default_options.keys():
            if key not in options:
                options[key] = default_options[key] 
        return options