import torch
from copy import deepcopy
from ..actorcitic import ActorCritic
from ...ZonoTorch.set import Zonotope
from ...ZonoTorch.losses import ZonotopePolicyGradient
from ...ZonoTorch.losses import ZonotopeActorGradient

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
        """
        Initializes the DDPG agent

        Parameters:
        -----------
        - actor: The actor network (torch.nn.Sequential)
        - critic: The critic network (torch.nn.Sequential)
        - options: Dictionary with options for the agent
        - device: cuda (gpu) or cpu
        """

        super().__init__(actor,critic,options,device)

        self.options = self.__validateDDPGOptions(options)
        self.exploration = torch.zeros((self.action_dim,1)).to(device)
        self.target_actor = deepcopy(self.actor).to(device)
        self.target_critic = deepcopy(self.critic).to(device)
        self.actor_loss = ZonotopePolicyGradient(self.options['actor_eta'],self.options['actor_omega'],self.options['noise'])
        self.actor_vol_loss = ZonotopeActorGradient(self.options['actor_eta'],self.options['actor_omega'],self.options['noise'])


    def act(self,state):
        """
        Returns the action augmented with exploration noise (either OU or Gaussian)

        Parameters:
        -----------
        - state: The state of the environment

        Returns:
        --------
        - action: The action augmented with exploration noise        
        """

        if state.dim() == 1:
            state = state.unsqueeze(0)

        if self.options['critic_train_mode'] == 'set':
            z_state = self.augmentState(state).permute(1,2,0)
            eval_state = Zonotope(z_state)

        elif self.options['critic_train_mode'] == 'adv_naive':
            eval_state = self.naive_attack(state)
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
            action = action + self.exploration.permute(1,0)
        return torch.clamp(action,self.options['action_lb'],self.options['action_ub'])
    
    
    def eval(self, state, action):
        return super().eval(state, action)
    

    def train_step(self):
        """
        Performs train step of the actor-critic DDPG agent
        
        Returns:
        --------
        - critic_loss: The critic loss
        - actor_loss: The actor loss
        """

        states, actions, rewards, next_states, dones = self.buffer.sample(self.options['batch_size'])

        with torch.no_grad():   
            next_target_action = self.target_actor(next_states)
            target_q = self.target_critic(torch.cat([next_states,next_target_action],dim=1))
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
        """
        Trains the actor network

        Parameters:
        -----------
        - states: The states of the environment

        Returns:
        --------
        - loss: The loss of the actor
        """

        self.actor_optim.zero_grad()

        actions = self.actor(states)

        if self.options['actor_train_mode'] == 'set':
            assert isinstance(states,Zonotope)
            if self.options['critic_train_mode'] == 'set':
                q_val = self.critic(torch.functional.cartesian_prod(states,actions))
                loss = self.actor_loss(q_val) + self.actor_vol_loss(actions)
            elif self.options['critic_train_mode'] == 'point':
                action_center = actions.extractCenter().permute(2,0,1).squeeze(2)
                states_center = states.getCenter().permute(2,0,1).squeeze(2)
                q_val = self.critic(torch.cat([states_center,action_center],dim=1))
                loss = -q_val.mean() + self.actor_vol_loss(actions)
            else:
                raise ValueError('Critic training only implemented point or set.')
            
        elif self.options['actor_train_mode'] in ['point','adv_naive','adv_grad'] and self.options['critic_train_mode'] == 'point':
            q_val = self.critic(torch.cat([states,actions],dim=1))
            loss = -q_val.mean()
        elif self.options['actor_train_mode'] == 'MAD' and self.options['critic_train_mode'] == 'point':
            q_val = self.critic(torch.cat([states,actions],dim=1))
            a_loss = self.mad_attack(states)
            loss = -q_val.mean() + a_loss
        else:
            raise ValueError("Invalid critic and actor training mode combination. Possible combinations: Actor: ('point','adv_naive','adv_grad','set','set') and Critic: ('point','point','point','point','set')")
                             
        loss.backward()
        self.actor_optim.step()
        
        return loss.item()
    
    def soft_update(self,net,net_target):
        """
        Soft update of the target network
        
        Parameters:
        -----------
        - net: The network to be updated
        - net_target: The target network
        """

        for param, target_param in zip(net.parameters(),net_target.parameters()):
            target_param.data = target_param.data * (1 - self.options['tau']) + param.data * self.options['tau']
        
    
    def __validateDDPGOptions(self,options):
        """
        Validates the options for the DDPG agent
        
        Parameters:
        -----------
        - options: The options for the agent

        Returns:
        --------
        - options: The validated options
        """
        
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
    
    def mad_attack(self,states):
        """
        Performs the MAD (Maximum Action Difference) attack on the actor network [1].
        [1] H. Zhang et.al. Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations, Int. Conf. on Neural Information Processing Systems (NeurIPS) 2020 

        Parameters:
        -----------
        - states: The states of the environment

        Returns:
        --------
        - action_loss: The loss of the actor 
        """

        step_eps = self.options["noise"]/self.options["actor_adv_num_samples"]
        adv_ub = states + self.options["noise"]
        adv_lb = states - self.options["noise"]

        actions = self.actor(states)

        beta = 1e-5
        noise_factor = torch.sqrt(2*torch.tensor(step_eps))*beta
        noise = torch.randn_like(states) * noise_factor


        adv_states = (states.clone().detach() + noise.sign() * step_eps).detach().requires_grad_()

        for i in range(self.options["actor_adv_num_samples"]):
            adv_loss = (self.actor(adv_states) - actions.detach()).pow(2).mean()
            adv_loss.backward()
            noise_factor = torch.sqrt(2*torch.tensor(step_eps))*beta /(i+2)
            update = (adv_states.grad + noise_factor * adv_states.grad.sign()).detach() * step_eps
            adv_states = (adv_states + update).clamp(adv_lb, adv_ub).detach().requires_grad_()
        
        action_loss = (self.actor(adv_states) - actions.detach()).pow(2).mean()
        
        return action_loss
    
    def naive_attack(self, states):
        """
        Performs the Naive attack on the actor network [1].
        [1] Pattanaik, A. et al. 'Robust Deep Reinforcement Learning with Adversarial Attacks', Int. Conf. on Autonomous Agents and Multiagent Systems (AAMAS) 2018

        Parameters:
        -----------
        - states: The states of the environment

        Returns:
        - adv_states: The adversarial states
        """

        alpha = self.options["actor_adv_alpha"]
        beta = self.options["actor_adv_beta"]

        eps = self.options["noise"]

        action = self.actor(states)
        Q_val = self.target_critic(torch.cat([states,action],dim=1))

        states_batch = states.repeat(self.options["actor_adv_num_samples"],1)

        beta = torch.distributions.beta.Beta(torch.tensor(alpha),torch.tensor(beta))
        noise = beta.sample(states_batch.shape).to(self.device) * eps

        adv_states = states_batch + noise
        adv_actions = self.actor(adv_states)
        adv_Q_val = self.target_critic(torch.cat([adv_states,adv_actions],dim=1))

        [min_idx] = torch.argmin(adv_Q_val,dim=0)

        if adv_Q_val[min_idx] < Q_val:
            adv_states = states_batch[min_idx]
        else:
            adv_states = states_batch[0]
        return adv_states


    def grad_attack(self, states):
        """
        Performs the Gradient attack on the actor network [1].
        [1] Pattanaik, A. et al. 'Robust Deep Reinforcement Learning with Adversarial Attacks', Int. Conf. on Autonomous Agents and Multiagent Systems (AAMAS) 2018

        Parameters:
        -----------
        - states: The states of the environment

        Returns:
        - adv_states: The adversarial states
        """

        alpha = self.options["actor_adv_alpha"]
        beta = self.options["actor_adv_beta"]
        eps = self.options["noise"]

        states = states.clone().detach().requires_grad_()

        action = self.actor(states)
        Q_val = self.critic(torch.cat([states,action],dim=1))

        Q_val.backward()
        grad = states.grad.sign()

        grad_batch = grad.repeat(self.options["actor_adv_num_samples"],1)
        beta = torch.distributions.beta.Beta(torch.tensor(alpha),torch.tensor(beta))
        
        adv_states = states - beta.sample(states.shape).to(self.device) * eps * grad_batch
        adv_actions = self.actor(adv_states)
        adv_Q_val = self.critic(torch.cat([adv_states,adv_actions],dim=1))

        [min_idx] = torch.argmin(adv_Q_val,dim=0)
        if adv_Q_val[min_idx] < Q_val:
            adv_states = states[min_idx]
        else:
            adv_states = states[0]
        return adv_states