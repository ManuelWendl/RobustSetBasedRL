import torch
from .core import Zonotope, cartesian_prod

class ZonotopeRegressionLoss(torch.nn.Module):
    """Implements the set-base regression loss"""
    def __init__(self, eta, noise):
        super().__init__()
        self.eta = eta
        self.noise = noise

    def forward(self, output, target):
        loss = ZRL.apply(output, target, self.eta, self.noise)
        return loss

class ZonotopeClassificationLoss(torch.nn.Module):
    """Implements the set-base classification loss"""
    def __init__(self, eta, noise):
        super().__init__()
        self.eta = eta
        self.noise = noise

    def forward(self, output, target):
        loss = ZCL.apply(output, target, self.eta, self.noise)
        return loss
    
class ZonotopePolicyGradient(torch.nn.Module):
    """Implements the set-base regression loss"""
    def __init__(self, eta, omega, noise):
        super().__init__()
        self.eta = eta
        self.omega = omega
        self.noise = noise

    def forward(self, action, q_val):
        loss = ZPG.apply(action, q_val, self.eta, self.omega, self.noise)
        return loss

class ZRL(torch.autograd.Function):
    """
    ZRL (Zonotope Regeression Loss): 
    ===============================

    Implementation of set-based regression loss for zonotopes. 
    It simultaneously trains the half-sqarred regression loss and penalizes the size of the zonotope.

    Functions:
    - forward: Computation of set-based regression loss with parameters eta and noise
    - backward: Backward propagation
    """
    @staticmethod
    def forward(ctx, output, target, eta, noise):
        ctx.save_for_backward(output,target)
        ctx.eta = eta
        ctx.noise = noise
        cLoss = 1/2*(output.getCenter()-target._tensor)**2
        GLoss = eta/noise*torch.log(2*torch.sum(torch.abs(output.getGenerators())))
        if (GLoss<-1000).any():
            GLoss[GLoss<-1000] = -1000
        loss = 1/output._batchSize*torch.sum(cLoss+GLoss)
        return loss
    
    @staticmethod
    def backward(ctx,loss):
        output,target = ctx.saved_tensors
        gradient_center = 1/output._batchSize*(output.getCenter()-target._tensor)
        radius = torch.abs(output.getGenerators()).sum(1).unsqueeze(1)
        gradient_generators = 1/output._batchSize*ctx.eta/ctx.noise*torch.sign(output.getGenerators())/radius
        gradient_generators[gradient_generators.isnan()] = 0
        gradient_generators = torch.clamp(gradient_generators,-1e6,1e6)
        gradient = Zonotope(torch.cat([gradient_center,gradient_generators],dim=1))
        return gradient, None, None, None
    
class ZCL(torch.autograd.Function):
    """"
    ZCL (Zonotope Classification Loss): 
    ===============================

    Implementation of set-based classification loss for zonotopes. 
    It simultaneously trains the cross entropy classification loss and penalizes the size of the zonotope.

    Functions:
    - forward: Computation of set-based classification loss with parameters eta and noise
    - backward: Backward propagation
    """
    @staticmethod
    def forward(ctx, output, target, eta, noise):
        ctx.save_for_backward(output,target)
        ctx.eta = eta
        ctx.noise = noise
        output_center = torch.reshape(output.getCenter(),(-1,output._dim))
        cLoss = 1/output._batchSize*torch.nn.functional.cross_entropy(output_center,target._tensor.squeeze())
        GLoss = eta/noise*torch.log(2*torch.sum(torch.abs(output.getGenerators())))
        if (GLoss<-1000).any():
            GLoss[GLoss<-1000] = -1000
        loss = 1/output._batchSize*(cLoss+GLoss)
        return loss
    
    @staticmethod
    def backward(ctx,loss):
        output,target = ctx.saved_tensors
        output_center = torch.reshape(output.getCenter(),(-1,output._dim))
        gradient_center = (torch.softmax(output_center,dim=1)-torch.nn.functional.one_hot(target._tensor.squeeze(), num_classes=output._dim)).reshape(output._dim,1,-1)
        radius = torch.abs(output.getGenerators()).sum(1).unsqueeze(1)
        gradient_generators = 1/output._batchSize*ctx.eta*ctx.noise*torch.sign(output.getGenerators())/radius
        gradient_generators[gradient_generators.isnan()] = 0
        gradient_generators = torch.clamp(gradient_generators,-1e6,1e6)
        gradient = Zonotope(torch.cat([gradient_center,gradient_generators],dim=1))
        return gradient, None, None, None
            
class ZPG(torch.autograd.Function):
    """
    ZPG (Zonotope Policy Gradient):
    ===============================

    Implementation of set-based policy gradient loss for zonotopes.
    It simultaneously trains the policy gradient loss and penalizes the size of the zonotope.

    Functions:
    - forward: Computation of set-based policy gradient loss with parameters eta, omega and noise
    - backward: Backward propagation
    """

    @staticmethod
    def forward(ctx, actions, states, critic, eta, omega, noise):
        ctx.save_for_backward(actions)
        ctx.eta = eta
        ctx.omega = omega
        ctx.noise = noise

        q_value = critic(cartesian_prod(states,actions))

        if isinstance(q_value,Zonotope):
            GLoss = eta/noise*((1-omega)*torch.log(2*torch.sum(torch.abs(actions.getGenerators()))) + omega*torch.log(2*torch.sum(torch.abs(q_value.getGenerators()))))
            q = torch.reshape(q_value.getCenter(),(-1,1))
        else:
            q = q_value
            GLoss = eta/noise*torch.log(2*torch.sum(torch.abs(actions.getGenerators())))
        cLoss = -q
        if (GLoss<-1000).any():
            GLoss[GLoss<-1000] = -1000
        loss = 1/actions._batchSize*torch.sum(cLoss+GLoss)
        return loss

    # TODO Implement backward
    @staticmethod
    def backward(ctx, grad_output):
        actions, states = ctx.saved_tensors
        critic = ctx.critic
        eta = ctx.eta
        omega = ctx.omega
        noise = ctx.noise

        # Compute gradients for actions
        q_value = critic(cartesian_prod(states, actions))
        if isinstance(q_value, Zonotope):
            dGLoss_dActions = eta / noise * ((1 - omega) * (1 / torch.sum(torch.abs(actions.getGenerators()))) * torch.sign(actions.getGenerators()))
            dGLoss_dQ = eta / noise * omega * (1 / torch.sum(torch.abs(q_value.getGenerators()))) * torch.sign(q_value.getGenerators())
            dQ_dActions = torch.autograd.grad(q_value.getCenter(), actions, grad_outputs=dGLoss_dQ, retain_graph=True)[0]
            dLoss_dActions = -1 / actions._batchSize * (1 + dGLoss_dActions + dQ_dActions)
        else:
            dGLoss_dActions = eta / noise * (1 / torch.sum(torch.abs(actions.getGenerators()))) * torch.sign(actions.getGenerators())
            dLoss_dActions = -1 / actions._batchSize * (1 + dGLoss_dActions)

        return dLoss_dActions, None, None, None, None, None
        