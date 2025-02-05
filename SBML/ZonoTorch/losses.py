import torch
from .set import Zonotope

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
        gradient_generators = 1/output._batchSize*ctx.eta/ctx.noise*torch.sign(output.getGenerators())/radius
        gradient_generators[gradient_generators.isnan()] = 0
        gradient_generators = torch.clamp(gradient_generators,-1e6,1e6)
        gradient = Zonotope(torch.cat([gradient_center,gradient_generators],dim=1))
        return gradient, None, None, None
            
class ZPG(torch.autograd.Function):
    """
    ZPG (Zonotope Policy Gradient):

    Implements a set-based policy gradient loss that penalizes both the critic’s output (via its zonotope center)
    and the “size” of the zonotopes (via the generators). The gradients w.r.t. the center and generators are computed independently.
    
    Note: To backpropagate through the critic branch (i.e. through q_value), we need to be able to re-run the critic.
          In this example, we assume that `states` and `critic` do not require custom gradient treatment and that
          actions (a Zonotope) is the only argument we treat specially.
    """
    
    @staticmethod
    def forward(ctx, action, q_value, eta, omega, noise):
        
        ctx.save_for_backward(action, q_value)
        ctx.eta = eta
        ctx.omega = omega
        ctx.noise = noise
        
        if isinstance(q_value, Zonotope):
            GLoss = (eta/noise) * (
                        (1-omega)*torch.log(2*torch.sum(torch.abs(action.getGenerators()))) +
                        omega*torch.log(2*torch.sum(torch.abs(q_value.getGenerators())))
                    )
            q = torch.reshape(q_value.getCenter(), (-1, 1))
        else:
            q = q_value
            GLoss = (eta/noise) * torch.log(2*torch.sum(torch.abs(action.getGenerators())))
        
        cLoss = -q
        GLoss = torch.where(GLoss < -1000, torch.full_like(GLoss, -1000), GLoss)
        loss = (1 / action._batchSize) * torch.sum(cLoss + GLoss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation for the ZPG loss.

        We must return gradients for each input to forward:
          - grad_actions: gradients for the actor’s zonotope (Zonotope underlying tensor)
          - None for states, critic, eta, omega, and noise (assuming we do not update them with this loss)
        """
        action, q_value = ctx.saved_tensors
        eta = ctx.eta
        omega = ctx.omega
        noise = ctx.noise
        B = action._batchSize  
        
        if isinstance(q_value, Zonotope):
            grad_center = -1 / B * torch.ones_like(q_value.getCenter())
            radius_q = torch.sum(torch.abs(q_value.getGenerators()))
            grad_q_gen = (1 / B)*(eta/noise)*omega * (torch.sign(q_value.getGenerators()) / radius_q)
            grad_q_gen[grad_q_gen.isnan()] = 0
            grad_q_gen = torch.clamp(grad_q_gen, -1e6, 1e6)
            grad_q = Zonotope(torch.cat([grad_center, grad_q_gen], dim=1))
        else:
            grad_q = -1 / B * torch.ones_like(q_value)
        

        A_gens = action.getGenerators()
        radius_a = torch.sum(torch.abs(A_gens))
        grad_A_gens = (1 / B)*(eta/noise)*(1-omega) * (torch.sign(A_gens) / radius_a)
        grad_A_gens[grad_A_gens.isnan()] = 0
        grad_A_gens = torch.clamp(grad_A_gens, -1e6, 1e6)
        grad_A_center = torch.zeros_like(action.getCenter())
        grad_action = Zonotope(torch.cat([grad_A_center, grad_A_gens], dim=1))

        return grad_action, grad_q, None, None, None

        
        