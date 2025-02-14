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

    def forward(self, q_val):
        loss = ZPG.apply(q_val, self.eta, self.omega, self.noise)
        return loss
    
class ZonotopeActorGradient(torch.nn.Module):
    """implements the set-based actor gradient"""
    def __init__(self, eta, omega, noise):
        super().__init__()
        self.eta = eta
        self.omega = omega
        self.noise = noise

    def forward(self, action):
        loss = ZAG.apply(action, self.eta, self.omega, self.noise)
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
    
def radius(zonotope):
        return torch.sum(torch.abs(zonotope.getGenerators()))
            
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
    def forward(ctx, q_value, eta, omega, noise):
        
        ctx.save_for_backward(q_value)
        ctx.eta = eta
        ctx.omega = omega
        ctx.noise = noise
        
        GLoss = (eta/noise) * omega*torch.log(2*radius(q_value)).reshape(-1, 1)
        q = torch.reshape(q_value.getCenter(), (-1, 1))
        
        cLoss = -q
        GLoss = torch.where(GLoss < -1000, torch.full_like(GLoss, -1000), GLoss)
        loss = (1 / q_value._batchSize) * torch.sum(cLoss + GLoss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation for the ZPG loss.

        We must return gradients for each input to forward:
          - grad_actions: gradients for the actor’s zonotope (Zonotope underlying tensor)
          - None for states, critic, eta, omega, and noise (assuming we do not update them with this loss)
        """
        q_value, = ctx.saved_tensors
        eta = ctx.eta
        omega = ctx.omega
        noise = ctx.noise
        B = q_value._batchSize  
        
        grad_center = -1 / B * torch.ones_like(q_value.getCenter())
        grad_q_gen = (1 / B)*(eta/noise)*omega * (torch.sign(q_value.getGenerators()) / radius(q_value))
        grad_q_gen[grad_q_gen.isnan()] = 0
        grad_q_gen = torch.clamp(grad_q_gen, -1e6, 1e6)
        z_grad_q = torch.cat([grad_center, grad_q_gen], dim=1)
        
        grad_q = Zonotope(z_grad_q)

        return grad_q, None, None, None, None

class ZAG(torch.autograd.Function):
    """
    ZAG (Zonotope Actor Gradient):

    Implements a set-based actor gradient loss that penalizes the actor’s output.
    """

    @staticmethod
    def forward(ctx, action, eta, omega, noise):
        ctx.save_for_backward(action)
        ctx.eta = eta
        ctx.omega = omega
        ctx.noise = noise

        GLoss = (eta/noise) * (1-omega)*torch.log(2*radius(action)).reshape(-1, 1)

        GLoss = torch.where(GLoss < -1000, torch.full_like(GLoss, -1000), GLoss)
        loss = (1 / action._batchSize) * torch.sum(GLoss)
        return loss
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation for the ZAG loss.

        We must return gradients for each input to forward:
          - grad_actions: gradients for the actor’s zonotope (Zonotope underlying tensor)
          - None for states, eta, omega, and noise (assuming we do not update them with this loss)
        """
        action, = ctx.saved_tensors
        eta = ctx.eta
        omega = ctx.omega
        noise = ctx.noise
        B = action._batchSize

        grad_a_gen = (1 / B)*(eta/noise)* (1-omega) * (torch.sign(action.getGenerators()) / radius(action))
        grad_a_gen[grad_a_gen.isnan()] = 0
        grad_a_gen = torch.clamp(grad_a_gen, -1e6, 1e6)
        z_grad_a = torch.cat([torch.zeros_like(action.getCenter()), grad_a_gen], dim=1)
        
        grad_a = Zonotope(z_grad_a)

        return grad_a, None, None, None
        