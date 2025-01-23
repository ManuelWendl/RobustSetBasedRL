import torch
import functools

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    """Register a torch function override for Zonotope"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

class Zonotope(torch.Tensor):
    '''
    Zonotope: Continuous Set-Representation:
    ========================================

    Zonotopes are a continuous set representation with center c and generators G.
    
    Attributes: 
    -----------
    - _tensor: The tensor contains center c and generators G with tensor size (dims, num. Generators + 1, num samples) and concatenation [c,G]
    - _dim: first dimension of tensor
    - _numGenerators: second dimension of tensor 
    - _batchsize: third dimension of tensor (if multiple zonotopes are stored)
    '''
    def __init__(self, value):
        self._tensor = torch.as_tensor(value,dtype=torch.float)
        if self._tensor.dim() == 1:
            self._tensor = self._tensor.unsqueeze(0)
            
        if self._tensor.dim() == 3:
            self._batchSize = self._tensor.size(2)
        else:
            self._batchSize = 0

        self._dim = self._tensor.size(0)
        self._numGenerators = self._tensor.size(1)

    def __repr__(self):
        """Implements the print function"""
        return "Zonotope(center={}, generators={})".format(self._tensor[:,0,...], self._tensor[:,1:,...])
    
    def getCenter(self):
        """Returns the center(s) of the zonotope(s)"""
        return self._tensor[:,0,...].unsqueeze(1)
    
    def getGenerators(self):
        """Returns the generator(s) of the zonotope(s)"""
        return self._tensor[:,1:,...]

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Class Method for implementation of PyTorch operations"""
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, Zonotope))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
@implements(torch.add)
def add(input, other):
    """Implements the Minkowsky sum of Zonotopes"""
    if input._dim == other._dim and input._batchSize == other._batchSize:
        return Zonotope(torch.cat([input.getCenter()+other.getCenter(),input.getGenerators(),other.getGenerators()],dim=1))
    else:
        raise ValueError("Shape mismatch of added Zonotopes. Dimensions={}, Batchsizes={}".format([input._dim,other._dim],[input._batchSize,other._batchSize]))

@implements(torch.mul)
def mul(A,B,out=None):
    """Implements the Elementwise Multiplication of Zonotopes"""
    if isinstance(A,Zonotope):
        ATensor = A._tensor
    else:
        ATensor = A
    if isinstance(B,Zonotope):
        BTensor = B._tensor
    else:
        BTensor = B
    return Zonotope(torch.mul(ATensor,BTensor,out=out))

@implements(torch.tensordot)
def tensordot(A, B, dims, out=None):
    """Implements the Matrix Multiplication with a Tensor"""
    if isinstance(A,Zonotope):
        ATensor = A._tensor
    else:
        ATensor = A
    if isinstance(B,Zonotope):
        BTensor = B._tensor
    else:
        BTensor = B
    return Zonotope(torch.tensordot(ATensor,BTensor,dims=dims,out=out))

@implements(torch.cartesian_prod)
def cartesian_prod(input, other):
    """Implements the Cartesian Product of Zonotopes"""
    if input._batchSize == other._batchSize:
        diffGenerators = input._numGenerators - other._numGenerators
        if diffGenerators > 0:
            otherPadded = torch.cat([other._tensor,torch.zeros(other._dim,diffGenerators,other._batchSize)],1)
            return Zonotope(torch.cat([input._tensor,otherPadded],0))
        else:
            inputPadded = torch.cat([input._tensor,torch.zeros(input._dim,-diffGenerators,input._batchSize)],1)
            return Zonotope(torch.cat([inputPadded,other._tensor],0))
    else:
        raise ValueError("Batchsize mismatch of added Zonotope Batches.") 
    
@implements(torch.nn.functional.linear)
def linear(input,weight,bias):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeLinear.apply(input,weight,bias)

@implements(torch.nn.functional.relu)
def relu(input,inplace):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeReLU.apply(input)

@implements(torch.nn.functional.tanh)
def tanh(input):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeTanh.apply(input)

class ZonotopeLinear(torch.autograd.Function):
    """
    ZonotopeLinear: Linear Layer for Zonotopes
    ==========================================

    This class implements the linear layer for zonotopes with an affine transformation

    Functions:
    ----------
    - forward: Forward pass
    - backward: Backward propagation
    """
    @staticmethod
    def forward(ctx,input,weight,bias):
        # Save input and weight for backprop
        ctx.save_for_backward(input, weight)
        output = torch.tensordot(weight,input,dims=1)
        output = torch.add(output,Zonotope(bias.unsqueeze(1).unsqueeze(2).expand_as(output.getCenter())))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        center_Term = torch.matmul(grad_output.getCenter().squeeze(),inputs.getCenter().squeeze().t())
        input_G = torch.nn.functional.pad(inputs.getGenerators(),(0,0,0,grad_output._numGenerators-inputs._numGenerators))
        gen_Term = torch.sum(1/3*torch.einsum('ijk,ljk->ilk',grad_output.getGenerators(),input_G),dim=2)
        
        grad_input = torch.tensordot(weight.t(),grad_output,dims=1)
        grad_weight = center_Term + gen_Term
        grad_bias = torch.sum(torch.sum(grad_output.getCenter(),dim=1),dim=1)
        return grad_input, grad_weight, grad_bias
    
class ZonotopeReLU(torch.autograd.Function):
    """
    ZonotopeReLU: ReLU layer for Zonotopes 
    ======================================

    This class implemets the ReLU layer for zonotopes using the fast image enclosure

    Functions:
    ----------
    - forward: Forward pass
    - backward: Backward propoagation
    """
    @staticmethod
    def forward(ctx,input):
        c = input.getCenter()
        G = input.getGenerators()

        u = c + torch.abs(G).sum(1).unsqueeze(1)
        l = c - torch.abs(G).sum(1).unsqueeze(1)

        m = torch.zeros(l.shape,device=c.device)
        m[(l>0)*(u>0)] = 1
        m[(l<0)*(u>0)] = u[(l<0)*(u>0)]/(u[(l<0)*(u>0)]-l[(l<0)*(u>0)])

        # Save slope for backprop
        ctx.save_for_backward(m)

        t = torch.zeros(l.shape,device=c.device)
        t[(l<0)*(u>0)] = (-l[(l<0)*(u>0)]*u[(l<0)*(u>0)])/(2*(u[(l<0)*(u>0)]-l[(l<0)*(u>0)]))
        d = (t.permute(2,0,1)*torch.eye(t.size(0),device=c.device)).permute(1,2,0)
        
        output = torch.add(torch.mul(m,input),Zonotope(torch.cat([t,d],1)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.saved_tensors[0]
        grad_input = torch.mul(m,grad_output)
        grad_input = Zonotope(grad_input._tensor[:,0:-grad_input._dim,...])
        return grad_input
    
class ZonotopeTanh(torch.autograd.Function):
    """
    ZonontopeTanh: Tanh layer for Zonotopes
    =======================================

    This class implemets the Tanh layer for zonotopes using the fast image enclosure

    Functions:
    ----------
    - forward: Forward pass
    - backward: Backward Propagation
    """
    @staticmethod
    def forward(ctx,input):
        c = input.getCenter()
        G = input.getGenerators()

        u = c + torch.abs(G).sum(1).unsqueeze(1)
        l = c - torch.abs(G).sum(1).unsqueeze(1)

        m = (torch.tanh(u)-torch.tanh(l))/(u-l)
        ctx.save_for_backward(m)

        t = 1/(u-l)*torch.log(torch.cosh(u)/torch.cosh(l))-m*c

        # Find approximation error
        x1 = torch.arctanh(torch.sqrt(1-m))
        x2 = -x1
        x1[x1>u] = u[x1>u]
        x1[x1<l] = l[x1<l]
        x2[x2>u] = u[x2>u]
        x2[x2<l] = l[x2<l]
        x = torch.stack([l,u,x1,x2],dim=1)

        d = torch.max(torch.abs(torch.tanh(x)-(m*x+t)),axis=1)
        d = (t.permute(2,0,1)*torch.eye(t.size(0))).permute(1,2,0)

        output = torch.add(torch.mul(m,input),Zonotope(torch.cat([t,d],1)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.saved_tensors[0]
        grad_input = torch.mul(m,grad_output)
        return grad_input
    
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
        mask = radius>0
        radius[radius==0] = 1
        gradient_generators = 1/output._batchSize*mask*ctx.eta/ctx.noise*torch.sign(output.getGenerators())/radius
        gradient = Zonotope(torch.cat([gradient_center,gradient_generators],dim=1))
        return gradient, None, None, None

class ZonotopeRegressionLoss(torch.nn.Module):
    """Implements the set-base regression loss"""
    def __init__(self, eta, noise):
        super().__init__()
        self.eta = eta
        self.noise = noise

    def forward(self, output, target):
        loss = ZRL.apply(output, target, self.eta, self.noise)
        return loss
