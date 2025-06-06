import torch
import functools

HANDLED_FUNCTIONS = {}

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
        assert isinstance(value, torch.Tensor); "Zonotope must be initialized with a torch.Tensor"

        if value.requires_grad:
            self._tensor = torch.as_tensor(value,dtype=value.dtype,device=value.device).requires_grad_()
        else:
            self._tensor = torch.as_tensor(value,dtype=value.dtype,device=value.device)
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
    
    def tensor(self):
        return self._tensor
    
    def getCenter(self):
        """Returns the center(s) of the zonotope(s)"""
        return self._tensor[:,0,...].unsqueeze(1)
    
    def getGenerators(self):
        """Returns the generator(s) of the zonotope(s)"""
        return self._tensor[:,1:,...]
    
    def extractCenter(self):
        """Extracts the center of the zonotope"""
        return ZonotopeGetCenter.apply(self)

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
    
class Interval:
    '''
    Interval: Continuous Set-Representation:
    ========================================

    Intervals are a continuous set representation with lower bound l and upper bound u.
    
    Attributes: 
    -----------
    - _tensor: The tensor contains lower bound l and upper bound u with tensor size (dims, 2, num samples) and concatenation [l,u]
    - _dim: first dimension of tensor
    - _batchsize: third dimension of tensor (if multiple zonotopes are stored)
    '''
    def __init__(self, value):
        self._tensor = torch.as_tensor(value,dtype=value.dtype)
        if self._tensor.dim() == 1:
            self._tensor = self._tensor.unsqueeze(0)
            
        if self._tensor.dim() == 3:
            self._batchSize = self._tensor.size(2)
        else:
            self._batchSize = 0

        self._dim = self._tensor.size(0)

    def __repr__(self):
        """Implements the print function"""
        return "Interval(lower={}, upper={})".format(self._tensor[:,0,...], self._tensor[:,1,...])
    
    def getLower(self):
        """Returns the lower bound(s) of the interval(s)"""
        return self._tensor[:,0,...]
    
    def getUpper(self):
        """Returns the upper bound(s) of the interval(s)"""
        return self._tensor[:,1,...]
    
    
def implements(torch_function):
    """Register a torch function override for Zonotope"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

class ZonotopeGetCenter(torch.autograd.Function):
    """
    ZonotopeGetCenter: Get Center of Zonotope
    ========================================
    
    This class implements the operation to get the center of a zonotope.
    
    Functions:
    ----------
    - forward: Forward pass
    - backward: Backward pass
    """
    @staticmethod
    def forward(ctx, input):
        """Forward pass of the Get Center operation"""
        ctx.save_for_backward(input)
        return input._tensor[:, 0, ...].unsqueeze(1)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the Get Center operation"""
        input, = ctx.saved_tensors
        center_grad = grad_output 
        generator_grad = torch.zeros_like(input._tensor[:, 1:, ...])
        grad_output = Zonotope(torch.cat([center_grad, generator_grad], dim=1))
        return grad_output