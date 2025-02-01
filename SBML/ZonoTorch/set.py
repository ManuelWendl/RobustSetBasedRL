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
        self._tensor = torch.as_tensor(value,dtype=value.dtype)
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
    
def implements(torch_function):
    """Register a torch function override for Zonotope"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator