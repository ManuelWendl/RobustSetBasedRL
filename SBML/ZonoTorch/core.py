import torch
from .set import *
from .layers import *
from .losses import *

@implements(torch.add)
def add(input, other):
    """
    Implements the Minkowsky sum of Zonotopes

    Parameters:
    -----------
    - input: First Zonotope
    - other: Second Zonotope

    Returns:
    --------
    - Zonotope: The Minkowsky sum of the two Zonotopes
    """

    if isinstance(input,Zonotope) and isinstance(other,Zonotope):
        if input._dim == other._dim and input._batchSize == other._batchSize:
            return Zonotope(torch.cat([input.getCenter()+other.getCenter(),input.getGenerators(),other.getGenerators()],dim=1))
        else:
            raise ValueError("Shape mismatch of added Zonotopes. Dimensions={}, Batchsizes={}".format([input._dim,other._dim],[input._batchSize,other._batchSize]))
        

@implements(torch.mul)
def mul(A,B,out=None):
    """
    Implements the Elementwise Multiplication of Zonotopes
    
    Parameters:
    -----------
    - A: First Zonotope
    - B: Second Zonotope

    Returns:
    --------
    - Zonotope: The Elementwise Multiplication of the two Zonotopes
    """
    
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
    """
    Implements the Linear Map with a Tensor
    
    Parameters:
    -----------
    - A: First Zonotope
    - B: Second Zonotope
    - dims: Dimensions to contract

    Returns:
    --------
    - Zonotope: The Linear Map of a Zonotope with a Tensor
    """

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
    return ZonotopeCartesian.apply(input,other)
    
@implements(torch.clamp)
def clamp(input, min, max):
    """Implements the Clamping of Zonotopes. Only clamp the zonotope center"""
    return Zonotope(torch.cat([torch.clamp(input.getCenter(),min,max),input.getGenerators()],dim=1))
    
@implements(torch.nn.functional.linear)
def linear(input,weight,bias):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeLinear.apply(input,weight,bias)

@implements(torch.nn.functional.relu)
def relu(input,inplace):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeReLU.apply(input)

@implements(torch.tanh)
def tanh(input):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeTanh.apply(input)

@implements(torch.nn.functional.softmax)
def softmax(input, dim, **kwargs):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeSoftmax.apply(input)