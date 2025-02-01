import torch
from .set import *
from .layers import *
from .losses import *

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

@implements(torch.tanh)
def tanh(input):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeTanh.apply(input)

@implements(torch.nn.functional.softmax)
def softmax(input, dim, **kwargs):
    """Implements the Affine Map of Zonotopes"""
    return ZonotopeSoftmax.apply(input)