import torch
from .core import Zonotope

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
        m = torch.where(m.isnan(), torch.tensor(0.0, device=m.device, dtype=m.dtype), m)
        ctx.save_for_backward(m)

        # Find approximation error
        x1 = torch.arctanh(torch.sqrt(1-m))
        x2 = -x1
        x1[x1>u] = u[x1>u]
        x1[x1<l] = l[x1<l]
        x2[x2>u] = u[x2>u]
        x2[x2<l] = l[x2<l]

        x = torch.cat([l,u,x1,x2],dim=1)

        du = torch.max(torch.tanh(x)-(m*x),axis=1).values.unsqueeze(1)
        dl = torch.min(torch.tanh(x)-(m*x),axis=1).values.unsqueeze(1)
        t = 1/2*(du+dl)
        d = 1/2*(du-dl)

        output = torch.add(torch.mul(m,input),Zonotope(torch.cat([t,d],1)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.saved_tensors[0]
        grad_input = torch.mul(m,grad_output)
        grad_input = Zonotope(grad_input._tensor[:,0:-grad_input._dim,...])
        return grad_input
    
class ZonotopeSoftmax(torch.autograd.Function):
    """
    ZonotopeSoftmax: Softmax layer for Zonotopes
    ===========================================

    This class implemets the Softmax layer for zonotopes using the fast image enclosure

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

        m = (torch.nn.functional.softmax(u,dim=0)-torch.nn.functional.softmax(l,dim=0))/(u-l)
        m = torch.where(m.isnan(), torch.tensor(0.0, device=m.device), m)
        ctx.save_for_backward(m)

        t = 1/2*(1/2*(torch.nn.functional.softmax(u,dim=0)+torch.nn.functional.softmax(l,dim=0))-m*c)
        
        output = torch.add(torch.mul(m,input),Zonotope(torch.cat([t,torch.zeros(t.size(0),t.size(0),t.size(2),device=c.device)],1)))
        return output
    

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.saved_tensors[0]
        grad_input = torch.mul(m,grad_output)
        grad_input = Zonotope(grad_input._tensor[:,0:-grad_input._dim,...])
        return grad_input
    
class ZonotopeCartesian(torch.autograd.Function):
    """
    ZonotopeCartesian: Cartesian Product of Zonotopes
    ================================================
    
    This class implements the Cartesian Product of Zonotopes with gradient preservation.
    
    Functions:
    ----------
    - forward: Forward pass
    - backward: Backward pass
    """
    @staticmethod
    def forward(ctx, input, other):
        """Forward pass of the Cartesian Product"""
        if input._batchSize != other._batchSize:
            raise ValueError("Batchsize mismatch of added Zonotope Batches.")
        diffGenerators = input._numGenerators - other._numGenerators
        if diffGenerators > 0:
            otherPadded = torch.cat([
                other._tensor, 
                torch.zeros(other._dim, diffGenerators, other._batchSize, 
                            device=other._tensor.device, dtype=other._tensor.dtype, 
                            requires_grad=other._tensor.requires_grad)
            ], dim=1)
            output = Zonotope(torch.cat([input._tensor, otherPadded], dim=0))
        else:
            inputPadded = torch.cat([
                input._tensor, 
                torch.zeros(input._dim, -diffGenerators, input._batchSize, 
                            device=input._tensor.device, dtype=input._tensor.dtype, 
                            requires_grad=input._tensor.requires_grad)
            ], dim=1)
            output = Zonotope(torch.cat([inputPadded, other._tensor], dim=0))
        ctx.save_for_backward(input, other)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the Cartesian Product"""
        input, other = ctx.saved_tensors
        z_grad_input = grad_output._tensor[:input._dim, :input._numGenerators+1, :]
        z_grad_other = grad_output._tensor[input._dim:, :other._numGenerators+1:, :]
        return Zonotope(z_grad_input), Zonotope(z_grad_other)