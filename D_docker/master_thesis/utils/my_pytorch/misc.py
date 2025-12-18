import torch
import torch.nn as nn
from typing import List,Callable, Optional,Dict, Union, Literal,Iterable
import logging

def pos_matrix_random_sampling(K:int,A:torch.Tensor) -> torch.Tensor:
    """
    Evaluates the quadratic form z^T A z for K random vectors z normalized to unit length. Returns the K values in a Tensor.
    """
    z = torch.randn(K, A.size(-1),1,device=A.device)
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = torch.vmap(lambda z: (z.transpose(0,1)@A@z).squeeze())(z)
    return zTAz

def clamp_within_limits(x,x_lim_lower,x_lim_upper):
    """ 
    returns 0 if x is within limits, otherwise the biggest violation
    >>> x = torch.linspace(-30,30,500)
    >>> y = torch.linspace(-30,30,500)
    >>> x, y = torch.meshgrid(x, y, indexing='ij').
    >>> t= torch.cat((x.reshape(-1,1),y.reshape(-1,1)),1)
    >>> x_lim_upper = torch.tensor([4,5.])
    >>> x_lim_lower = torch.tensor([4,5.])*-2
    >>> z = torch.exp(0.1*torch.vmap(lambda t: clamp_within_limits(t,x_lim_lower,x_lim_upper))(t))
    >>> ax = plt.axes(projection='3d')
    >>> ax.plot_surface(x.numpy(), y.numpy(), z.reshape(500,500).numpy())
    """
    return torch.functional.F.relu(torch.max(torch.max(x - x_lim_upper),-torch.min(x - x_lim_lower)))

def constraint_loss_1(loss,x,x_lim_lower,x_lim_upper, sigmoid_slope = 1):
    """
    returns loss when constraints are not violated, otherwise the up to the square of it + itself
    """
    return loss*(1. + loss*( -1 + 2*torch.sigmoid(sigmoid_slope * clamp_within_limits(x,x_lim_lower,x_lim_upper)))) 

def constraint_loss_2(loss,x,x_lim_lower,x_lim_upper, sigmoid_slope = 10):
    """
    returns ~loss when constraints are not violated, otherwise the value of loss + exp(sigmoid_slope * violation)
    """
    return loss + torch.exp(sigmoid_slope * clamp_within_limits(x,x_lim_lower,x_lim_upper))

def constraint_loss_3(x,x_lim_lower,x_lim_upper, exponent_coeff = 1, mul_coeff = 0.1):
    return mul_coeff*(torch.exp(-exponent_coeff*(x_lim_upper - x)) + torch.exp(-exponent_coeff*(x - x_lim_lower)))


def build_sequential_layers(input_size:int,hidden_size:Union[int,List[int]] , output_size:int, num_hidden_layers:int = 0, mode: Literal['constant_size','linear_interpolation'] = 'constant_size',droprate:int=0.4,activation:Callable=lambda *args,**kwargs: nn.Softplus(True)):
    if isinstance(hidden_size, Iterable):
        layer_dims = [input_size] + list(hidden_size)
        logging.warning("hidden_size is an iterable, so num_hidden_layers and mode are ignored")
    else:
        if mode == 'constant_size':
            layer_dims = [input_size] + [hidden_size]*num_hidden_layers
        elif mode == 'linear_interpolation':
            layer_dims = torch.linspace(input_size, output_size, num_hidden_layers, dtype=int)
            logging.warning("Mode is linear_interpolation, so hidden_size is ignored")
    
    layers = []
    for dim in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[dim], layer_dims[dim + 1]))
        layers.append(nn.Dropout(droprate))
        layers.append(activation(index=dim,input_dimension = layer_dims[dim],output_dimension = layer_dims[dim + 1] ))

    layers.append(nn.Linear(layer_dims[-1], output_size))
    return nn.Sequential(*layers)