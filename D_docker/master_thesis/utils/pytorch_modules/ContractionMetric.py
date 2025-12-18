import torch
from torch import nn
import traceback
from typing import List,Callable, Optional,Dict, Union, Literal,Iterable
from utils.pytorch_modules import MonomialMatrixModule, DynamicalSystem, ContractionMetric
from utils.my_pytorch.misc import pos_matrix_random_sampling, build_sequential_layers

class ContractionMetric(torch.nn.Module):
    def __init__(self,num_states,normal_or_dual_metric: str = 'normal'):
        super().__init__()
        self.num_states = num_states
        self.normal_or_dual_metric = normal_or_dual_metric
    def M(self,x):
        raise NotImplementedError
    def W(self,x):
        raise NotImplementedError
    def M_M_dot(self,x,x_dot):
        if self.normal_or_dual_metric == 'normal':
            M,M_dot = torch.func.jvp(self.M,(x,),(x_dot,))
        elif self.normal_or_dual_metric == 'dual':
            M,M_dot = torch.func.jvp(lambda x: torch.inverse(self.W(x)),(x,),(x_dot,))
        return {'M':M,'M_dot':M_dot}
    
    def forward(self, x):
        M = self.M(x)#*1e-12
        return M.transpose(0,1).matmul(M) + self.Iw
    def W_W_dot(self,x,x_dot):
        if self.normal_or_dual_metric == 'normal':
            W,W_dot = torch.func.jvp(lambda x: torch.inverse(self.M(x)),(x,),(x_dot,))
        elif self.normal_or_dual_metric == 'dual':
            W,W_dot = torch.func.jvp(self.W,(x,),(x_dot,))
        return {'W':W,'W_dot':W_dot}
    
    def sample_W_quadratic_form(self,x:torch.Tensor,K:int=512):
        """
        Returns K samples of the quadratic form z^T W(x) z for K random vectors z normalized to unit length.
        """
        return pos_matrix_random_sampling(K,self.W(x))
    def sample_M_quadratic_form(self,x:torch.Tensor,K:int=512):
        """
        Returns K samples of the quadratic form z^T M(x) z for K random vectors z normalized to unit length.
        """
        return pos_matrix_random_sampling(K,self.M(x))
    def sample_quadratic_form(self,A:torch.Tensor,K:int=512):
        """
        Returns K samples of the quadratic form z^T A z for K random vectors z normalized to unit length.
        """
        return pos_matrix_random_sampling(K,A)


class PolynomialMetric(ContractionMetric):
    def __init__(self,num_states,w = 1,normal_or_dual_metric: str = 'normal',monomial_order = 4):
        super().__init__(normal_or_dual_metric = normal_or_dual_metric, num_states = num_states, w = w)
        self.num_states = num_states
        self.monomial_order = monomial_order
        self.normal_or_dual_metric = normal_or_dual_metric
        C = MonomialMatrixModule((num_states,num_states), num_states, self.monomial_order, symmetric=True)
        C.coeffs.data = C.coeffs*1e-6
        if normal_or_dual_metric == 'normal':
            self.M = lambda x: C(x).transpose(0,1).matmul(C(x)) + w*torch.eye(num_states)
        elif normal_or_dual_metric == 'dual':
            self.W = lambda x: C(x).transpose(0,1).matmul(C(x)) + w*torch.eye(num_states)
        else: raise Exception("normal_or_dual_metric must be 'normal' or 'dual'")

class BasicBitchMetric(ContractionMetric):
    def __init__(self,num_states,input_size:int,hidden_size:Union[int,List[int]] , output_size:int, num_hidden_layers:int, mode: Literal['constant_size','linear_interpolation'] = 'constant_size',normal_or_dual_metric: str = 'normal',droprate:int=0.4,activation:Callable=lambda *args,**kwargs: nn.Softplus(True),w:int = 1):
        super().__init__(normal_or_dual_metric = normal_or_dual_metric, num_states = num_states, w = w)
        self.num_states = num_states
        
        self.normal_or_dual_metric = normal_or_dual_metric
        self.layers = build_sequential_layers(input_size = num_states, hidden_size = 2*num_states, output_size = num_states**2, num_layers = 2, droprate = 0.0, activation = lambda *args,**kwargs: nn.Softplus(True))
        if normal_or_dual_metric == 'normal':
            self.M = lambda x: self.layers(x).view(self.num_states,self.num_states).transpose(0,1).matmul(self.layers(x).view(self.num_states,self.num_states)) + w*torch.eye(num_states)
            
        elif normal_or_dual_metric == 'dual':
            self.W = lambda x: self.layers(x).view(self.num_states,self.num_states).transpose(0,1).matmul(self.layers(x).view(self.num_states,self.num_states)) + w*torch.eye(num_states)
            
        else: raise Exception("normal_or_dual_metric must be 'normal' or 'dual'")

