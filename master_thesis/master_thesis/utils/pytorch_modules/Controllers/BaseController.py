import torch
from torch import nn
import traceback

class BaseController(nn.Module):
    def __init__(self,num_states,num_inputs):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        
        self.layers_output_sizes = 2
        self.w1  = torch.nn.Sequential(
        torch.nn.Linear((num_states+2)*2, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_states+2)*self.layers_output_sizes, bias=True))
        
        self.w2  = torch.nn.Sequential(
        torch.nn.Linear((num_states+2)*2, 128, bias=True),
        torch.nn.ELU(),
        torch.nn.Linear(128, self.layers_output_sizes, bias=True))
        
        # self.w1  = torch.nn.Sequential(
        # torch.nn.Linear((num_states+2)*2, 64, bias=True),
        # torch.nn.Tanh(),
        # torch.nn.Linear(64, (num_states+2)*self.layers_output_sizes, bias=True))
        
        # self.w2  = torch.nn.Sequential(
        # torch.nn.Linear((num_states+2)*2, 64, bias=True),
        # torch.nn.Tanh(),
        # torch.nn.Linear(64, self.layers_output_sizes, bias=True))
        
    def forward(self, x,x_ref, u_ref):
        new_x = torch.empty_like(x).resize_(x.shape[0]+2).view(-1)
        new_x[0:4] = torch.cat([torch.cos(x[0:2]),torch.sin(x[0:2])])
        new_x[4:] = x[2:]
        x = new_x
        new_x_ref = torch.empty_like(x_ref).resize_(x_ref.shape[0]+2)
        new_x_ref[0:4] = torch.cat([torch.cos(x_ref[0:2]),torch.sin(x_ref[0:2])])
        new_x_ref[4:] = x_ref[2:]
        x_ref = new_x_ref
        
        layer_input = torch.cat([new_x,new_x_ref])
        w1 = self.w1(layer_input).view(self.layers_output_sizes,x.shape[0])
        w2 = self.w2(layer_input).view(1,self.layers_output_sizes)
        u = u_ref + w2@torch.tanh(w1@(x-x_ref).view(x.shape[0],1))
        # return x[0:2]
        return u.squeeze()