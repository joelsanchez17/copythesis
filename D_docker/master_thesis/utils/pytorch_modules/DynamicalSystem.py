from functools import partial
from typing import List,Callable, Optional,Dict, Union, Literal, Any, Iterable
import torch
from torch import nn
import traceback
import logging
from toolz import pipe
from utils.my_drake.pytorch import FunctionInput, create_drake_dynamical_system
class DynamicalSystem(nn.Module):
        
    @property
    def num_states(self):
        return len(self.state_names)
    @property
    def num_inputs(self):
        return len(self.input_names)
    @property
    def num_outputs(self):
        return len(self.output_names)
    
    def __init__(self,states: Union[List[str],int],inputs: Optional[Union[List[str],int]] = None, outputs: Optional[Union[List[str],int,Literal['states']]] = 'states', functions: Optional[Dict[str,Callable]] = None):
        super().__init__()        
        if not isinstance(states, (Iterable,int)):
            raise Exception("states must be iterable or int")
        if states is int:
            if states <= 0:
                raise Exception("states must be positive")
            states = [f"x{i}" for i in range(states)]
        if inputs is int:
            inputs = [f"u{i}" for i in range(inputs)]
        if outputs is int:
            outputs = [f"x{i}" for i in range(outputs)]
        if outputs == 'states':
            outputs = states
        self.state_names = states
        self.input_names = inputs
        self.output_names = outputs
        if functions is not None: 
            for key, value in functions.items():
                setattr(self, key, value)
        if not hasattr(self,'f_x'):
            logging.WARNING('Setting f_x = partial(self.f_x_u,u = torch.zeros(self.num_inputs))')
            self.f_x = partial(self.f_x_u,u = torch.zeros(self.num_inputs))
        if not hasattr(self,'B'):
            logging.WARNING('Assuming the system is input affine, setting B = diff(f_x_u,u)')
            if self.num_inputs == 0:
                self.B = lambda x,u = torch.empty((0,)),t = torch.zeros(1): torch.empty((self.num_states,0))
            else:
                self.B = lambda x,u = torch.empty((self.num_inputs,)),t = torch.zeros(1): torch.func.jacfwd(self.f_x_u,argnums=1)(x,u,t)
        if not hasattr(self,'g'):
            if outputs == 'states':
                self.g = lambda x,u = torch.empty((self.num_inputs,)),t = torch.zeros(1): x
                logging.WARNING('Setting g = lambda x,u,t: x')
            else:
                raise Exception("Output function g not defined")
        if not hasattr(self,'df_dx'):
            self.df_dx = lambda x,t: torch.func.jacfwd(self.f_x,argnums=0)(x,t)
            logging.WARNING('Setting self.df_dx = lambda x,t: torch.func.jacfwd(self.f_x,argnums=0)(x,t)')
        if not hasattr(self,'A'):
            self.df_dx = lambda x,u,t: self.df_dx(x,t) + pipe(torch.func.jacfwd(self.B,argnums=0)(x,u,t),lambda dB_dx: self.df_dx(x,t) + sum(u[i]*dB_dx[:,i,:] for i in range(self.num_inputs)))
            logging.WARNING('Setting self.A = lambda x,u,t: self.df_dx(x,t) + pipe(torch.func.jacfwd(self.B,argnums=0)(x,u,t),lambda dB_dx: self.df_dx(x,t) + sum(u[i]*dB_dx[:,i,:] for i in range(self.num_inputs)))')
    

    def f_x_u(self,x=None,u=None,t=None):
        raise Exception("f_x_u not set")
    
    def outputs(self,x=None,u=None,t=None):
        return self.g(x,u,t)        
    
    def dynamics_terms(self,x=None,u=None,t=None):
        """
        afsd
        """
        # f_x = self.f_x(x,t)
        f_x = self.f_x(x,t)
        if self.num_inputs == 0:
            return {'x_dot':f_x,'f_x':f_x,'B': torch.empty((self.num_states,0))}
        B = self.B(x,t)
        x_dot = f_x + B.matmul(u.view(-1,1))
        return {'x_dot':x_dot,'f_x':f_x,'B':B}
    
    
    def differential_dynamics_terms(self,x=None,u=None,t=None):
        df_dx = self.df_dx(x,t)
        A = self.A(x,u,t = t, df_dx = df_dx)
        return {'A':A,'df_dx':df_dx}
    
    def runge_kutta_4(self,x,u,dt):
        k1 = self.dynamics(x,u)['x_dot']
        k2 = self.dynamics(x + k1 * dt/2,u)['x_dot']
        k3 = self.dynamics(x + k2 * dt/2,u)['x_dot']
        k4 = self.dynamics(x + k3 * dt,u)['x_dot']
        return x + (k1 + 2*k2 + 2*k3 + k4) * dt/6
    
    def create_drake_dynamical_system(self):
        """
        Iterate over methods of DynamicalSystem, create a FunctionInput for each method, and return a DrakeDynamicalSystem.
        
        """
        pass
    
    
    @staticmethod
    def create_from_drake(drake_system):
        pass
        # return create_from_drake(drake_system)
    @staticmethod
    def create_from_sympy(sympy_system):
        pass
    @staticmethod
    def create_from_string(system_strings):
        pass
    
    