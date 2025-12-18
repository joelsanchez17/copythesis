import torch
import numpy as np
from typing import Union
def torch_controller(model,x:np.ndarray ,t: Union[float, np.ndarray],x_ref:np.ndarray,u_ref:np.ndarray,t_ref:np.ndarray,):
    # x is a numpy array of dimension (num_states,) of the current state
    # t is a scalar of the current simulation time
    # x_ref, is a numpy array of dimension (num_states,num_points,)
    # u_ref is a numpy array of dimension (num_inputs,num_points,)
    # t_ref is a numpy array of dimension (num_points,), which is the time at each reference value
    # model is the controller in pytorch
    x_r_last = torch.tensor([x_ref[t_ref <= t][-1]],dtype=torch.float32).reshape(-1)
    u_r_last = torch.tensor([u_ref[t_ref <= t][-1]],dtype=torch.float32).reshape(-1)
    x_r_next = torch.tensor([x_ref[t_ref >= t][0]],dtype=torch.float32).reshape(-1)
    u_r_next = torch.tensor([u_ref[t_ref >= t][0]],dtype=torch.float32).reshape(-1)
    t_last = torch.tensor([t_ref[t_ref <= t][-1]],dtype=torch.float32).reshape(-1)
    t_next = torch.tensor([t_ref[t_ref >= t][0]],dtype=torch.float32).reshape(-1)
    dt = t_next - t_last
    if dt == 0:
        u_r = u_r_last
        x_r = x_r_last
    else:
        u_r = u_r_next*(t - t_last)/dt + u_r_last*(1 - (t - t_last)/dt)
        x_r = x_r_next*(t - t_last)/dt + x_r_last*(1 - (t - t_last)/dt)
        # interpolate w/ cubic spline properly for hermitian colocation, sometime
    x = torch.tensor([x],dtype=torch.float32).reshape(-1)
    return model.controller(x,x_r,u_r).detach().numpy()