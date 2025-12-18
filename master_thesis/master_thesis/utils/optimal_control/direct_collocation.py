from casadi import *
import numpy as np

def direct_collocation(dynamic,x0,num_inputs,num_points,t_start,t_end, initial_guess_states = None,initial_guess_inputs = None, cost = None, extra_constraints = None):
    # dynamic: function handle for the dynamics, should take in x and u and return dx, which is a tuple of the derivatives of x
    # 
    # 
    # 
    num_states = x0.shape[0]

    def f_vector(x,u):
        temp = [vertcat(*dynamic(x[:,i],u[:,i])) for i in range(x.shape[1]) ]
        dx = horzcat(*temp)
        return dx
    def x_half_vec(t,x,fx):
        def pairwise_sum(arr):
            return arr[:,:-1] + arr[:,1:]
        return 1/2*(pairwise_sum(x)) + (diff(t,1,1))/8*(-diff(fx,1,1))
    def constraint(x):
        u = x[num_states*(num_points+1):].reshape((num_inputs,num_points+1))
        x = x[:num_states*(num_points+1)].reshape((num_states,num_points+1))
        xx = x
        u_half =  1/2*(u[:,:-1] + u[:,1:])
        ff = f_vector(xx,u)
        x_half = x_half_vec(t_vector,xx,ff)
        x_integral = xx[:,:-1] + 1/6*(np.diff(t_vector))* (4*f_vector(x_half,u_half)+ff[:,:-1]+ff[:,1:]) - xx[:,1:]
        return x_integral.reshape((-1,1))
    t_vector = np.tile(np.linspace(t_start,t_end,num_points+1),( num_states,1)) 

    x = MX.sym('x',num_states*(num_points + 1),1)
    u = MX.sym('u',num_inputs*(num_points + 1),1)
    xx = vertcat(x,u)
    x_var_0 = x.reshape((num_states,num_points+1))[:,0]
    # x_var_end = x.reshape((num_states,num_points+1))[:,-1]
    
    constraints = vertcat(constraint(xx))
    constraints = vertcat(x0-x_var_0,constraints)

    # constraints = vertcat(x_end-x_var_end,constraints)

    ubg = [0]*constraints.shape[0]
    lbg = [0]*constraints.shape[0]
    if extra_constraints is not None:
        lbg_extra,ubg_extra, extra_constraints = extra_constraints(x.reshape((num_states,num_points+1)),u.reshape((num_inputs,num_points+1)))
        ubg += ubg_extra
        lbg += lbg_extra
        # print(extra_constraints)
        constraints = vertcat(constraints,vertcat(*extra_constraints))
    nlp = {}
    nlp['x']= vertcat(xx)
    nlp['g'] = constraints             # constraints

    if cost is None:
        nlp['f'] = 0
    else:
        nlp['f'] = cost(x.reshape((num_states,num_points+1)),u.reshape((num_inputs,num_points+1)))
    # print((x*x).shape)
    solver_opts = {'ipopt.print_level': 0}
    F = nlpsol('F','ipopt',nlp, solver_opts)
    initial_guess = []
    if initial_guess_states is None:
        initial_guess_states = [0]*num_states*(num_points+1)
    if initial_guess_inputs is None:
        initial_guess_inputs = [0]*num_inputs*(num_points+1)
    sol = F(x0=initial_guess_states+initial_guess_inputs,ubg=ubg,lbg=lbg)

    xx_res = sol['x']
    x = xx_res[:num_states*(num_points+1)].reshape((num_states,num_points+1))
    u = xx_res[num_states*(num_points+1):].reshape((num_inputs,num_points+1))
    # print(x)

    return np.linspace(t_start,t_end,num_points+1),x,u