import numba
import numpy as np
from collections import deque
from utils.math.BSpline import BSpline
import casadi as ca



@numba.njit
def integrate_q_trapezoidal(q0, qdots, dt):
    # q = [q0]  # Initialize the list of q values with the initial q0
    q = np.empty_like(qdots)
    q[0] = q0
    for i in range(1, len(qdots)):
        # Trapezoidal rule: q_{i+1} = q_i + 0.5 * dt * (qdots[i] + qdots[i-1])
        q_new = q[i-1] + 0.5 * dt * (qdots[i] + qdots[i - 1])
        q[i] = (q_new)
    return q
class BSplineFit:
    def __init__(self,num_control_points,order,number_of_positions,number_of_samples):
        
        opti = ca.Opti()
        options = {
            'expand':1, #MX -> SX
            'jit':True,
            'compiler':'shell',
            'print_time':False,
            'jit_options':
                {
                'flags': '-O3 -march=native -shared',
                },
            'ipopt': 
                {
                'print_level': 0, 
                'hessian_approximation': 'exact',
                'linear_solver': 'ma57', 
                'max_iter': 10000,
                'constr_viol_tol':1e-5,
                'max_wall_time':5
                }
            }
        control_points = opti.variable(number_of_positions,num_control_points)
        points_sym = opti.parameter(number_of_positions,number_of_samples)
        self.points_sym = points_sym
        self.order = order
        spline_sol = BSpline(control_points,order=order)
        spline_samples = ca.horzcat(*[spline_sol.evaluate(i) for i in np.linspace(0,1,number_of_samples)])
        obj = ca.sumsqr(points_sym-spline_samples)
        # opti.subject_to(points_sym[:,0] == spline_samples[:,0])
        opti.minimize(obj)
        opti.solver('ipopt',options)
        self.opti = opti
        self.control_points = control_points
        # sol = opti.solve()
    def solve(self,points):
        self.opti.set_value(self.points_sym,points)
        sol = self.opti.solve()
        return BSpline(sol.value(self.control_points),order=self.order)
    def __call__(self,points):
        return self.solve(points)
class BSplineFitWithInitialConstraint:
    def __init__(self,num_control_points,order,number_of_positions,number_of_samples):
        
        opti = ca.Opti()
        options = {
            'expand':1, #MX -> SX
            'jit':False,
            'compiler':'shell',
            'print_time':False,
            'jit_options':
                {
                'flags': '-O3 -march=native -shared',
                },
            'ipopt': 
                {
                'print_level': 0, 
                'hessian_approximation': 'exact',
                'linear_solver': 'ma57', 
                'max_iter': 10000,
                'constr_viol_tol':1e-5,
                'max_wall_time':5
                }
            }
        control_points = opti.variable(number_of_positions,num_control_points)
        points_sym = opti.parameter(number_of_positions,number_of_samples)
        start_velocity_sym = opti.parameter(number_of_positions,1)
        self.points_sym = points_sym
        self.order = order
        spline_sol = BSpline(control_points,order=order)
        spline_dot_sol = spline_sol.create_derivative_spline()
        spline_samples = ca.horzcat(*[spline_sol.evaluate(i) for i in np.linspace(0,1,number_of_samples)])
        obj = ca.sumsqr(points_sym-spline_samples)# + ca.sumsqr(ca.diff(spline_samples,1))*10
        opti.subject_to(points_sym[:,0] == spline_samples[:,0])
        # opti.subject_to(points_sym[:,1] - points_sym[:,0] == spline_samples[:,1] - spline_samples[:,0])
        # opti.subject_to(start_velocity_sym == spline_samples[:,1] - spline_samples[:,0])
        opti.subject_to(start_velocity_sym == spline_dot_sol.evaluate(0))
        opti.minimize(obj)
        opti.solver('ipopt',options)
        self.opti = opti
        self.control_points = control_points
        self.start_velocity_sym = start_velocity_sym
        print('created bspline fitter')
        # sol = opti.solve()
    def solve(self,points,velocity):
        self.opti.set_value(self.points_sym,points)
        self.opti.set_value(self.start_velocity_sym,velocity)
        sol = self.opti.solve()
        return BSpline(sol.value(self.control_points),order=self.order)
    def __call__(self,points,velocity):
        return self.solve(points,velocity)
def torch_to_drake_point_cloud(torch_pc):
    from pydrake.all import PointCloud
    import torch
    pc = PointCloud(torch_pc.shape[0])
    pc.mutable_xyzs()[:] = torch.as_tensor(torch_pc).cpu().numpy().T
    return pc