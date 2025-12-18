
import multiprocessing
from toolz import memoize, pipe, accumulate, groupby, compose, compose_left
from toolz.curried import do
from collections import namedtuple
from sympy import lambdify
import sympy
import casadi as ca
import time

import pydrake
from pydrake.all import (
    MultibodyPlant, MakeVectorVariable, JacobianWrtVariable, Frame,Body
)

from utils.my_sympy.misc import *
from utils.my_sympy.conversions import *
from utils.misc import *
from utils.my_drake.misc import *
from utils.math.quaternions import quaternion_dot_to_angular_velocity, hamiltonian_product
from utils.my_casadi.math import analytical_jacobian_to_geometric_matrix, quaternion_to_euler_angles, rot_matrix_to_quaternion

def quaternion_exponential(v:Iterable,eps:float = 1e-8) -> ca.MX:
    # https://theorangeduck.com/page/exponential-map-angle-axis-angular-velocity
    half_angle = ca.norm_2(v)
    result = ca.if_else(half_angle < eps,ca.vertcat(1,v[0],v[1],v[2]),ca.vertcat(ca.cos(half_angle),*(v*ca.sin(half_angle)/half_angle).nz))
    return result

class CasadiMultiBodyPlantWrapper:
    """
    TODO: 
    - Dynamics
    - if using BSpline derivative, the derivatives of the orientation of free bodies are quaternion derivatives, not angular velocities
        - do something using the q_dot_to_angular_velocity function
    - maybe inherit from multibodyplant instead of having it as a member
    - FIX: having to call function(casadi_variable.nz) instead of function(casadi_variable)
    - calc_euler_angles_for_free_body(q,model_instance,'xyz')
    - conversion between jacobian types pg 24
        - https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/RD2016script.pdf
        - https://math.stackexchange.com/questions/3887845/mapping-from-quaternion-jacobian-to-geometric-jacobian
            quaternion one might me incorrect
    """
# q = ca.SX.sym('x',21,1)
# Js_1 = MPC.casadi_plant.calc_spatial_velocity_jacobian(q.nz,JacobianWrtVariable.kQDot, MPC.EE_frame_1,[0,0,0],MPC.world_frame,MPC.world_frame,clean_small_coeffs=1e-6)
# T = MPC.casadi_plant.calc_frame_pose_in_frame(q.nz,MPC.EE_frame_1,MPC.world_frame,clean_small_coeffs=1e-2)
# R = T[0:3,0:3]
# t = T[0:3,3]
# Js_c = ca.jacobian(R,q)
# Jt_c = ca.jacobian(t,q)


# # 
# # print(F1(q))
# # print(ca.densify(F2(q)))

# q_dot = ca.SX.sym('dx',21,1)
# W = (Js_c@q_dot).reshape((3,3))@R.T
# w_1 = W[2,1]
# w_2 = W[0,2]
# w_3 = W[1,0]
# F = ca.Function('F',[q,q_dot],[ca.cse(ca.vertcat(w_1,w_2,w_3,Jt_c@q_dot))])
# F2 = ca.Function('F2',[q,q_dot],[ca.cse(Js_1@q_dot)])
    def __init__(self, plant: MultibodyPlant,cse=True) -> None:
        self.plant = plant
        
        self.mapping = {'ImmutableDenseMatrix': ca.blockcat,
                        'MutableDenseMatrix': ca.blockcat,
                        'Abs': ca.fabs
                        }
        self.index_notation = 'parenthesis'
        self.drake_position_variables = MakeVectorVariable(
            self.plant.num_positions(), name='x')
        self.drake_velocity_variables = MakeVectorVariable(
            self.plant.num_velocities(), name='dx')
        
        self.sym_plant = plant.ToSymbolic()
        self.sym_context = self.sym_plant.CreateDefaultContext()
        self.sym_plant.SetPositions(self.sym_context, self.drake_position_variables)
        self.sym_plant.SetVelocities(self.sym_context, self.drake_velocity_variables)
        self.sym_world_frame = self.sym_plant.world_frame()
        self.cse = cse
        
        self.sympy_velocity_variables = [pydrake.symbolic.to_sympy(var) for var in self.drake_velocity_variables]
        self.sympy_position_variables = [pydrake.symbolic.to_sympy(var) for var in self.drake_position_variables]
        
        self.free_bodies = get_all_free_bodies(self.plant)
        
        # self.forward_euler = self.make_forward_integration_euler_function()
        self.plant_named_view = make_namedview_positions(plant,'')
        
        self.position_upper_limits = self.plant.GetPositionUpperLimits()
        self.position_lower_limits = self.plant.GetPositionLowerLimits()
    
    def get_free_body_position_with_euler_angles(self, q:Iterable, model_instance:ModelInstanceIndex, order:str):
        model_instances_of_free_bodies = [body.model_instance() for body in self.free_bodies]
        assert model_instance in model_instances_of_free_bodies, f"Model instance {model_instance} not in {model_instances_of_free_bodies}"
        current_position_object = self.get_positions_from_array(model_instance,q)
        current_euler_angles = quaternion_to_euler_angles(current_position_object[0:4],order,extrinsic=False)
        current_translation = current_position_object[4:]
        current_position_object = ca.vertcat(current_euler_angles,current_translation)
        return current_position_object
    def named_vector(self, name:str,vector:Iterable):
        """
        Gets a named view of the vector `vector` with name `name`.
        """
        if isinstance(vector,(ca.MX,ca.SX,ca.DM)):
            return self.plant_named_view(name,vector.nz)
        return self.plant_named_view(name,vector)
    def set_position_upper_limits(self, position_upper_limits:Iterable):
        self.position_upper_limits = position_upper_limits
    def set_position_lower_limits(self, position_lower_limits:Iterable):
        self.position_lower_limits = position_lower_limits
        
    def get_joints_midpoints(self):
        lower_limit = self.position_upper_limits
        upper_limit = self.position_lower_limits
        if isinstance(lower_limit, (tuple,list)):
            lower_limit = np.array(lower_limit)
        if isinstance(upper_limit, (tuple,list)):
            upper_limit = np.array(upper_limit)    
        q_middle = (lower_limit+upper_limit)/2
        return q_middle
    
    def get_positions_from_array(self, model_instance:ModelInstanceIndex, q:Iterable) -> Iterable:
        """
        Extract elements from the array q at indices that align with the generalized positions of `model_instance` within the whole plant position vector." 
        
        I.e: `return q[appropriate_indices]`
        
        Generic version of `plant.GetPositionsFromArray(model_instance,q)` that works with CasADi variables.
        """
        
        x = list(range(0,self.num_positions()))
        return q[self.plant.GetPositionsFromArray(model_instance,x).astype(int)]
    def set_positions_in_array(self, model_instance:ModelInstanceIndex, q:Iterable, q_instance:Iterable) -> Iterable:
        """
        Updates the positions in the array `q` based on the positions associated with `model_instance`. 
        
        This method takes indices corresponding to `model_instance` within the plant, and replaces 
        the values at these indices in `q` with the values from `q_instance`.
        
        I.e: `q[appropriate_indices] = q_instance`
        
        Generic version of `plant.SetPositionsInArray(model_instance,q_instance,q)` that works with CasADi variables.
        """
        x = list(range(0,self.num_positions()))
        q[self.plant.GetPositionsFromArray(model_instance,x).astype(int)] = q_instance
        return q
    
    def get_velocities_from_array(self, model_instance:ModelInstanceIndex, q:Iterable) -> Iterable:
        """
        Extract elements from the array q at indices that align with the velocities of `model_instance` within the whole plant velocity vector." 
        
        I.e: `return v[appropriate_indices]`
        
        Generic version of `plant.GetVelocitiesFromArray(model_instance,q)` that works with CasADi variables.
        
        The difference between this method and `get_positions_from_array` is that this method returns the velocities of the plant, 
        which has angular velocity instead of quaternion derivatives.
        """
        
        x = list(range(0,self.num_velocities()))
        return q[self.plant.GetVelocitiesFromArray(model_instance,x).astype(int)]
    def set_velocities_in_array(self, model_instance:ModelInstanceIndex, v:Iterable, v_instance:Iterable) -> Iterable:
        """
        Updates the positions in the array `v` based on the velocities associated with `model_instance`. 
        
        This method takes indices corresponding to `model_instance` within the plant, and replaces 
        the values at these indices in `v` with the values from `v_instance`.
        
        I.e: `v[appropriate_indices] = v_instance`
        
        Generic version of `plant.SetVelocitiesInArray(model_instance,q_instance,q)` that works with CasADi variables.
        """
        x = list(range(0,self.num_velocities()))
        v[self.plant.GetVelocitiesFromArray(model_instance,x).astype(int)] = v_instance
        return v
    def make_forward_integration_euler_function(self):
        """
        Makes a function that integrates the plant forward in time using Euler integration. Appropriately handles quaternions by using the exponential map.
        """
        q = ca.SX.sym('q',self.num_positions())
        v = ca.SX.sym('v',self.num_velocities())
        dt = ca.SX.sym('dt')
        
        
        v_named = namedview('',self.plant.GetVelocityNames())([a for a in v.nz])
        q_named = namedview('',self.plant.GetPositionNames())([a for a in q.nz])
        q_new = []
        for name in self.plant.GetPositionNames():
            last_el = name.split('_')[-1]
            name = name[:-len(last_el)-1]
            if last_el == 'qw':
                # v = 
                qw = eval("q_named" + "." +  name+ "_qw")
                qx = eval("q_named" + "." +  name+ "_qx")
                qy = eval("q_named" + "." +  name+ "_qy")
                qz = eval("q_named" + "." +  name+ "_qz")
                wx = eval("v_named" + "." +  name+ "_wx")
                wy = eval("v_named" + "." +  name+ "_wy")
                wz = eval("v_named" + "." +  name+ "_wz")
                
                quat = ca.vertcat(qw,qx,qy,qz)
                w = ca.vertcat(wx,wy,wz)
                new_quaternions = hamiltonian_product(quat,quaternion_exponential(w/2*dt))
                q_new.append(new_quaternions)
                pass
            elif last_el in ['qx','qy','qz']:
                
                continue
            else:
                
                state = eval("q_named" + "." +  name + "_" + last_el)
                def get_v_name(last_el):
                    match last_el:
                        case 'q':
                            return 'w'
                        case 'x':
                            return 'vx'
                        case 'y': 
                            return 'vy'
                        case 'z':
                            return 'vz'
                v_name = get_v_name(last_el)
                v_ = eval("v_named" + "." +  name + "_" + v_name)
                q_new.append(state + v_*dt)
        
        # new_quaternions = {}
        # for body in self.free_bodies:
        #     instance = body.model_instance()
        #     # object_quaternion = self.get_positions_from_array(instance,q)[0:4]
        #     # quaternion_dot = self.get_positions_from_array(instance,v)[0:4]
            
        #     # w = quaternion_dot_to_angular_velocity(object_quaternion,quaternion_dot)
        #     object_quaternion = self.get_positions_from_array(instance,q)[0:4]
        #     w = self.get_velocities_from_array(instance,v)[0:3]
            
        #     # new_quaternions[instance] = hamiltonian_product(object_quaternion,quaternion_exponential(w/2))
        #     new_quaternions[instance] = hamiltonian_product(object_quaternion,quaternion_exponential(w/2*dt))
        # q_new = q + v*dt
        # for instance, quat in new_quaternions.items():
        #     temp = self.get_positions_from_array(instance,q_new)
        #     temp[0:4] = quat
        #     self.set_positions_in_array(instance,q_new,temp)
        
        f = ca.Function('forward_integration_euler',[q,v,dt],[ca.vertcat(*q_new)],{'cse':True})
        return f
    
    @memoize
    @staticmethod
    def get_euler_dot_to_angular_velocity_matrix(order):
        psi, theta, phi = ca.SX.sym('psi'), ca.SX.sym('theta'), ca.SX.sym('phi')
        M = analytical_jacobian_to_geometric_matrix(order,[psi,theta,phi])
        
        return ca.Function('euler_dot_to_angular_velocity_matrix',[psi,theta,phi],[M],{'cse':True,'post_expand':True,})
    
    def calc_euler_dot_to_angular_velocity_matrix(self, euler_angles:Iterable, order):
        psi, theta, phi = euler_angles[0], euler_angles[1], euler_angles[2]
        return CasadiMultiBodyPlantWrapper.get_euler_dot_to_angular_velocity_matrix(order)(psi,theta,phi)
    
    @memoize 
    @staticmethod
    def get_angular_velocity_to_euler_dot_matrix(order):
        psi, theta, phi = ca.SX.sym('psi'), ca.SX.sym('theta'), ca.SX.sym('phi')
        M = analytical_jacobian_to_geometric_matrix(order,[psi,theta,phi])
        
        return ca.Function('angular_velocity_to_euler_dot_matrix',[psi,theta,phi],[M],{'cse':True,'post_expand':True,})
    def calc_angular_velocity_to_euler_dot_matrix(self, euler_angles:Iterable, order):
        psi, theta, phi = euler_angles[0], euler_angles[1], euler_angles[2]
        return CasadiMultiBodyPlantWrapper.get_angular_velocity_to_euler_dot_matrix(order)(psi,theta,phi)
    
    
    @memoize
    def get_frame_relative_velocity_function(self, frame:Frame, frame_measured_in:Frame,frame_expressed_in, clean_small_coeffs):
        sym_frame:Frame = self.sym_plant.GetFrameByName(frame.name(),frame.model_instance())
        sym_frame_measured_in = self.sym_plant.GetFrameByName(frame_measured_in.name(),frame_measured_in.model_instance())
        sym_frame_expressed_in = self.sym_plant.GetFrameByName(frame_expressed_in.name(),frame_expressed_in.model_instance())
        velocity = sym_frame.CalcSpatialVelocity(self.sym_context,sym_frame_measured_in,sym_frame_expressed_in)
        translational_velocity = velocity.translational().tolist()
        rotational_velocity = velocity.rotational().tolist()
        velocity = [pydrake.symbolic.to_sympy(element) for element in rotational_velocity + translational_velocity]
        velocity = sp.Matrix(velocity).reshape(6,1)
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in velocity.atoms(sympy.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            velocity = velocity.subs(d)
        f = lambdify([self.sympy_position_variables,self.sympy_velocity_variables],velocity,modules=[self.mapping,ca],cse=self.cse)
        return f
    def calc_frame_relative_velocity(self, q :Iterable, v :Iterable,frame:Frame, frame_measured_in:Frame,frame_expressed_in:Frame, clean_small_coeffs:float = -1):
        """
            Calculate the velocity of `frame` relative to `frame_measured_in` expressed in `frame_expressed_in` using `q` as the generalized positions of the plant and `v` as the velocities of the plant.
            
            Note: If there are quaternion derivatives in `v` you have to convert them to angular velocities first using `quaternion_dot_to_angular_velocity` for now.
            
            If `clean_small_coeffs` is set to a positive number, coefficients of the resulting velocity smaller than that number will be set to zero, simplifying the expression.
        """
        
        return self.get_frame_relative_velocity_function(frame,frame_measured_in,frame_expressed_in,clean_small_coeffs)(q,v)
    
    @memoize
    def get_frame_pose_in_frame_function(self, frame:Frame, frame_ref:Frame, clean_small_coeffs):
        sym_frame:Frame = self.sym_plant.GetFrameByName(frame.name(),frame.model_instance())
        sym_frame_ref = self.sym_plant.GetFrameByName(frame_ref.name(),frame_ref.model_instance())
        pose = sym_frame.CalcPose(self.sym_context,sym_frame_ref).GetAsMatrix4().reshape(-1).tolist()
        pose = [pydrake.symbolic.to_sympy(element) for element in pose]
        pose = sp.Matrix(pose).reshape(4,4)
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in pose.atoms(sympy.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            pose = pose.subs(d)
        # MakeVectorVariable(
            # self.plant.num_velocities(), name='dx')
        f = lambdify([self.sympy_position_variables],pose,modules=[self.mapping,ca],cse=self.cse)
        return f
    def calc_frame_pose_in_frame(self, q :Iterable,frame:Frame, frame_expressed:Frame, clean_small_coeffs:float = -1):
        """
            Calculate the pose of `frame` expressed `frame_ref` using `q` as the generalized positions of the plant.
            
            This means that the point `p` in `frame` is expressed in `frame_ref` as `p_ref = pose @ p`.
            
            If `clean_small_coeffs` is set to a positive number, coefficients of the resulting pose smaller than that number will be set to zero, simplifying the expression. (recommended: 1e-6)
        """
        
        return self.get_frame_pose_in_frame_function(frame,frame_expressed,clean_small_coeffs)(q)
    def calc_frame_pose_in_frame_euler(self, q :Iterable,frame:Frame, frame_expressed:Frame, order:str,clean_small_coeffs:float = -1):
        """
            Calculate the pose of `frame` expressed `frame_ref` using `q` as the generalized positions of the plant.
            
            Returns the pose as an array `[psi,theta,phi,x,y,z]` where `psi,theta,phi` are the euler angles of the orientation of `frame` expressed in `frame_ref` and `x,y,z` are the translation of `frame` expressed in `frame_ref`.
            
            If `clean_small_coeffs` is set to a positive number, coefficients of the resulting pose smaller than that number will be set to zero, simplifying the expression. (recommended: 1e-6)
        """
        
        matrix = self.get_frame_pose_in_frame_function(frame,frame_expressed,clean_small_coeffs)(q)
        translation = matrix[0:3,3]
        quaternion = rot_matrix_to_quaternion(matrix[0:3,0:3])
        euler = quaternion_to_euler_angles(quaternion,order,extrinsic=False)
        return ca.vertcat(euler,translation)
        
        
    @memoize
    def get_spatial_velocity_jacobian(self, with_respect_to:JacobianWrtVariable,frame_of_point:Frame,frame_measured_in:Frame,frame_expressed_in, clean_small_coeffs):
        sym_frame:Frame = self.sym_plant.GetFrameByName(frame_of_point.name(),frame_of_point.model_instance())
        sym_frame_measured_in = self.sym_plant.GetFrameByName(frame_measured_in.name(),frame_measured_in.model_instance())
        sym_frame_expressed_in = self.sym_plant.GetFrameByName(frame_expressed_in.name(),frame_expressed_in.model_instance())
        point = MakeVectorVariable(3, name='point')
        point_sympy = [pydrake.symbolic.to_sympy(var) for var in point]
        jacobian = self.sym_plant.CalcJacobianSpatialVelocity(self.sym_context, with_respect_to, sym_frame,point,sym_frame_measured_in,sym_frame_expressed_in).reshape(-1).tolist()
        jacobian = [pydrake.symbolic.to_sympy(element) for element in jacobian]
        if with_respect_to == JacobianWrtVariable.kQDot:
            
            jacobian = sp.Matrix(jacobian).reshape(6, self.num_positions())
        else:
            jacobian = sp.Matrix(jacobian).reshape(6, self.num_velocities())
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in jacobian.atoms(sympy.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            jacobian = jacobian.subs(d)
        f = lambdify([self.sympy_position_variables,point_sympy],jacobian,modules=[self.mapping,ca],cse=self.cse)
        return f
    def calc_spatial_velocity_jacobian(self, q :Iterable,with_respect_to:JacobianWrtVariable,frame_of_point:Frame,point:Iterable,frame_measured_in:Frame,frame_expressed_in, clean_small_coeffs:float = -1):
        """
        Get `[delta_rotation,delta_translation] / delta_q` of point `point` in `frame_of_point` relative to `frame_measured_in` expressed in `frame_expressed_in` with respect to `with_respect_to` variables calculated at plant position `q`.
        
        - `with_respect_to`: `JacobianWrtVariable.kQDot` or `JacobianWrtVariable.kV`
            -    `JacobianWrtVariable.kQDot`: the jacobian will be calculated with respect to the generalized velocities of the plant (so if there is a orientation of a free body in the plant, the jacobian will be calculated with respect to the quaternion (`[w_x,w_y,w_z]_point = J * [delta_q_w,delta_q_x,delta_q_y,delta_q_z]_plant`))
            -    `JacobianWrtVariable.kV`: the jacobian will be calculated with respect to the generalized velocities of the plant (so if there is a orientation of a free body in the plant, the jacobian will be calculated with respect to the angles (`[w_x,w_y,w_z]_point = J * [w_x,w_y,w_z]_plant`))

        OBS/TODO: This can get pretty slow (especially when substituting small coefficients) so maybe its a good idea to use `ca.jacobian` instead.
        
        """
        return self.get_spatial_velocity_jacobian(with_respect_to,frame_of_point,frame_measured_in,frame_expressed_in,clean_small_coeffs)(q,point)
    def calc_squared_distance_between_points(self, q :Iterable,frame_1:Frame, point_1:Iterable, frame_2:Frame,point_2:Iterable, clean_small_coeffs:float = -1, jacobian = False) -> Union[ca.MX,Iterable]:

        """
        Calculate the distance between `point_1` in `frame_1` and `point_2` in `frame_2` using `q` as the generalized positions of the plant.
        
        If `clean_small_coeffs` is set to a positive number, coefficients of the resulting distance smaller than that number will be set to zero, simplifying the expression.
        
        If `jacobian` is set to `True`, the function will also return the jacobian of the distance with respect to `q`. Returns `tuple(distance,jacobian)`.
        """
        frame_1_in_2_pose = self.calc_frame_pose_in_frame(q,frame_1,frame_2,clean_small_coeffs=clean_small_coeffs)
        p1 = frame_1_in_2_pose[0:3,0:3] @ point_1 + frame_1_in_2_pose[0:3,3]
        p2 = point_2
        distance = ca.sumsqr(p1-p2)
        if jacobian:
            jacobian = ca.jacobian(distance,q)
            return distance,jacobian
        else:
            return distance
    def calc_norm_2_distance_between_points(self, q :Iterable,frame_1:Frame, point_1:Iterable, frame_2:Frame,point_2:Iterable, clean_small_coeffs:float = -1, jacobian = False) -> Union[ca.MX,Iterable]:

        """
        Calculate the distance between `point_1` in `frame_1` and `point_2` in `frame_2` using `q` as the generalized positions of the plant.
        
        If `clean_small_coeffs` is set to a positive number, coefficients of the resulting distance smaller than that number will be set to zero, simplifying the expression.
        
        If `jacobian` is set to `True`, the function will also return the jacobian of the distance with respect to `q`. Returns `tuple(distance,jacobian)`.
        """
        frame_1_in_2_pose = self.calc_frame_pose_in_frame(q.nz,frame_1,frame_2,clean_small_coeffs=clean_small_coeffs)
        p1 = frame_1_in_2_pose[0:3,0:3] @ point_1 + frame_1_in_2_pose[0:3,3]
        p2 = point_2
        distance = ca.norm_2(p1-p2)
        if jacobian:
            jacobian = ca.jacobian(distance,q)
            return distance,jacobian
        else:
            return distance
    def forward_integration_euler(self, q :Iterable, v :Iterable, dt:float) -> Iterable:
        """
        Integrates the plant forward in time using Euler integration. Appropriately handles quaternions by using the exponential map.
        """
        # assert len(q) == len(v)
        # this check doesn't work with casadi variables so comment out if you want to do symbolic integration
        # if len(q) != len(v):
        #     raise ValueError(f"Length of q ({len(q)}) and v ({len(v)}) must be the same (non generalized velocities not implemented)")
        return self.forward_euler(q,v,dt)
    
    def num_positions(self):
        return self.plant.num_positions()

    def num_velocities(self):
         return self.plant.num_velocities()