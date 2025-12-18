

from functools import partial
from toolz import memoize, pipe, accumulate,groupby, compose,compose_left, merge
from collections import namedtuple
import pathlib,os
import typing as T
import pydrake
# from pydrake.all import (
#     MultibodyPlant,
#     ModelInstanceIndex,
#     Body, namedview,
#     JacobianWrtVariable,
#     Frame, Diagram
# )
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex, Body, Frame
from pydrake.systems.framework import Diagram
from pydrake.multibody.tree import JacobianWrtVariable




import symforce
symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")
import sympy as sp
from sympy.codegen import CodeBlock,Assignment
import symforce.symbolic as sf
import casadi as ca
from projects.refactor_mpb.symforce_casadi.casadi_config import CasadiConfig
from projects.refactor_mpb.dynamical_system_wrapper import DynamicalSystemWrapper
from symforce.codegen.backends.pytorch import PyTorchConfig, PyTorchCodePrinter
from symforce.ops.storage_ops import StorageOps

import pydrake
from pydrake.symbolic import to_sympy as drake_to_sympy
# from pydrake.all import (
#     MultibodyPlant, MakeVectorVariable, JacobianWrtVariable, Frame,Body
# )
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import Frame, Body
from pydrake.symbolic import MakeVectorVariable#, JacobianWrtVariable

from utils.my_sympy.misc import *
from utils.my_sympy.conversions import *
from utils.misc import *
from utils.my_drake.misc import *
from utils.math.quaternions import quaternion_dot_to_angular_velocity, hamiltonian_product
from utils.my_casadi.math import analytical_jacobian_to_geometric_matrix, quaternion_to_euler_angles, rot_matrix_to_quaternion

## Relevant code snippets:
# def _print_AppliedUndef(self,expr):
#     return f"torch.{expr.func.name}({', '.join(map(self._print, expr.args))})"
# setattr(PyTorchCodePrinter, '_print_AppliedUndef', _print_AppliedUndef)
def replace_constants_pytorch(path):
    """
    Quick hax to replace constant tensors in a pytorch file with variables.
    """
    import re
    with open(path, 'r') as file:
        content = file.read()

    # Regular expression to match constant tensor declarations including fractions
    pattern = r"torch\.tensor\(([\d\.-]+|[\d\.-]+\ / [\d\.-]+), \*\*tensor_kwargs\)"
    matches = re.findall(pattern, content)

    # Evaluate the fractions and convert to float, create a map of unique constants to new variable names
    constant_to_variable = {}
    unique_constants = {}
    for match in set(matches):
        try:
            constant = float(eval(match))
            
            var_name = f"cte_{len(unique_constants) + 1}"
            if constant not in unique_constants:
                unique_constants[constant] = var_name
            constant_to_variable[match] = constant
        except Exception as e:
            print(f"Error evaluating match '{match}': {e}")
            continue

    # Replace the occurrences of constant tensor declarations with new variable names
    for match, constant in constant_to_variable.items():
        # constant_str = str(constant) if '/' not in str(constant) else f'({str(constant)})'
        var_name = unique_constants[constant]
        content = re.sub(
            pattern.replace(r'([\d\.-]+|[\d\.-]+\ / [\d\.-]+)', (match)),
            var_name,
            content
        )
    new_declarations = '\n'.join(f"    {var_name} = torch.tensor({constant}, **tensor_kwargs)" for constant, var_name in unique_constants.items())
    content = content.replace('# Input arrays','# Input arrays\n\n'+new_declarations)
    with open(path, 'w') as file:
        file.write(content)
def drake_matrix_to_sympy(matrix, memo):
    matrix = matrix.tolist()
    row_list = []
    for row in matrix:
        column_list = []
        for element in row:
            element = pydrake.symbolic.to_sympy(element,memo = memo)
            column_list.append(element)
        row_list.append(column_list)
    return sp.Matrix(row_list)
def quaternion_exponential(v:Iterable,eps:float = 1e-8) -> ca.MX:
    # https://theorangeduck.com/page/exponential-map-angle-axis-angular-velocity
    half_angle = ca.norm_2(v)
    result = ca.if_else(half_angle < eps,ca.vertcat(1,v[0],v[1],v[2]),ca.vertcat(ca.cos(half_angle),*(v*ca.sin(half_angle)/half_angle).nz))
    return result
def sympy_expression_to_symforce(sympy_expr, sympy_to_symforce_dict):
    # assert isinstance(sympy_expr,(sp.MutableSparseMatrix,sp.ImmutableSparseMatrix,sp.Matrix,sp.MutableDenseMatrix,sp.ImmutableMatrix,sp.ImmutableDenseMatrix)) , "im only really considering matrices right now so edit this to change to symforce"
    if isinstance(sympy_expr,(sp.MutableSparseMatrix,sp.ImmutableSparseMatrix,sp.Matrix,sp.MutableDenseMatrix,sp.ImmutableMatrix,sp.ImmutableDenseMatrix)):
        return sf.Matrix(sympy_expr.subs(sympy_to_symforce_dict).tolist())
    
    return sympy_expr

def angular_velocity_to_quat_dot_matrix(q):
    """
    Calculates the quaternion derivative from the angular velocity vector using CasADI.
    """
    H = sp.Matrix( [[-q[1], q[0], -q[3], q[2]],
                    [-q[2], q[3], q[0], -q[1]],
                    [-q[3], -q[2], q[1], q[0]]])
    return 1/2*H.T

def sympy_to_symforce(var,):
    # symengine.sympify might just do this better?
    # functions on matrices is fucked sp.Function('whatever')(sp.Matrix(...)) won't work
    if isinstance(var,(sp.Symbol,sp.Number)):
        return StorageOps.from_storage(sf.Symbol,[var])
    elif isinstance(var,(sp.MatrixSymbol)):
        # https://github.com/symforce-org/symforce/blob/42c4df720f85d212380ba2d4c57b6dbb3f5fbe76/symforce/databuffer.py#L14 ?
        raise ValueError("MatrixSymbols don't work with sympy->symengine conversion")
    elif isinstance(var,(sp.IndexedBase)):
        raise ValueError("IndexedBase don't work with sympy->symengine conversion")
    elif isinstance(var,(sp.Matrix,)):
        return sf.Matrix(var)
    else:
        try:
            return StorageOps.from_storage(sf.Expr,[var])
        except:
            print("Tried to convert",var, " with StorageOps.from_storage(sf.Expr,[var])")
            raise ValueError(f"Type {type(var)} not implemented in sympy_var_to_symforce")
def cse_then_print(expr,path):
    expr_cse = sp.cse(expr,order = 'none')
    temp_terms = [Assignment(*term) for term in (expr_cse)[0]]
    preamble = "from sympy import *"
    vars = sp.python(expr.free_symbols).split("\n")[:-1]
    intermediate_results = sp.pycode(CodeBlock(*temp_terms),fully_qualified_modules=False)
    output = sp.python(expr_cse[1][0]).split("\n")[-1]

    text = "\n".join([preamble] + vars + [intermediate_results] + [output])
    with open(path,'w') as f:
        f.write(text)
        
class MultiBodyPlantWrapper(DynamicalSystemWrapper):
    """
    TODO: 
    - checking small coeffs when checking if the function exists
    """

    def __init__(self, plant: MultibodyPlant,diagram: Diagram, temp_folder = None,clean_small_coeffs = 1e-6) -> None:
        self.plant = plant
        self.clean_small_coeffs = clean_small_coeffs
        self.is_discrete = plant.time_step() > 0
        self.diagram = diagram.ToSymbolic()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.sym_plant = self.diagram.GetSubsystemByName(plant.get_name())
        self.sym_context = self.sym_plant.GetMyContextFromRoot(self.diagram_context)
        self.sym_world_frame = self.sym_plant.world_frame()
        
        self.drake_position_variables = MakeVectorVariable(
            plant.num_positions(), name='q')
        self.drake_velocity_variables = MakeVectorVariable(
            plant.num_velocities(), name='v')
        self.drake_actuation_variables = MakeVectorVariable(
            self.sym_plant.GetInputPort('actuation').size(), name='u')
        self.sym_plant.SetPositions(self.sym_context, self.drake_position_variables)
        self.sym_plant.SetVelocities(self.sym_context, self.drake_velocity_variables)
        self.sym_plant.GetInputPort('actuation').FixValue(self.sym_context, self.drake_actuation_variables)
        self.variables_memo = {}
        self.variables_memo.update({q.get_id(): sp.Symbol(f'q_{i}') for i,q in enumerate(self.drake_position_variables)})
        self.variables_memo.update({v.get_id(): sp.Symbol(f'v_{i}') for i,v in enumerate(self.drake_velocity_variables)})
        self.variables_memo.update({u.get_id(): sp.Symbol(f'u_{i}') for i,u in enumerate(self.drake_actuation_variables)})

        # self.variables_memo = (
        #     {q.get_id(): sp.Symbol(f'q_{i}') for i,q in enumerate(self.drake_position_variables)} | 
        #     {v.get_id(): sp.Symbol(f'v_{i}') for i,v in enumerate(self.drake_velocity_variables)} | 
        #     {u.get_id(): sp.Symbol(f'u_{i}') for i,u in enumerate(self.drake_actuation_variables)}
        # )
        

        self.sympy_velocity_variables = [drake_to_sympy(var, memo = self.variables_memo) for var in self.drake_velocity_variables]
        self.sympy_position_variables = [drake_to_sympy(var, memo = self.variables_memo) for var in self.drake_position_variables]
        self.sympy_actuation_variables = [drake_to_sympy(var, memo = self.variables_memo) for var in self.drake_actuation_variables]

        super().__init__(self.sympy_position_variables + self.sympy_velocity_variables,self.sympy_actuation_variables,temp_folder)

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
    def num_positions(self):
        return self.plant.num_positions()

    def num_velocities(self):
         return self.plant.num_velocities()
    
    def get_sym_frame(self, frame:Frame):
        return self.sym_plant.GetFrameByName(frame.name(),frame.model_instance())
    
    @memoize
    def get_frame_pose_in_frame_sympy(self, frame:Frame, frame_ref:Frame, clean_small_coeffs = None):
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        sym_frame:Frame = self.get_sym_frame(frame)
        sym_frame_ref = self.get_sym_frame(frame_ref)
        
        pose = drake_matrix_to_sympy(sym_frame.CalcPose(self.sym_context,sym_frame_ref).GetAsMatrix4(),self.variables_memo)
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in pose.atoms(sp.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            pose = pose.subs(d)
        return pose
    @memoize
    def get_frame_pose_in_frame_function(self, frame:Frame, frame_ref:Frame, clean_small_coeffs:float = None, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        function_name = f"frame_pose_in_frame_{frame.name()}_{frame_ref.name()}"
        inputs = [sp.Matrix(self.sympy_position_variables)]
        path = self.TEMP_FOLDER / "frame_pose_in_frame"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')

        except:
            sympy_expression = self.get_frame_pose_in_frame_sympy(frame,frame_ref,clean_small_coeffs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        
        inputs = [sp.Matrix(self.sympy_position_variables)]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)

    
    def calc_frame_pose_in_frame(self, q :Iterable,frame:Frame, frame_expressed:Frame, clean_small_coeffs:float = None, module = 'casadi'):
        """
            Calculate the pose of `frame` expressed `frame_ref` using `q` as the generalized positions of the plant.
            
            This means that the point `p` in `frame` is expressed in `frame_ref` as `p_ref = pose @ p`.
            
            If `clean_small_coeffs` is set to a positive number, coefficients of the resulting pose smaller than that number will be set to zero, simplifying the expression. (recommended: 1e-6)
        """
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        if module == 'sympy':
            return self.get_frame_pose_in_frame_function(frame,frame_expressed,clean_small_coeffs, module = module).subs({i:q for i,q in zip(self.sympy_position_variables,q)})
        return self.get_frame_pose_in_frame_function(frame,frame_expressed,clean_small_coeffs, module = module)(q)
    
    @memoize
    def velocity_to_generalized_velocity_matrix_sympy(self):
        M = sp.Matrix.zeros(self.num_positions(),self.num_velocities())

        v_named = namedview('',self.plant.GetVelocityNames())([a for a in self.sympy_velocity_variables])
        q_named = namedview('',self.plant.GetPositionNames())([a for a in self.sympy_position_variables])
        count = 0
        for i,name in enumerate(self.plant.GetPositionNames()):
            last_el = name.split('_')[-1]
            name = name[:-len(last_el)-1]
            if last_el == 'qw':
                # v = 
                count += 1
                qw = eval("q_named" + "." +  name+ "_qw")
                qx = eval("q_named" + "." +  name+ "_qx")
                qy = eval("q_named" + "." +  name+ "_qy")
                qz = eval("q_named" + "." +  name+ "_qz")
                m = angular_velocity_to_quat_dot_matrix([qw,qx,qy,qz])
                M[i:i+4,i:i+3] = m
                pass
            elif last_el in ['qx','qy','qz']:
                
                continue
            else:
                
                M[i,i-count] = 1
        return M
    @memoize
    def get_velocity_to_generalized_velocity_matrix_function(self, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy'] , "module must be one of ['casadi','pytorch','sympy']"
        
        function_name = f"velocity_to_generalized_velocity_matrix"
        inputs = [sp.Matrix(self.sympy_position_variables)]
        path = self.TEMP_FOLDER / "velocity_to_generalized_velocity_matrix"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')
        except:
            sympy_expression = self.velocity_to_generalized_velocity_matrix_sympy()
            inputs = [sp.Matrix(self.sympy_position_variables)]
            return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    def calc_velocity_to_generalized_velocity_matrix(self, q :Iterable, module = 'casadi'):
        """
        Calculate the matrix that maps the velocities of the plant to the generalized velocities of the plant.
        """
        if module == 'sympy':
            return self.get_velocity_to_generalized_velocity_matrix_function(module = module).subs({i:q for i,q in zip(self.sympy_position_variables,q)})
        return self.get_velocity_to_generalized_velocity_matrix_function(module = module)(q)
    
    @memoize
    def get_spatial_velocity_jacobian_sympy(self, with_respect_to:JacobianWrtVariable,frame_of_point:Frame,frame_measured_in:Frame,frame_expressed_in, clean_small_coeffs = None):
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        T1 = self.get_frame_pose_in_frame_function(frame_of_point,frame_measured_in,clean_small_coeffs,module= 'sympy')
        T2 = self.get_frame_pose_in_frame_function(frame_measured_in,frame_expressed_in,clean_small_coeffs,module= 'sympy')
        point = sp.Matrix([[sp.Symbol('x')],[sp.Symbol('y')],[sp.Symbol('z')],[1.0]])
        R = T1[0:3,0:3]
        t = (T1@point)[0:3,0]
        J_R = R.reshape(9,1).jacobian(self.sympy_position_variables)
        J_t = t.jacobian(self.sympy_position_variables)
        for i in range(self.num_positions()):
            J_R[:,i] = (J_R[:,i].reshape(3,3)@R.T).reshape(9,1)
        J_R = J_R[2*3+1,:].col_join(J_R[0*3+2,:]).col_join(J_R[1*3+0,:])
        rotational_velocities = T2[0:3,0:3]@J_R
        translational_velocities = T2[0:3,0:3]@J_t
        J = (rotational_velocities.col_join(translational_velocities))
        if with_respect_to == JacobianWrtVariable.kV:
            M = self.get_velocity_to_generalized_velocity_matrix_function(module = 'sympy')
            J = J@M
        return J
    @memoize
    def get_spatial_velocity_jacobian_function(self, with_respect_to:JacobianWrtVariable,frame_of_point:Frame,frame_measured_in:Frame,frame_expressed_in, clean_small_coeffs = None, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy'] , "module must be one of ['casadi','pytorch','sympy']"
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        function_name = f"spatial_velocity_jacobian_{with_respect_to.name}_{frame_of_point.name()}_{frame_measured_in.name()}_{frame_expressed_in.name()}"
        inputs = [sp.Matrix(self.sympy_position_variables), sp.Matrix([[sp.Symbol('x')],[sp.Symbol('y')],[sp.Symbol('z')]])]
        path = self.TEMP_FOLDER / "spatial_velocity_jacobian"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')
        except:
            sympy_expression = self.get_spatial_velocity_jacobian_sympy(with_respect_to, frame_of_point, frame_measured_in,frame_expressed_in,clean_small_coeffs)
            return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    def calc_spatial_velocity_jacobian(self, q,with_respect_to:JacobianWrtVariable,frame_of_point:Frame,point:Iterable,frame_measured_in:Frame,frame_expressed_in, clean_small_coeffs = None, module = 'casadi'):
        """
            Calculate the pose of `frame` expressed `frame_ref` using `q` as the generalized positions of the plant.
            
            This means that the point `p` in `frame` is expressed in `frame_ref` as `p_ref = pose @ p`.
            
            If `clean_small_coeffs` is set to a positive number, coefficients of the resulting pose smaller than that number will be set to zero, simplifying the expression. (recommended: 1e-6)
        """
        # assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"
        # if module == 'casadi':
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        if module == 'sympy':
            subs = merge({i:q for i,q in zip(self.sympy_position_variables,q)},
                        {sp.Symbol('x'):point[0],sp.Symbol('y'):point[1],sp.Symbol('z'):point[2]})
            return self.get_spatial_velocity_jacobian_function(with_respect_to,frame_of_point,frame_measured_in,frame_expressed_in, clean_small_coeffs, module = 'sympy').subs(subs)
        return self.get_spatial_velocity_jacobian_function(with_respect_to,frame_of_point,frame_measured_in,frame_expressed_in, clean_small_coeffs, module = module)(q,point)
    @memoize
    def get_mass_matrix_sympy(self, clean_small_coeffs = None):
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        M = self.sym_plant.CalcMassMatrix(self.sym_context).reshape(self.num_positions(),self.num_positions())
        M = drake_matrix_to_sympy(M,self.variables_memo)
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in M.atoms(sp.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            M = M.subs(d)
        return M
    @memoize
    def get_mass_matrix_function(self, clean_small_coeffs:float = None, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        
        inputs = [sp.Matrix(self.sympy_position_variables + self.sympy_velocity_variables),]
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw('mass_matrix', path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw('mass_matrix',path = path, module = 'sympy')
            self.make_function_from_expression('mass_matrix', sympy_expression, None, path = path, module = 'sympy') #saves csed expression

        except:
            sympy_expression = self.get_mass_matrix_sympy(clean_small_coeffs)
        return self.make_function_from_expression('mass_matrix', sympy_expression, inputs, path = path, module = module)
    @memoize
    def get_bias_term_sympy(self, clean_small_coeffs = None):
        # C(q, v)v
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        C_v = self.sym_plant.CalcBiasTerm(self.sym_context).reshape(-1,1)
        C_v = drake_matrix_to_sympy(C_v,self.variables_memo)
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in C_v.atoms(sp.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            C_v = C_v.subs(d)
        return C_v
    @memoize
    def get_bias_term_function(self, clean_small_coeffs:float = None, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        
        function_name = 'bias_term'
        
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')

        except:
            sympy_expression = self.get_bias_term_sympy(clean_small_coeffs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.sympy_position_variables + self.sympy_velocity_variables)]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    @memoize
    def get_gravity_term_sympy(self, clean_small_coeffs = None):
        # tau_g(q)
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        tau_g = self.sym_plant.CalcGravityGeneralizedForces(self.sym_context).reshape(-1,1)
        tau_g = drake_matrix_to_sympy(tau_g,self.variables_memo)
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in tau_g.atoms(sp.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            tau_g = tau_g.subs(d)
        return tau_g
    @memoize
    def get_gravity_term_function(self, clean_small_coeffs:float = None, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        
        function_name = 'gravity_term'
        
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')

        except:
            sympy_expression = self.get_gravity_term_sympy(clean_small_coeffs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.sympy_position_variables + self.sympy_velocity_variables),]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    @memoize
    def get_damping_term_sympy(self, clean_small_coeffs = None):
        # tau_g(q)
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        damping = np.diag(np.concatenate([joint.default_damping_vector() for joint in get_all_joints(self.sym_plant)]))@np.array(self.drake_velocity_variables)
        damping = damping.reshape(-1,1)
        damping = drake_matrix_to_sympy(damping,self.variables_memo)
        if clean_small_coeffs > 0:
            small_numbers = set([e for e in damping.atoms(sp.Number) if abs(e) < clean_small_coeffs])
            d = {s: 0 for s in small_numbers}
            damping = damping.subs(d)
        return damping

    @memoize
    def get_damping_term_function(self, clean_small_coeffs:float = None, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        
        function_name = 'damping_term'
        
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')

        except:
            sympy_expression = self.get_damping_term_sympy(clean_small_coeffs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.sympy_position_variables + self.sympy_velocity_variables),]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    @memoize
    def get_time_derivative_sympy(self, clean_small_coeffs = None):
        # Mv̇ + C(q, v)v = tau_g(q) + tau_app
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        M = self.get_mass_matrix_function(clean_small_coeffs, module = 'sympy')
        C_v = self.get_bias_term_function(clean_small_coeffs, module = 'sympy')
        G = self.get_gravity_term_function(clean_small_coeffs, module = 'sympy')
        D = self.get_damping_term_function(clean_small_coeffs, module = 'sympy')
        # Do using symengine because much faster
        actuation_matrix = drake_matrix_to_sympy(self.sym_plant.MakeActuationMatrix(),self.variables_memo)
        q_dd = sf.Matrix(M).inv()*(sf.Matrix(G - C_v - D) + sf.Matrix(actuation_matrix)*sf.Matrix(self.sympy_actuation_variables).reshape(self.num_inputs(),1))
        
        assert len(self.sympy_velocity_variables) == len(self.sympy_position_variables), "this will be wrong if there are quaternions in q I think"
        q_d = sp.Matrix(self.sympy_velocity_variables).reshape(self.num_velocities(),1)
        
        if clean_small_coeffs > 0:
            small_numbers = set(e 
                     for element in q_dd.to_flat_list()
                     for e in element.atoms(sf.Number) if abs(e) < clean_small_coeffs 
                     )
            d = {s: 0 for s in small_numbers}
            q_dd = q_dd.subs(d)
        # return back to sympy because symforce is being weird about generating from symengine 
        # TODO: I am not sure if Drake returns Mq'' such that q have quaternions or angular acceleration
        return q_d.col_join(sp.Matrix(q_dd).reshape(self.num_positions(),1)).reshape(self.num_states(),1)
    @memoize
    def get_time_derivative_function(self, clean_small_coeffs:float = None, module = 'casadi'):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"
        if not clean_small_coeffs:
            clean_small_coeffs = self.clean_small_coeffs
        
        function_name = 'time_derivative'
        
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')

        except:
            sympy_expression = self.get_time_derivative_sympy(clean_small_coeffs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.sympy_position_variables + self.sympy_velocity_variables),sp.Matrix(self.sympy_actuation_variables)]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
