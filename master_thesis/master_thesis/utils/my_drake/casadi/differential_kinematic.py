
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
    MultibodyPlant, MakeVectorVariable, JacobianWrtVariable, Frame
)

from utils.my_sympy.misc import *
from utils.my_sympy.conversions import *
from utils.misc import *
from utils.my_drake.misc import *
from utils.my_drake.casadi.multibody_wrapper import CasadiMultiBodyPlantWrapper
from utils.math.BSpline import BSpline
class DifferentialKinematicOptimization:
    """
    `TODO:` 
    
    - Result to BSpline
    - Position integration constraint
    - Add way to output misc results from the optimization
    - use_angular_Velocity for free bodies
    """

    def __init__(self, casadi_plant: CasadiMultiBodyPlantWrapper,cse = True,solver_name: Literal['qpoases','ipopt'] = 'qpoases', use_angular_velocity = True) -> None:

        if solver_name == 'qpoases':
            self.opti = ca.Opti('conic')
        elif solver_name == 'ipopt':
            self.opti = ca.Opti()
        else:
            raise ValueError(f"Solver {solver_name} not implemented")
        self.solver_name = solver_name
        self.casadi_plant = casadi_plant
        self.constraints = []
        if use_angular_velocity:
            self._num_velocities = casadi_plant.num_velocities()
            self.jacobian_wrt_to_variable = JacobianWrtVariable.kV
        else:
            self._num_velocities = casadi_plant.num_positions()
            self.jacobian_wrt_to_variable = JacobianWrtVariable.kQDot
        
        self.last_solution = None
        self.positions = self.opti.parameter(casadi_plant.num_positions(),1) # q of the plant
        self.velocities = self.opti.variable(self.num_velocities(),1) # q_dot of the plant
        self.parameters = {'positions':self.positions}
        self.variables = {'velocities':self.velocities}
        self.cost = 0
        self.solver = None
        self.cse = cse
    def num_velocities(self):
        return self._num_velocities
    def num_positions(self):
        return self.casadi_plant.num_positions()
    def create_parameter(self, num_rows, num_columns, name_parameter: str = None) -> ca.MX:
        p = self.opti.parameter(num_rows, num_columns)
        if name_parameter is not None:
            self.parameters[name_parameter] = p
        else:
            self.parameters[f"parameter_{len(self.parameters.keys())}"] = p
        self.solver = None
        return p
    
    def create_variable(self, num_rows, num_columns, name_variable: str = None) -> ca.MX:

        p = self.opti.variable(num_rows, num_columns)
        if name_variable is not None:
            
            self.variables[name_variable] = p
        else:          
            name_variable = f"variable_{len(self.variables)}"                                                                                    
            self.variables[name_variable] = p
        self.solver = None
        
        return p
    def set_variable_bounds(self,variable, lbx:Union[float,Iterable],ubx:Union[float,Iterable]):
        if isinstance(lbx,(float,int)):
            lbx = [lbx]
        if isinstance(ubx,(float,int)):
            ubx = [ubx]
        for i in range(0,len(lbx)):
            self.constraints.append(self.opti.bounded(lbx[i],variable[i],ubx[i]))
        self.solver = None
            
    # def set_position_
    def add_jacobian_times_q_dot_constraint(self, frame_1, point, frame_2, lower_bound: Iterable = None, upper_bound: Iterable = None):
        """
        Constrains the problem such that `J@q_dot` is between `lower_bound` and `upper_bound`, where `J` is jacobian of the `point` in `frame_1` with respect to `frame_2` and `q_dot` is the generalized velocity of the plant.  
        """
        # J_between_object_and_EE_1 = (casadi_plant.calc_spatial_velocity_jacobian(q.nz,JacobianWrtVariable.kQDot, EE_frame_1,[0,0,0],carried_object_EE_1_frame,world_frame,clean_small_coeffs=1e-2))
        if lower_bound is None:
            lower_bound = np.array([-np.inf, -np.inf, -np.inf,-np.inf, -np.inf, -np.inf])
        if upper_bound is None:
            upper_bound = np.array([np.inf, np.inf, np.inf,np.inf, np.inf, np.inf])
        world_frame = self.casadi_plant.plant.world_frame()
        J = (self.casadi_plant.calc_spatial_velocity_jacobian(self.positions.nz,self.jacobian_wrt_to_variable, frame_1,point,frame_2,world_frame,clean_small_coeffs=1e-6))
        
        expr = J@self.velocities
        
        for i,c in enumerate(expr.nz):
            self.constraints.append(self.opti.bounded(lower_bound[i],c,upper_bound[i]))
        self.solver = None
    def add_constraint(self, constraint: Union[ca.MX, ca.SX, ca.DM],lower_bound: Union[float,Iterable] = None,upper_bound: Union[float,Iterable] = None):
        assert isinstance(constraint, (ca.MX, ca.SX, ca.DM))
        if lower_bound is None:
            lower_bound = -np.ones(constraint.shape)*np.inf
        if upper_bound is None:
            upper_bound = np.ones(constraint.shape)*np.inf
        if isinstance(lower_bound,(float,int)):
            lower_bound = [lower_bound]
        if isinstance(upper_bound,(float,int)):
            upper_bound = [upper_bound]
        for i,c in enumerate(constraint.nz):
            self.constraints.append(self.opti.bounded(lower_bound[i],c,upper_bound[i]))
        # print()
    def add_cost(self, cost: Union[ca.MX, ca.SX, ca.DM]):
        assert isinstance(cost, (ca.MX, ca.SX, ca.DM))
        if isinstance(cost, (ca.MX, ca.SX, ca.DM)):
            self.cost += cost
    def add_quadratic_error_cost(self,variable:Union[ca.MX, ca.SX, ca.DM],weight_matrix:Union[ca.MX, ca.SX, ca.DM,np.ndarray,list,tuple],offset:Union[ca.MX, ca.SX, ca.DM, np.ndarray,list,tuple]):
        """
        Adds a quadratic cost of the form `(variable-offset).T@weight_matrix@(variable-offset)` to the optimization problem.
        """
        assert isinstance(variable, (ca.MX, ca.SX, ca.DM))
        assert isinstance(weight_matrix, (ca.MX, ca.SX, ca.DM,np.ndarray,list,tuple))
        assert isinstance(offset, (ca.MX, ca.SX, ca.DM,np.ndarray,list,tuple))
        
        self.cost += (variable-offset).T@weight_matrix@(variable-offset) 
    
    def make_solver(self,solver_options:Dict=None,):
        """
        Initializes the solver object (either 'qpoases' or 'ipopt', set by `solver_name` argument) using the provided `solver_options`. If `jit` is set to True, it compiles immediately (doesn't work with `qpoases`).
        
        The solver will be stored in `self.solver`. It is a function of the initial guesses for the variables and the parameters in the format:
            `decision_variables = self.solver(*initial_guesses_variables,*parameters)`
        
        where decision_variables is a list of the decision variables in the order they were created.
        
        Helper function:
        
        solve(self,initial_guesses: Dict[ca.MX,np.ndarray] = None, parameters: Dict[ca.MX,np.ndarray] = None)
        """
        if solver_options is None:
            solver_options = {}
        self.opti.solver(self.solver_name,solver_options)
        self.last_solution = None
        if self.cse:
            g = ca.cse(ca.vertcat(*self.constraints))

            self.opti.minimize(ca.cse(self.cost))
        else:
            self.opti.minimize(self.cost)
            g = ca.vertcat(*self.constraints)
        self.opti.subject_to(g.nz)
        parameters = list(self.parameters.values())
        variables = list(self.variables.values())
        
        self.solver = self.opti.to_function('nlp', 
                                [
                                    *variables,
                                    *parameters
                                    ], 
                                [
                                    *variables
                                    ],
                                )
        # TODO: use the other opti.to_function method to name the inputs and outputs of the function
        return self.solver
            
    def solve(self,initial_guesses: Dict[ca.MX,np.ndarray] = None, parameters: Dict[ca.MX,np.ndarray] = None):
        """
        Solve the optimization problem.
        Initial guesses for the variables can be provided in the `initial_guesses` dictionary. Any variable not specified will be initialized with zeros.
        All the parameters must be specified in the `parameters` dictionary.
        
        """
        if initial_guesses is None:
            initial_guesses = {}
            
        # puts the initial guesses in the right order and fills any non specified variables with zeros
        guess_input = []
        for key,value in self.variables.items():
            if not (value in initial_guesses):
                guess_input.append(np.zeros(value.shape))
            else:
                guess_input.append(initial_guesses[value])
        if parameters is None:
            parameters = {}
        parameter_input = []
        missing = []
        for key,value in self.parameters.items():
            if not (value in parameters):
                missing.append(key)
            else:
                parameter_input.append(parameters[value])
        assert len(missing) == 0, f"Missing parameters: {missing}"
        
        assert self.solver is not None, "Solver not created, call make_solver() first"
        
        result = self.solver(*(guess_input + parameter_input))
        
        return {key:value for key,value in zip(self.variables.values(),result)}
