
# import multiprocessing
from toolz import memoize, pipe, accumulate, groupby, compose, compose_left
from toolz.curried import do
from collections import namedtuple

# import sympy
import casadi as ca
import time
import numpy as np
from utils.misc import *
import typing as T
# from utils.my_drake.casadi.multibody_wrapper import CasadiMultiBodyPlantWrapper

from projects.refactor_mpb.multibody_wrapper import MultiBodyPlantWrapper
from utils.math.BSpline import BSpline

constraint = namedtuple('constraint',['expression','lbg','ubg'])

class KinematicTrajectoryOptimization:
    """
    `TODO:` 
    - Dynamics
    - The derivative bspline bsplines the position. The rotations in the position spline are quaternions. The velocities in the Multibodyplant are angular velocities. Do sth about that.
    """

    def __init__(self, casadi_plant: MultiBodyPlantWrapper, number_of_control_points: int, order: int,cse = True) -> None:
        self.opti = ca.Opti()
        self.casadi_plant = casadi_plant
        self.constraints = []
        self.number_of_positions = casadi_plant.num_positions()
        self.number_of_control_points = number_of_control_points
        # r(s)
        
        self.control_points = self.opti.variable(
            self.number_of_positions, number_of_control_points)
        
        self.duration_variable = self.opti.variable()
        self.variable_slices = {'control_points': [0,self.number_of_positions*self.number_of_control_points],'duration':[self.number_of_positions*self.number_of_control_points,self.number_of_positions*self.number_of_control_points+1]}
        # self.opti.subject_to(self.duration_variable >= 1e-2)
        self.constraints.append(self.duration_variable >= 1e-2)
        
        # q(t) = r(t/T) = r(s)
        self.order = order
        self.degree = order - 1
        self.number_of_knots = number_of_control_points + order
        self.spline = BSpline(
            control_points=self.control_points, order=self.order)
        self.derivative_spline = self.spline.create_derivative_spline()
        self.last_solution = None
        self.parameters = {}
        self.variables = {'control_points':self.control_points,'duration':self.duration_variable}
        self.lbx = [-np.inf] * (number_of_control_points*self.number_of_positions + 1)
        self.ubx = [np.inf] * (number_of_control_points*self.number_of_positions + 1)
        self.lbx[self.variable_slices["duration"][0]] = 0
        self.cost = 0
        self.solver = None
        
        self.solver = None
        self.cse = cse
        
        # self.lower_bounds = []
        # self.upper_bounds = []
    def duration_limit_constraint(self, T):
        assert T >= 1e-2
        # self.opti.subject_to(self.duration_variable <= T)
        # self.ubx[self.variable_slices["duration"][0]] = T
        self.constraints.append(self.duration_variable <= T)
        self.solver = None

    def create_parameter(self, num_rows, num_columns, name_parameter: str = None) -> ca.MX:
        p = self.opti.parameter(num_rows, num_columns)
        if name_parameter is not None:
            self.parameters[name_parameter] = p
        else:
            self.parameters[f"parameter_{len(self.parameters.keys())}"] = p
        self.solver = None
        return p
    def create_variable(self, num_rows, num_columns, name_variable: str = None, lbx:T.Union[float,T.Iterable] = None,ubx:T.Union[float,T.Iterable] = None) -> ca.MX:

        p = self.opti.variable(num_rows, num_columns)
        last_variable = list(self.variables.keys())[-1]
        if name_variable is not None:
            
            self.variable_slices[name_variable] = [self.variable_slices[last_variable][1],self.variable_slices[last_variable][1]+num_rows*num_columns]
            self.variables[name_variable] = p
        else:          
            name_variable = f"variable_{len(self.variable_slices.keys())}"                                                                                    
            self.variable_slices[name_variable] = [self.variable_slices[last_variable][1],self.variable_slices[last_variable][1]+num_rows*num_columns]
            self.variables[name_variable] = p
        self.solver = None
        return p
    def set_variable_bounds(self,variable, lbx:T.Union[float,T.Iterable],ubx:T.Union[float,T.Iterable]):
        if isinstance(lbx,(float,int)):
            lbx = [lbx]
        if isinstance(ubx,(float,int)):
            ubx = [ubx]
        for i in range(0,len(lbx)):
            self.constraints.append(self.opti.bounded(lbx[i],variable[i],ubx[i]))
        self.solver = None
    def add_constraint(self, expr,):
        for i,c in enumerate(expr.nz):
            self.constraints.append(c)
        self.solver = None

    def get_forward_kinematic_translation_constraint_expression(self, q, frame_1, point_1, frame_2, point_2):
        world_frame = self.casadi_plant.plant.world_frame()
        frame_pose_1 = self.casadi_plant.calc_frame_pose_in_frame(q,frame_1,world_frame,clean_small_coeffs=1e-6)
        frame_pose_2 = self.casadi_plant.calc_frame_pose_in_frame(q,frame_2,world_frame,clean_small_coeffs=1e-6)
        translation_1 = frame_pose_1[0:3, 3] + frame_pose_1[0:3, 0:3] @ point_1
        translation_2 = frame_pose_2[0:3, 3] + frame_pose_2[0:3, 0:3] @ point_2
        expr = translation_2 - translation_1
        return expr
    def get_forward_kinematic_orientation_constraint_expression(self, q, frame_name_1, rotation_1, frame_name_2, rotation_2, theta_bound):
        
        
        frame_2_in_1 = self.casadi_plant.calc_frame_pose_in_frame(q,frame_name_2,frame_name_1,clean_small_coeffs=1e-6)
        trace_R1_R2 = ca.trace(rotation_1.T@frame_2_in_1[0:3, 0:3]@rotation_2)
        return trace_R1_R2 - (2*ca.cos(theta_bound) + 1)
    def add_forward_kinematic_translation_constraint(self, s, frame_1, point_1, frame_2, point_2, lower_bound: T.Iterable = None, upper_bound: T.Iterable = None):
        """
        Constraint that makes `point_1` described in `frame_1` coincide with `point_2` described in `frame_2` at time `t`.

        It is implemented as: `point_2 - point_1` (both points described in the `world_frame`) is constrained to be between `lower_bound` and `upper_bound
        """
        if lower_bound is None:
            lower_bound = np.array([-np.inf, -np.inf, -np.inf])
        if upper_bound is None:
            upper_bound = np.array([np.inf, np.inf, np.inf])
        if not isinstance(lower_bound, (ca.MX, ca.SX, ca.DM)):
            lower_bound = ca.vertcat(lower_bound)
        if not isinstance(upper_bound, (ca.MX, ca.SX, ca.DM)):
            upper_bound = ca.vertcat(upper_bound)
        position_at_t = self.spline.evaluate(s)
        world_frame = self.casadi_plant.plant.world_frame()
        frame_pose_1 = self.casadi_plant.calc_frame_pose_in_frame(position_at_t,frame_1,world_frame,clean_small_coeffs=1e-6)
        frame_pose_2 = self.casadi_plant.calc_frame_pose_in_frame(position_at_t,frame_2,world_frame,clean_small_coeffs=1e-6)
        translation_1 = frame_pose_1[0:3, 3] + frame_pose_1[0:3, 0:3] @ point_1
        translation_2 = frame_pose_2[0:3, 3] + frame_pose_2[0:3, 0:3] @ point_2
        
        expr = translation_2 - translation_1
        
        for i,c in enumerate(expr.nz):
            self.constraints.append(self.opti.bounded(lower_bound[i],c,upper_bound[i]))
        self.solver = None

    def add_forward_kinematic_orientation_constraint(self, s, frame_name_1, rotation_1, frame_name_2, rotation_2, theta_bound):
        """
        Constraint that makes the orientation described by `rotation_1` in `frame_1` coincide with the orientation described by `rotation_2` in `frame_2` at time `t` with a maximum angle of `theta_bound` between the orientations.
        
        This is to say that there exists an axis and an angle `theta` such that orientation 1 can be rotated around this axis by that theta to make it coincide with orientation 2. `theta_bound` is the maximum value of `theta` allowed.

        It is implemented as: `trace_R1_R2 >= 2*ca.cos(theta_bound) + 1` where `trace_R1_R2` is the trace of the matrix that rotates orientation 1 to orientation 2. 
        """
        position_at_t = self.spline.evaluate(s)
        frame_2_in_1 = self.casadi_plant.calc_frame_pose_in_frame(position_at_t,frame_name_2,frame_name_1,clean_small_coeffs=1e-6)
        trace_R1_R2 = ca.trace(rotation_1.T@frame_2_in_1[0:3, 0:3]@rotation_2)
        self.constraints.append(trace_R1_R2 >= 2*ca.cos(theta_bound) + 1)
        self.solver = None

    def add_plant_velocity_bounds(self, lower_bound: T.Union[T.Iterable, ca.DM, ca.SX, ca.MX] = None, upper_bound: T.Union[T.Iterable, ca.DM, ca.SX, ca.MX] = None):
        """
        The control points of the derivative spline are constrained to be between `lower_bound` and `upper_bound`.
        """
        if not isinstance(lower_bound, (ca.MX, ca.SX, ca.DM)):
            lower_bound = ca.vertcat(*lower_bound)
        if not isinstance(upper_bound, (ca.MX, ca.SX, ca.DM)):
            upper_bound = ca.vertcat(*upper_bound)
        new_constraints = []
        for i in range(self.number_of_control_points - 1):
            for j in range(self.number_of_positions):
                new_constraints.append(self.opti.bounded(lower_bound[j]*self.duration_variable,self.derivative_spline.control_points[j, i],upper_bound[j]*self.duration_variable))
        self.constraints += new_constraints
        self.solver = None
        return new_constraints

    def add_plant_position_bounds(self, lower_bound: T.Union[T.Iterable, ca.DM, ca.SX, ca.MX] = None, upper_bound: T.Union[T.Iterable, ca.DM, ca.SX, ca.MX] = None):
        """
        The control points of the spline are constrained to be between `lower_bound` and `upper_bound`.
        
        """
        if not isinstance(lower_bound, (ca.MX, ca.SX, ca.DM)):
            lower_bound = ca.vertcat(*lower_bound)
        if not isinstance(upper_bound, (ca.MX, ca.SX, ca.DM)):
            upper_bound = ca.vertcat(*upper_bound)
        new_constraints = []
        for i in range(self.number_of_control_points):
            for j in range(self.number_of_positions):
                
                self.constraints.append(self.opti.bounded(lower_bound[j],self.spline.control_points[j, i],upper_bound[j]))
        self.constraints += new_constraints
        self.solver = None
        return new_constraints
    def add_quaternion_constraint_on_sample(self, s:float):
        """
        At time `s`, the quaternions (in the position spline) of the bodies that have quaternions are constrained to be unitary.
        """
        from pydrake.all import BodyIndex
        position = self.spline.evaluate(s)
        
        for body_index in range(self.casadi_plant.plant.num_bodies()):
            body = self.casadi_plant.plant.get_body(BodyIndex(body_index))
            
            if body.has_quaternion_dofs():
                self.constraints.append(sum(i**2 for i in position[body.floating_positions_start():body.floating_positions_start()+4].nz) == 1)
                # self.opti.subject_to(sum(i**2 for i in position[body.floating_positions_start():body.floating_positions_start()+4].nz) == 1)
        self.solver = None
    def add_quaternion_constraint_on_control_points(self):
        """
        The control points of the quaternions (in the position spline) of the bodies that have quaternions are constrained to be unitary.
        """
        from pydrake.all import BodyIndex
        
        
        
        for body_index in range(self.casadi_plant.plant.num_bodies()):
            body = self.casadi_plant.plant.get_body(BodyIndex(body_index))
            if not body.has_quaternion_dofs():
                continue
            for i in range(0,self.number_of_control_points):
                position = self.spline.control_points[:,i]
                self.constraints.append(sum(i**2 for i in position[body.floating_positions_start():body.floating_positions_start()+4].nz) == 1)
                # self.opti.subject_to(sum(i**2 for i in position[body.floating_positions_start():body.floating_positions_start()+4].nz) == 1)
        self.solver = None
    def add_path_length_cost(self, weight: float = 1.0):
        # upper bound on length, sum of abs of the differences between control points
        q2 = self.spline.control_points[:, 1:]
        q1 = self.spline.control_points[:, :-1]
        self.cost += weight * ca.sum2((ca.sum1(ca.fabs(q2 - q1))))
        # self.cost += weight * ca.sum2((ca.sum1((q2 - q1)**2)))
        self.solver = None
        
        
    def add_position_error_cost_on_sample(self, s:float, reference:T.Iterable,weight: T.Union[T.Iterable,float] = 1.0):
        """
        Add quadratic cost to the position of the plant at time `s` to be close to `reference`.
        """
        if isinstance(weight,(float,int)):
            weight = [weight]*self.number_of_positions
            weight = ca.vertcat(*weight)
        position = self.spline.evaluate(s)
        self.cost +=  ca.sum1(((position - reference)*weight)**2)
        
    def add_position_error_cost_on_control_points(self, reference,weight: T.Union[float,int,T.Iterable] = 1.0):
        if isinstance(weight,(float,int)):
            weight = [weight]*self.number_of_positions
            weight = ca.vertcat(*weight)
            
        for i in range(self.number_of_control_points):
            position = self.spline.control_points[:, i]
            self.cost += ca.sum1(((position - reference)*weight)**2)
    
    # def get_variable_from_solution(self,solution,variable_name):
    #     return solution['x'][self.variable_slices[variable_name][0]:self.variable_slices[variable_name][1]]
    def get_variable_from_decision_variables_vector(self,variable_name):
        return self.opti.x[self.variable_slices[variable_name][0]:self.variable_slices[variable_name][1]]
    def get_variable_from_list(self,variable_name, list_):
        return list_[self.variable_slices[variable_name][0]:self.variable_slices[variable_name][1]]
    # def set_variable_in_list(self,variable_name, list_to,list_from):
    #     """
    #     `list_to: list that will be changed`
    #     `list_from: values will be copied from this list`
        
    #     """
    #     assert len(list_from) == self.variable_slices[variable_name][0]-self.variable_slices[variable_name][1]
    #     for i,val in list_from:
    #         list_to[self.variable_slices[variable_name][0]+i] = val
    def make_solver(self,solver_options:T.Dict=None):        
        """
        Creates the solver object with the `solver_options` provided. If `jit:True`, it will compile already.
        
        The solver will be stored in `self.solver`. It is a function of the initial guesses for the variables and the parameters in the format:
            `decision_variables = self.solver(*initial_guesses_variables,*parameters)`
        
        where decision_variables is a list of the decision variables in the order they were created.
        
        Helper function:
        
        solve(self,initial_guesses: T.Dict[ca.MX,np.ndarray] = None, parameters: T.Dict[ca.MX,np.ndarray] = None)
        """
        if solver_options is None:
            solver_options = {}
        self.last_solution = None

            
        g = ca.cse(ca.vertcat(*self.constraints))

        self.opti.minimize(ca.cse(self.cost))

        self.opti.subject_to(g.nz)
        
        if "iteration_callback" in solver_options:
            solver_options["iteration_callback"].nx = self.opti.nx
            solver_options["iteration_callback"].ng = self.opti.ng
            solver_options["iteration_callback"].np = self.opti.np
            solver_options["iteration_callback"].construct(solver_options["iteration_callback"].name, solver_options["iteration_callback"].opts)
        self.opti.solver('ipopt',solver_options)
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
        
        # eps = ca.MX.sym('epsilon')
        

        # self.constraints = ca.Function('g',variables + parameters,[self.opti.g])
        # self.ubg = ca.Function('ubg',variables + parameters,[self.opti.ubg])
        # self.lbg = ca.Function('lbg',variables + parameters,[self.opti.lbg])
        # self.cost = ca.Function('f',variables + parameters,[self.opti.f])
        # self._constraint_violation = ca.Function('Fg',variables + parameters + [eps],[(ca.logic_and(self.opti.ubg - self.opti.g >= -eps, self.opti.g - self.opti.lbg >= -eps))])
        # self.constraint_violation = lambda *args,eps: all(self._constraint_violation(*args,eps).nz)
        # all(Fg(*result.values(),object_start_pose.translation(),object_end_pose.translation(),object_start_pose.rotation().matrix(),object_end_pose.rotation().matrix(),0.0000001).nz)
        # return self.solver
            
    def solve(self,initial_guesses: T.Dict[ca.MX,np.ndarray] = None, parameters: T.Dict[ca.MX,np.ndarray] = None):
        """
        Solve the optimization problem.
        Initial guesses for the variables can be provided in the `initial_guesses` T.Dictionary. Any variable not specified will be initialized with zeros.
        All the parameters must be specified in the `parameters` T.Dictionary.
        
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
            # self.opti.set_initial(self.variables[key],value)
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

    # def compile(self, solver_options:T.Dict=None):
    #     if solver_options is None:
    #         solver_options = {}
    #     import subprocess
    #     self.opti.minimize(self.cost)
    #     opti = self.opti
    #     objective_function = opti.f
    #     if self.cse:
    #         constraints_function = ca.cse(opti.g)
    #     else:
    #         constraints_function = (opti.g)
    #     decision_variables = opti.x
        
    #     # parameters_objects = ca.vertcat(*[*self.parameters.values()])
    #     parameters_objects = [*self.parameters.values()]
    #     prob = {'f': objective_function, 'x': decision_variables, 'g': constraints_function, 'p': ca.vertcat(*[p.reshape((p.shape[0]*p.shape[1],1)) for p in parameters_objects])}
    #     solver = ca.nlpsol('solver', 'ipopt', prob,solver_options)
    #     # print(solver)
    #     TEMP_FOLDER.mkdir(exist_ok=True)
    #     path_so =  TEMP_FOLDER / ('nlp.so')
    #     path_c = TEMP_FOLDER / ('nlp.c')
        
    #     solver.generate_dependencies('nlp.c')
    #     os.rename(str(pathlib.Path(os.getcwd())/'nlp.c' ), path_c)
    #     # gcc -O3 -march=native -ffast-math -fno-finite-math-only -fPIC -shared -fopenmp -fPIC -c jit_tmpqme2DP.c -o ./tmp_casadi_compiler_shell6mdVvY.o"
    #     # gcc ./tmp_casadi_compiler_shell6mdVvY.o -o ./tmp_casadi_compiler_shell6mdVvY.so -fopenmp -shared"
    #     subprocess.run(["gcc", "-shared", "-Ofast","-march=native","-fopenmp","-ffast-math","-fno-finite-math-only", str(path_c), "-o", str(path_so)])
    #     subprocess.run(["gcc", str(path_so), "-o", str(path_so),"-fopenmp","-shared"])