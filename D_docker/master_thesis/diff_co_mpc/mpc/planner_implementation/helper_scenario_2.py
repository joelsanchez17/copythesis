from diff_co.geometrical_model import GeometricalModel
from mpc.helper_functions import BSplineFitWithInitialConstraint
from utils.math.BSpline import BSpline
from utils.my_drake.misc import VisualizerHelper
from mpc.planner_implementation.helper_mixin import PlannerMixin
from mpc.optimisation import OptimizationData, OptimizationInfo
from mpc.planner_implementation.lcm_handler import LCMSubscriptionHandler, SVM
from mpc.planner_implementation.ros_handler import ROSHandler
import numpy as np
import torch
from pydrake.all import DrakeLcm, Sphere
import threading, time
import typing as T
from enum import Enum
# print('aaa')
class Trajectory(T.TypedDict):
    bspline_1: BSpline
    bspline_2: BSpline
    bspline_1_dot: BSpline
    bspline_2_dot: BSpline
    bspline_obj: BSpline
    duration: float
    start_time: float
    grasping_x_EE_1: float
    grasping_x_EE_2: float
class Planner(PlannerMixin):
    class PlannerState(Enum):
        PLAN = 0
        REPLAN = 1
        IDLE = 2
    collision_model_robot_1: GeometricalModel
    collision_model_robot_2: GeometricalModel
    initial_plan_data: OptimizationData
    parallel_solve_data: OptimizationData
    replan_data: OptimizationData
    lcm_subscription_handler: LCMSubscriptionHandler
    ros_handler: ROSHandler
    state: PlannerState
    planned_trajectory: Trajectory
    implemented_trajectory: Trajectory
    current_trajectory_robot_1_final_position: np.ndarray
    current_trajectory_robot_2_final_position: np.ndarray
    current_trajectory_robot_1_final_normalized_velocity: np.ndarray
    current_trajectory_robot_2_final_normalized_velocity: np.ndarray
    last_planning_time: float
    max_speed: float
    min_time: float
    _stop: bool
    _ignore_vision: bool = False
    def __init__(self, opt_collision:OptimizationInfo, meshcat, lcm_subscription_handler_configuration: dict[str,T.Any], ros_handler: ROSHandler, viz_helper:VisualizerHelper, num_parallel_plans):
        self.meshcat = meshcat
        self.robot_1_inverse_kinematics = opt_collision.robots[0].inverse_kinematics
        self.robot_2_inverse_kinematics = opt_collision.robots[1].inverse_kinematics
        self.robot_1_collision_model = opt_collision.robots[0].collision_model
        self.robot_2_collision_model = opt_collision.robots[1].collision_model
        self.initial_plan_data = opt_collision.make_optimization_data()
        self.parallel_solve_data = opt_collision.make_optimization_data(num_parallel_plans)
        self.replan_data = opt_collision.make_optimization_data()
        self.num_control_points_mpc = opt_collision.num_control_points
        self.order_mpc = opt_collision.order
        self.viz_helper = viz_helper
        
        self.parallel_solve_data.bufferize_solver_and_bounds_g(
            opt_collision.solver.map(num_parallel_plans, "openmp"),
            
            opt_collision.lbg_ubg_func.map(num_parallel_plans),
            opt_collision.lbx_ubx_func.map(num_parallel_plans),
        )
        self.initial_plan_data.bufferize_solver_and_bounds_g(
            opt_collision.solver,
            opt_collision.lbg_ubg_func,
            opt_collision.lbx_ubx_func,
        )
        self.replan_data.bufferize_solver_and_bounds_g(
            opt_collision.solver,
            opt_collision.lbg_ubg_func,
            opt_collision.lbx_ubx_func,
        )

        self.planned_trajectory = Trajectory()
        self.implemented_trajectory = Trajectory()
        self.num_steps_per_trajectory = 600
        self.time_per_message = 0.05
        self.max_speed = 0.5
        self.min_time = 0.1
        self.collision_groups = {'robot_1': self.robot_1_collision_model.groups, 'robot_2': self.robot_2_collision_model.groups}
        self.lcm_subscription_handler = LCMSubscriptionHandler(**lcm_subscription_handler_configuration,meshcat= meshcat , collision_groups = self.collision_groups)
        self.ros_handler = ros_handler
        self.state = self.PlannerState.PLAN
        self.vision_weight = 0
        self._stop = True
        self.last_planning_time = 0.3
        self.num_solves = 1
        self.simulation = False
        self.num_control_points_fit,self.order_fit,self.number_of_positions,self.number_of_samples_fit = opt_collision.num_control_points, opt_collision.order, 7, 41
        self.fit_bspline_initial_constraint = BSplineFitWithInitialConstraint(self.num_control_points_fit,self.order_fit,self.number_of_positions,self.number_of_samples_fit)
        self.replan_transition_robot_slack = 0.007
        self.replan_transition_obj_slack = 0.05
        # self.fit_bspline_initial_constraint = BSplineFitWithInitialConstraint(self.num_control_points_fit,self.order_fit,self.number_of_positions,self.number_of_samples_fit)
        # self._stop_event = threading.Event()
        # self._new_plan_event = threading.Event()
        # # self.ros_thread = threading.Thread(target=self.thread_ros_trajectory_sender)
        # self.ros_thread.daemon = True
        # self.ros_thread.start()
        self.set_fixed_parameters()
    def set_time_bounds_replan(self, t_min = None, t_max = None):
        if t_min is not None:
            self.replan_data.set_parameter("t_min", np.array(t_min))
        if t_max is not None:
            self.replan_data.set_parameter("t_max", np.array(t_max))
    def set_time_bounds_plan(self, t_min = None, t_max = None):
        if t_min is not None:
            self.initial_plan_data.set_parameter("t_min", np.array(t_min))
            self.parallel_solve_data.set_parameter("t_min", np.array(t_min))
        if t_max is not None:
            self.initial_plan_data.set_parameter("t_max", np.array(t_max))
            self.parallel_solve_data.set_parameter("t_max", np.array(t_max))

    def set_simulation(self, simulation):
        self.simulation = simulation
    def set_plan_grasping_bounds(self,grasping_x_EE_1,grasping_x_EE_2):
        for opt_data in [self.initial_plan_data, self.parallel_solve_data]:
            opt_data.set_parameter("grasping_EE_1_bounds",grasping_x_EE_1)
            opt_data.set_parameter("grasping_EE_2_bounds",grasping_x_EE_2)
    def set_replan_grasping_position(self,grasping_x_EE_1,grasping_x_EE_2):
        self.replan_data.set_parameter("grasping_EE_1_bounds",np.repeat(np.array(grasping_x_EE_1,dtype= np.float64),2))
        self.replan_data.set_parameter("grasping_EE_2_bounds",np.repeat(np.array(grasping_x_EE_2,dtype= np.float64),2))
    def set_max_speed(self, max_speed):
        self.max_speed = max_speed
    def set_min_time(self, min_time):
        self.min_time = min_time
    def get_s(self, t):
        return (t - self.implemented_trajectory['start_time']) / self.implemented_trajectory['duration']
    def mend_trajectories(self,time,bspline_1,bspline_2):
        planned_q_1 = bspline_1.fast_batch_evaluate(np.linspace(0,1,self.number_of_samples_fit))
        planned_q_2 = bspline_2.fast_batch_evaluate(np.linspace(0,1,self.number_of_samples_fit))
        s = self.get_s(time)
        planned_q_1[0] = self.implemented_trajectory['bspline_1'].evaluate(s)
        planned_q_2[0] = self.implemented_trajectory['bspline_2'].evaluate(s)
        bspline_1_mended = self.fit_bspline_initial_constraint(planned_q_1.T,self.implemented_trajectory['bspline_1_dot'].evaluate(s).reshape(-1,1))
        bspline_2_mended = self.fit_bspline_initial_constraint(planned_q_2.T,self.implemented_trajectory['bspline_2_dot'].evaluate(s).reshape(-1,1))
        return bspline_1_mended,bspline_2_mended
    def start_replaning(self):
        self.state = self.PlannerState.REPLAN
        # self._new_plan_event.set()
    def set_costs(self, vision_cost, duration_cost, acceleration_cost, manipulability_cost, replan_connection_cost,slack_cost_weight,end_position_slack_cost_weight ):
        self.vision_weight = np.array(vision_cost)
        for opt_data in [self.initial_plan_data, self.parallel_solve_data, self.replan_data]:
            # opt_data.set_parameter("gaze_cost_weight", vision_cost) #changed in visibility function
            opt_data.set_parameter("duration_cost", np.array(duration_cost))
            opt_data.set_parameter("acceleration_cost_weight", np.array(acceleration_cost))
            opt_data.set_parameter("manipulability_weight", np.array(manipulability_cost))
            opt_data.set_parameter("slack_cost_weight", np.array(slack_cost_weight))
            opt_data.set_parameter("end_position_slack_cost_weight", np.array(end_position_slack_cost_weight))
        self.replan_data.set_parameter("replan_connection_cost", np.array(replan_connection_cost))
    def parallel_plans(self):
        self.set_start_position()
        self.set_end_position()
        self.update_collision_svm(self.PlannerState.PLAN)
        self.parallel_solve_data.solve(warm=True, recompute_bounds=True)
    def set_collision_bounds(self,lbg,ubg):
        for opt_data in [self.initial_plan_data, self.parallel_solve_data, self.replan_data]:
            for robot_name, group_names in self.collision_groups.items():
                for group_name in group_names:
                    name = f'{robot_name}_{group_name}_svm'
                    opt_data.set_parameter(
                        f"{name}_lbg",
                        np.array([float(lbg)]),
                    )
                    opt_data.set_parameter(
                        f"{name}_ubg",
                        np.array([float(ubg)]),
                    )
    def start(self):
        self._stop = False
    def stop(self):
        self._stop = True
        self.state = self.PlannerState.IDLE
        self._new_plan_event.clear()
        self.ros_handler.cancel_current_goal()
    def ignore_vision(self):
        self._ignore_vision = True
    def unignore_vision(self):
        self._ignore_vision = False
    def set_velocity_scaling(self, scale):
        for opt_data in [self.initial_plan_data, self.parallel_solve_data, self.replan_data]:
            opt_data.set_parameter(
                "robot_1_lower_limits_velocity",
                self.joint_vel_lower_limits*scale
                ,
            )
            opt_data.set_parameter(
                "robot_2_lower_limits_velocity",
                self.joint_vel_lower_limits*scale
                ,
            )
            opt_data.set_parameter(
                "robot_1_upper_limits_velocity",
                self.joint_vel_upper_limits*scale
                ,
            )
            opt_data.set_parameter(
                "robot_2_upper_limits_velocity",
                self.joint_vel_upper_limits*scale
                ,
            )
            opt_data.set_parameter(
                "object_lower_limits_velocity",
                np.array([-1.] * 7) * scale
                ,
            )
            opt_data.set_parameter(
                "object_upper_limits_velocity",
                np.array([1.] * 7) * scale
                ,
            )
    def recompute_replan_bounds(self,q_1,q_2,q_obj,q_1_dot,q_2_dot):
        self.update_collision_svm(self.PlannerState.REPLAN)
        self.update_vision_svm()
        # 'and' because the placing spot is "seen" if ignoring vision
        self.set_placing_spot_visibility(self.lcm_subscription_handler.is_placing_spot_hidden or  (self._ignore_vision))

        # self.replan_data.set_parameter(
        #     "robot_1_initial_configuration",
        #     q_1[:7],
        # )
        # self.replan_data.set_parameter(
        #     "robot_2_initial_configuration",
        #     q_2[:7],
        # )
        self.replan_data.set_parameter(
            "robot_1_initial_velocity",
            q_1_dot,
        )
        self.replan_data.set_parameter(
            "robot_2_initial_velocity",
            q_2_dot,
        )
        self.replan_data.set_parameter(
            "robot_1_initial_position",
            q_1[:7],
        )
        self.replan_data.set_parameter(
            "robot_2_initial_position",
            q_2[:7],
        )
        # robot_1_initial_position

        self.replan_data.bounds_eval()
        self.replan_data.bounds_eval_var()
        # NOTE: this should be calculated in the bounds_eval_var, but here we are
        self.set_robot_control_points_lbx_ubx(self.replan_data,q_1,q_2,q_obj)
    def recompute_control_points_warm_start(self, t):
        s = self.get_s(t)
        s_warm_start = np.linspace(s, 1, self.number_of_samples_fit)
        q_1 = self.implemented_trajectory['bspline_1'].fast_batch_evaluate(s_warm_start)
        q_2 = self.implemented_trajectory['bspline_2'].fast_batch_evaluate(s_warm_start)
        q_obj = self.planned_trajectory['bspline_obj'].fast_batch_evaluate(s_warm_start)
        q_1_dot = self.implemented_trajectory['bspline_1_dot'].evaluate(s)
        q_2_dot = self.implemented_trajectory['bspline_2_dot'].evaluate(s)
        q_obj_dot = self.planned_trajectory['bspline_obj_dot'].evaluate(s)
        bspline_obj_warm_start = self.fit_bspline_initial_constraint(q_obj.T,q_obj_dot.reshape(-1,1))
        bspline_1_warm_start = self.fit_bspline_initial_constraint(q_1.T,q_1_dot.reshape(-1,1))
        bspline_2_warm_start = self.fit_bspline_initial_constraint(q_2.T,q_2_dot.reshape(-1,1   ))
        self.replan_data.set_initial_guess(
            "robot_1_control_points", bspline_1_warm_start.control_points
        )
        self.replan_data.set_initial_guess(
            "robot_2_control_points", bspline_2_warm_start.control_points
        )
        self.replan_data.set_initial_guess(
            "carried_object_control_points", bspline_obj_warm_start.control_points
        )
    def set_start_position(self):
        # change ojbect_start_position,rottaion_matrix
        for opt_data in [self.initial_plan_data, self.parallel_solve_data, self.replan_data]:
            opt_data.set_parameter("object_start_position", self.initial_pose[:3,3:4])
            opt_data.set_parameter("object_start_rotation_matrix", self.initial_pose[:3,:3])

    def set_end_position(self):
        for opt_data in [self.initial_plan_data, self.parallel_solve_data, self.replan_data]:
            opt_data.set_parameter("object_end_position", self.placing_spot_pose[:3,3:4])
            opt_data.set_parameter("object_end_rotation_matrix", self.placing_spot_pose[:3,:3])
    def replan(self,num_solves):
        t_start = time.perf_counter()
        
        next_t = time.perf_counter() + self.last_planning_time
        next_s = self.get_s(next_t)
        q_1 = self.implemented_trajectory['bspline_1'].evaluate(next_s)
        q_2 = self.implemented_trajectory['bspline_2'].evaluate(next_s)
        q_obj = self.implemented_trajectory['bspline_obj'].evaluate(next_s)
        q_1_dot = self.implemented_trajectory['bspline_1_dot'].evaluate(next_s)/self.implemented_trajectory['duration']
        q_2_dot = self.implemented_trajectory['bspline_2_dot'].evaluate(next_s)/self.implemented_trajectory['duration']
        self.set_placing_spot_visibility(self.lcm_subscription_handler.is_placing_spot_hidden or  (self._ignore_vision))

        self.recompute_replan_bounds(q_1,q_2,q_obj,q_1_dot,q_2_dot)
        # self.recompute_control_points_warm_start(next_t)
        self.replan_data.solve(
            warm=True, recompute_bounds=False
        )
        for z in range(num_solves - 1):
            self.replan_data.solve(
                warm=True, recompute_bounds=False
            )

        self._new_plan_event.set()
        self.last_planning_time = time.perf_counter() - t_start

        bspline_1 = BSpline(
            self.replan_data.result["robot_1_control_points"],
            self.order_mpc,
        )
        bspline_2 = BSpline(
            self.replan_data.result["robot_2_control_points"],
            self.order_mpc,
        )
        bspline_obj = BSpline(
            self.replan_data.result["carried_object_control_points"],
            self.order_mpc,
        )

        self.planned_trajectory['bspline_1'] = bspline_1
        self.planned_trajectory['bspline_2'] = bspline_2
        self.planned_trajectory['bspline_obj'] = bspline_obj        
        self.planned_trajectory['bspline_obj_dot'] = bspline_obj.fast_create_derivative_spline()   
        self.planned_trajectory['duration'] = self.replan_data.result['duration'].item()
        self.planned_trajectory['start_time'] = time.perf_counter()

    def initial_plan(self):
        # self.lcm_subscription_handler.is_placing_spot_hidden = True
        self.set_start_position()
        self.set_end_position()
        self.update_collision_svm(self.PlannerState.PLAN)
        self.initial_plan_data.solve(warm=True, recompute_bounds=True)
    def setup_replan(self):
        # Trajectory.grasping_EE_1, Trajectory.grasping_EE_2, fix value
        # copy solve values to initial_guess
        # grasping_x_EE_1 = self.initial_plan_data.result["grasping_x_EE_1"]
        # grasping_x_EE_2 = self.initial_plan_data.result["grasping_x_EE_2"]
        # self.replan_data.set_parameter("grasping_EE_1_bounds",grasping_x_EE_1)
        # self.replan_data.set_parameter("grasping_EE_2_bounds",grasping_x_EE_2)
        self.replan_data.lam_g0[:] = self.initial_plan_data.lam_g
        self.replan_data.lam_x0[:] = self.initial_plan_data.lam_x
        self.replan_data.x0[:] = self.initial_plan_data.x
        bspline_1 = BSpline(
            self.initial_plan_data.result["robot_1_control_points"],
            self.order_mpc,
        )
        bspline_2 = BSpline(
            self.initial_plan_data.result["robot_2_control_points"],
            self.order_mpc,
        )
        bspline_obj = BSpline(
            self.initial_plan_data.result["carried_object_control_points"],
            self.order_mpc,
        )
        bspline_1_dot = bspline_1.fast_create_derivative_spline()
        bspline_2_dot = bspline_2.fast_create_derivative_spline()
        bspline_obj_dot = bspline_obj.fast_create_derivative_spline()
        # speed_scaling, actual_time = self.calculate_velocity_scaling(self.initial_plan_data.result['duration'].item(), self.max_speed, self.min_time, bspline_1_dot, bspline_2_dot)


        self.implemented_trajectory['bspline_1'] = bspline_1
        self.implemented_trajectory['bspline_2'] = bspline_2
        self.implemented_trajectory['bspline_obj'] = bspline_obj
        self.implemented_trajectory['bspline_1_dot'] = bspline_1_dot
        self.implemented_trajectory['bspline_2_dot'] = bspline_2_dot
        self.implemented_trajectory['bspline_obj_dot'] = bspline_obj_dot
        self.implemented_trajectory['duration'] = self.initial_plan_data.result['duration'].item()
        self.implemented_trajectory['start_time'] = time.perf_counter()
        self.planned_trajectory['bspline_1'] = bspline_1
        self.planned_trajectory['bspline_2'] = bspline_2
        self.planned_trajectory['bspline_obj'] = bspline_obj
        self.planned_trajectory['bspline_1_dot'] = bspline_1_dot
        self.planned_trajectory['bspline_2_dot'] = bspline_2_dot
        self.planned_trajectory['bspline_obj_dot'] = bspline_obj_dot
        self.planned_trajectory['duration'] = self.initial_plan_data.result['duration'].item()
        self.planned_trajectory['start_time'] = time.perf_counter()
        
    
    @torch.compile(dynamic=True,)
    def check_path_for_collision(self):
        assert False
        pass
    
    def stop_recording_simulation(self):
        q_1s = np.concatenate(self.q_1s,axis= 0 )
        q_2s = np.concatenate(self.q_2s,axis= 0 )
        q_objs = np.concatenate(self.q_obj,axis= 0 )
        ts = np.concatenate(self.ts,axis= 0 )
        self.meshcat.StartRecording(frames_per_second=30.0)
        for i in range(q_1s.shape[0]):
            q_1 = q_1s[i]
            q_1 = np.concatenate([q_1,np.zeros(2)])

            q_2 = q_2s[i]
            q_2 = np.concatenate([q_2,np.zeros(2)])
            q_obj = q_objs[i]
            self.viz_helper.set_position('robot_0',q_1)
            self.viz_helper.set_position('robot_1',q_2)
            self.viz_helper.set_position('carried_object',q_obj)
            self.viz_helper.diagram_context.SetTime(ts[i] - ts[0])
            self.viz_helper.publish()
        for i,q_obj in enumerate(self.q_obj):
            q_first = q_obj[0]
            q_last = q_obj[-1]
            self.meshcat.SetObject(
                f'/first_point/{i}', Sphere(0.01)
            )
            self.meshcat.SetProperty(
                f'/first_point/{i}', 'position', q_first[4:]
            )
            # self.meshcat.SetProperty(f"thing/{i}", "color",[1,0,0,1])
            self.meshcat.SetProperty(f"/first_point/{i}", "color", [1, 0, 0, 0.5])
            # print( q_first[3:])
            self.meshcat.SetObject(
                f'/last_point/{i}', Sphere(0.01)
            )
            self.meshcat.SetProperty(
                f'/last_point/{i}', 'position', q_last[4:]
            )
            self.meshcat.SetProperty(f"/last_point/{i}", "color", [0,1, 0, 0.5])
        self.meshcat.StopRecording()
        self.meshcat.PublishRecording()
    def start_recording_simulation(self):
        self.q_1s = []
        self.q_2s = []
        self.q_obj = []
        self.ts = []
        self.meshcat.StartRecording(frames_per_second=30.0)