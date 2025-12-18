from diff_co.geometrical_model import GeometricalModel
from mpc.optimisation import OptimizationData, OptimizationInfo
from mpc.planner_implementation.lcm_handler import LCMSubscriptionHandler, SVM
from mpc.planner_implementation.ros_handler import ROSHandler
import numpy as np
import torch
from pydrake.all import DrakeLcm
import threading
import typing as T
from enum import Enum

class PlannerMixin:
    initial_plan_data: OptimizationData
    parallel_solve_data: OptimizationData
    replan_data: OptimizationData
    

    def set_placing_spot_visibility(self, placing_spot_visible, state = None):
        if placing_spot_visible:
            end_position_lower_bounds = np.zeros(3,)
            end_position_upper_bounds = np.zeros(3,)
            end_angle_bounds = np.array([0.00])
            vision_weight = 0
        else:
            end_position_lower_bounds = (
                -np.ones(
                    3,
                )
                * 1.5
            )
            end_position_upper_bounds = (
                np.ones(
                    3,
                )
                * 1.5
            )
            vision_weight = self.vision_weight
            end_angle_bounds = np.array([np.pi])
        if state == self.PlannerState.PLAN:
            for opt_data in [self.initial_plan_data, self.parallel_solve_data, self.replan_data]:
                opt_data.set_parameter(
                    "gaze_cost_weight", np.array(vision_weight))
                opt_data.set_parameter(
                    "end_position_lower_bounds",
                    end_position_lower_bounds,
                )
                opt_data.set_parameter(
                    "end_position_upper_bounds",
                    end_position_upper_bounds,
                )
                opt_data.set_parameter(
                "end_angle_bounds",
                end_angle_bounds,
            )
        elif state == self.PlannerState.REPLAN:
            for opt_data in [self.replan_data]:
                opt_data.set_parameter(
                    "gaze_cost_weight", np.array(vision_weight))
                opt_data.set_parameter(
                    "end_position_lower_bounds",
                    end_position_lower_bounds,
                )
                opt_data.set_parameter(
                    "end_position_upper_bounds",
                    end_position_upper_bounds,
                )
                opt_data.set_parameter(
                "end_angle_bounds",
                end_angle_bounds,
            )
    def set_fixed_parameters(self):
        self.joint_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,])
        self.joint_upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,])
        self.joint_vel_lower_limits = np.array([-2.175, -2.175, -2.175, -2.175, -2.61 , -2.61 , -2.61 ,])
        self.joint_vel_upper_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.61 , 2.61 , 2.61 ,])
        self.joint_acc_lower_limits = np.array([-15. ,  -7.5, -10. , -12.5, -15. , -20. , -20. ,  ])
        self.joint_acc_upper_limits = np.array([15. ,  7.5, 10. , 12.5, 15. , 20. , 20. , ])
        for opt_data in [self.initial_plan_data,self.parallel_solve_data,self.replan_data]:
            if opt_data is self.replan_data:
                bound_robot_initial_configuration = 0.
                bound_robot_initial_velocity = 0.005
                bound_object_initial_configuration = np.inf
                start_position_bounds = np.inf
                start_orientation_bounds = np.pi
            else:
                bound_robot_initial_configuration = np.inf
                bound_object_initial_configuration = np.inf
                bound_robot_initial_velocity = 0.000
                start_orientation_bounds = 0
                start_position_bounds = 0.000
            # opt_data.set_parameter(
            #     "robot_1_initial_acceleration_lbg", np.array([-np.inf] * 7)
            # )
            # opt_data.set_parameter(
            #     "robot_2_initial_acceleration_lbg", np.array([-np.inf] * 7)
            # )
            # opt_data.set_parameter(
            #     "object_initial_acceleration_lbg", np.array([-np.inf] * 7)
            # )
            # opt_data.set_parameter(
            #     "robot_1_initial_acceleration_ubg", np.array([np.inf] * 7)
            # )
            # opt_data.set_parameter(
            #     "robot_2_initial_acceleration_ubg", np.array([np.inf] * 7)
            # )
            # opt_data.set_parameter(
            #     "object_initial_acceleration_ubg", np.array([np.inf] * 7)
            # )
            opt_data.set_parameter(
                "robot_1_terminal_velocity", np.array([0.] * 7)
            )
            opt_data.set_parameter(
                "robot_2_terminal_velocity", np.array([0.] * 7)
            )
            opt_data.set_parameter(
                "robot_1_terminal_velocity_lbg", np.array([-0.] * 7)
            )
            opt_data.set_parameter(
                "robot_2_terminal_velocity_lbg", np.array([-0.] * 7)
            )
            opt_data.set_parameter(
                "robot_1_terminal_velocity_ubg", np.array([0.] * 7)
            )
            opt_data.set_parameter(
                "robot_2_terminal_velocity_ubg", np.array([0.] * 7)
            )

            # opt_data.set_parameter(
            #     "robot_1_initial_configuration_lbg",
            #     -np.array(bound_robot_initial_configuration),
            # )
            # opt_data.set_parameter(
            #     "robot_2_initial_configuration_lbg",
            #     -np.array(bound_robot_initial_configuration),
            # )
            # opt_data.set_parameter(
            #     "object_initial_configuration_lbg",
            #     -np.array(bound_object_initial_configuration),
            # )
            # opt_data.set_parameter(
            #     "robot_1_initial_configuration_ubg",
            #     np.array(bound_robot_initial_configuration),
            # )
            # opt_data.set_parameter(
            #     "robot_2_initial_configuration_ubg",
            #     np.array(bound_robot_initial_configuration),
            # )
            # opt_data.set_parameter(
            #     "object_initial_configuration_ubg",
            #     np.array(bound_object_initial_configuration),
            # )
            opt_data.set_parameter(
                "robot_1_initial_velocity_lbg",
                np.array([-bound_robot_initial_velocity] * 7),
            )
            opt_data.set_parameter(
                "robot_2_initial_velocity_lbg",
                np.array([-bound_robot_initial_velocity] * 7),
            )
            opt_data.set_parameter(
                "object_initial_velocity_lbg",
                np.array([-np.inf] * 7),
            )
            opt_data.set_parameter(
                "robot_1_initial_velocity_ubg",
                np.array([bound_robot_initial_velocity] * 7),
            )
            opt_data.set_parameter(
                "robot_2_initial_velocity_ubg",
                np.array([bound_robot_initial_velocity] * 7),
            )
            opt_data.set_parameter(
                "object_initial_velocity_ubg",
                np.array([np.inf] * 7),
            )

            opt_data.set_parameter(
                "robot_1_initial_velocity",
                np.array([-0.] * 7),
            )
            opt_data.set_parameter(
                "robot_2_initial_velocity",
                np.array([-0.] * 7),
            )
            opt_data.set_parameter(
                "robot_1_initial_velocity",
                np.array([0.] * 7),
            )
            opt_data.set_parameter(
                "robot_2_initial_velocity",
                np.array([0.] * 7),
            )

            opt_data.set_parameter(
                "robot_1_lower_limits_velocity",
                self.joint_vel_lower_limits
                ,
            )
            opt_data.set_parameter(
                "robot_2_lower_limits_velocity",
                self.joint_vel_lower_limits
                ,
            )
            opt_data.set_parameter(
                "object_lower_limits_velocity",
                np.array([-1.] * 7)
                ,
            )
            opt_data.set_parameter(
                "robot_1_upper_limits_velocity",
                self.joint_vel_upper_limits
                ,
            )
            opt_data.set_parameter(
                "robot_2_upper_limits_velocity",
                self.joint_vel_upper_limits
                ,
            )
            opt_data.set_parameter(
                "object_upper_limits_velocity",
                np.array([1.] * 7)
                ,
            )
            opt_data.set_parameter(
                "robot_1_lower_limits",
                self.joint_lower_limits
                ,
            )
            opt_data.set_parameter(
                "robot_2_lower_limits",
                self.joint_lower_limits
                ,
            )
            opt_data.set_parameter(
                "robot_1_upper_limits",
                self.joint_upper_limits
                ,
            )
            opt_data.set_parameter(
                "robot_2_upper_limits",
                self.joint_upper_limits
                ,
            )

            opt_data.set_parameter(
                "start_position_lower_bounds",
                np.array([-start_position_bounds, -start_position_bounds, -start_position_bounds]),
            )
            opt_data.set_parameter(
                "start_position_upper_bounds",
                np.array([start_position_bounds, start_position_bounds, start_position_bounds]),
            )
            opt_data.set_parameter(
                "start_angle_bounds",
                np.array([start_orientation_bounds]),
            )
            for robot_name, group_names in self.collision_groups.items():
                for group_name in group_names:
                    name = f'{robot_name}_{group_name}_svm'
                    opt_data.set_parameter(
                        f"{name}_lbg",
                        np.array([-np.inf]),
                    )
                    opt_data.set_parameter(
                        f"{name}_ubg",
                        np.array([0.]),
                    )

            opt_data.set_parameter(
                "object_lower_limits", np.array([-1, -1, -1, -1, -2, -2, 0])
            )
            opt_data.set_parameter(
                "object_upper_limits", np.array([1, 1, 1, 1, 2, 2, 2])
            )
            opt_data.set_parameter(
                "grasping_rotation_z_EE_1", np.array(np.pi)
            )
            opt_data.set_parameter(
                "grasping_rotation_z_EE_2", np.array(0)
)
    def set_plan_collision_ubg(self,ubg, state = None):
        for opt_data in [self.initial_plan_data,self.parallel_solve_data]:
            opt_data.set_parameter(
                # robot_2_group_4_svm_ubg
                "robot_1_collision_ubg",
                np.array([ubg]),
            )
            opt_data.set_parameter(
                "robot_2_collision_ubg",
                np.array([ubg]),
            )
    def set_replan_collision_ubg(self,ubg):
        self.replan_data.set_parameter(
            "robot_1_collision_ubg",
            np.array([ubg]),
        )
        self.replan_data.set_parameter(
            "robot_2_collision_ubg",
            np.array([ubg]),
        )
    def set_robot_control_points_lbx_ubx(self,opt_data,q_1,q_2,q_obj):
        robot_1_control_points_var_ubx = (
            opt_data
            .ubx[
                opt_data.variable_slices["robot_1_control_points"]
            ]
            .copy()
            .reshape(
                opt_data.variable_shapes["robot_1_control_points"][
                    1
                ],
                opt_data.variable_shapes["robot_1_control_points"][
                    0
                ],
            )
            .T
        )
        robot_2_control_points_var_ubx = (
            opt_data
            .ubx[
                opt_data.variable_slices["robot_2_control_points"]
            ]
            .copy()
            .reshape(
                opt_data.variable_shapes["robot_2_control_points"][
                    1
                ],
                opt_data.variable_shapes["robot_2_control_points"][
                    0
                ],
            )
            .T
        )

        robot_1_control_points_var_lbx = (
            opt_data
            .lbx[
                opt_data.variable_slices["robot_1_control_points"]
            ]
            .copy()
            .reshape(
                opt_data.variable_shapes["robot_1_control_points"][
                    1
                ],
                opt_data.variable_shapes["robot_1_control_points"][
                    0
                ],
            )
            .T
        )
        robot_2_control_points_var_lbx = (
            opt_data
            .lbx[
                opt_data.variable_slices["robot_2_control_points"]
            ]
            .copy()
            .reshape(
                opt_data.variable_shapes["robot_2_control_points"][
                    1
                ],
                opt_data.variable_shapes["robot_2_control_points"][
                    0
                ],
            )
            .T
        )
        carried_object_control_points_var_lbx = (
            opt_data
            .lbx[
                opt_data.variable_slices["carried_object_control_points"]
            ]
            .copy()
            .reshape(
                opt_data.variable_shapes["carried_object_control_points"][
                    1
                ],
                opt_data.variable_shapes["carried_object_control_points"][
                    0
                ],
            )
            .T
        )
        carried_object_control_points_var_ubx = (
            opt_data
            .ubx[
                opt_data.variable_slices["carried_object_control_points"]
            ]
            .copy()
            .reshape(
                opt_data.variable_shapes["carried_object_control_points"][
                    1
                ],
                opt_data.variable_shapes["carried_object_control_points"][
                    0
                ],
            )
            .T
        )

        robot_1_control_points_var_lbx[:, 0] = q_1[:7] - self.replan_transition_robot_slack
        robot_1_control_points_var_ubx[:, 0] = q_1[:7] + self.replan_transition_robot_slack
        robot_2_control_points_var_lbx[:, 0] = q_2[:7] - self.replan_transition_robot_slack
        robot_2_control_points_var_ubx[:, 0] = q_2[:7] + self.replan_transition_robot_slack
        carried_object_control_points_var_lbx[:, 0] = q_obj[:7] - self.replan_transition_obj_slack
        carried_object_control_points_var_ubx[:, 0] = q_obj[:7] + self.replan_transition_obj_slack
        opt_data.set_variable_bound(
            "robot_1_control_points",
            robot_1_control_points_var_lbx,
            robot_1_control_points_var_ubx,
        )
        opt_data.set_variable_bound(
            "robot_2_control_points",
            robot_2_control_points_var_lbx,
            robot_2_control_points_var_ubx,
        )
        opt_data.set_variable_bound(
            "carried_object_control_points",
            carried_object_control_points_var_lbx,
            carried_object_control_points_var_ubx,
        )
    def update_collision_on_data(self,robot_name,group_name, svm:SVM, opt_data:OptimizationData):
        #  'robot_1_group_3_svm_weights': (1, 128),
        #  'robot_1_group_3_svm_support_vectors': (6, 128),
        #  'robot_1_group_3_svm_polynomial_weights': (7, 1),
        name = f'{robot_name}_{group_name}_svm'
        weights = np.zeros(opt_data.parameter_shapes[name + '_weights'])
        support_vectors = np.zeros(opt_data.parameter_shapes[name + '_support_vectors'])
        polynomial_weights = np.zeros(opt_data.parameter_shapes[name + '_polynomial_weights'])
        weights[:, :svm['weights'].shape[1]] = svm['weights'].reshape(1,-1)
        support_vectors[:, :svm['sv'].shape[1]] = svm['sv']
        ww = svm['pol_weights'].reshape(-1,1)
        polynomial_weights[:ww.shape[0]] = ww
        opt_data.set_parameter(name + '_weights', weights)
        opt_data.set_parameter(name + '_support_vectors', support_vectors)
        opt_data.set_parameter(name + '_polynomial_weights', polynomial_weights)
        
    def update_collision_svm(self, state):
        for robot_name, group_names in self.collision_groups.items():
            # self.last_svm[robot_name] = {}
            for group_name in group_names:
                if len(self.lcm_subscription_handler.last_svm[robot_name][group_name]) > 0:
                    svm: SVM = self.lcm_subscription_handler.last_svm[robot_name][group_name][-1]
                    if state is self.PlannerState.PLAN:
                        self.update_collision_on_data(robot_name,group_name,svm,self.initial_plan_data)
                        self.update_collision_on_data(robot_name,group_name,svm,self.parallel_solve_data)
                    elif state is self.PlannerState.REPLAN:
                        self.update_collision_on_data(robot_name,group_name,svm,self.replan_data)

    def update_vision_svm(self):
        pass
    
    def set_num_steps_per_trajectory(self,num_steps_per_trajectory):
        self.num_steps_per_trajectory = num_steps_per_trajectory


    def calculate_velocity_scaling(self, traj_duration, max_speed, min_time, bspline_1_vel, bspline_2_vel):
        speed_scaling_1 = min(
            1,
            1
            / max(
                (
                    (
                        np.abs(bspline_1_vel.control_points.min(axis=1))
                        / max_speed
                        / traj_duration
                    ).max()
                ),
                (
                    (
                        np.abs(bspline_1_vel.control_points.max(axis=1))
                        / max_speed
                        / traj_duration
                    ).max()
                ),
            ),
        )
        speed_scaling_2 = min(
            1,
            1
            / max(
                (
                    (
                        np.abs(bspline_2_vel.control_points.min(axis=1))
                        / max_speed
                        / traj_duration
                    ).max()
                ),
                (
                    (
                        np.abs(bspline_2_vel.control_points.max(axis=1))
                        / max_speed
                        / traj_duration
                    ).max()
                ),
            ),
        )
        speed_scaling = min(speed_scaling_2, speed_scaling_1)
        actual_time = (traj_duration / speed_scaling)
        correction = max(min_time / actual_time, 1)
        actual_time = max(actual_time, min_time)
        speed_scaling /= correction
        return speed_scaling, actual_time