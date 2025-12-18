import sys
sys.path.append('/workspaces/master_thesis')

from toolz import (
    memoize,
    pipe,
    accumulate,
    groupby,
    compose,
    compose_left,
    merge,
    first,
)
from toolz.curried import do
from collections import namedtuple
import typing as T
import numpy as np
import sympy as sp
import pydrake

import itertools
import torch
import casadi as ca
import pathlib, os
import time
import copy


from pydrake.all import (
    ModelInstanceIndex,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)

from utils.math.BSpline import BSpline

from utils.my_drake.misc import make_namedview_positions, VisualizerHelper
import importlib

import utils.my_casadi.misc as ca_utils
from .plant import plant_from_yaml
from projects.refactor_mpb.kinematic_trajectory import KinematicTrajectoryOptimization
from projects.refactor_mpb.kinematic_optimization import KinematicOptimization
from projects.refactor_mpb.multibody_wrapper import MultiBodyPlantWrapper
import utils.my_casadi.misc as ca_utils
from utils.my_casadi.misc import casadi_frame_A_in_frame_B, veccat, Jit


from diff_co import GeometricalModel
from .inverse_kinematics import FrankaInverseKinematics
from .casadi_custom import (
    replace_hessian,
    jac_chain_rule_with_mapped_outer_function,
    jac_jac_chain_rule_with_mapped_outer_function,
)

from dataclasses import dataclass, field, fields

from typing import TypedDict

# import typeddict
from typing import TypedDict
import yaml


class DiffCoGroupOptions(TypedDict):
    name: str
    num_obstacle_categories: int
    inner_map_size: int
    num_support_points_per_block: int
    outer_map_size: int
    hessian_parallelization: str
    jacobian_parallelization: str


class DiffCoRobotOptions(TypedDict):
    name: str
    groups: list[DiffCoGroupOptions]




class DiffCoOptions(TypedDict):
    robots: list[DiffCoRobotOptions]

    @staticmethod
    def from_yaml(path: pathlib.Path) -> "DiffCoOptions":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        svm_params = data["svm_params"]
        groups = svm_params["groups"]
        robot_1_options = DiffCoRobotOptions(
            name="robot_1",
            groups={
                group_name: DiffCoGroupOptions(
                    name=group_name,
                    num_obstacle_categories=values["num_obstacle_categories"],
                    inner_map_size=values["inner_map_size"],
                    num_support_points_per_block=values["num_support_points_per_block"],
                    outer_map_size=values["outer_map_size"],
                    hessian_parallelization=values["hessian_parallelization"],
                    jacobian_parallelization=values["jacobian_parallelization"],
                )
                for group_name, values in groups.items()
            },
        )
        robot_2_options = DiffCoRobotOptions(
            name="robot_2",
            groups={
                group_name: DiffCoGroupOptions(
                    name=group_name,
                    num_obstacle_categories=values["num_obstacle_categories"],
                    inner_map_size=values["inner_map_size"],
                    num_support_points_per_block=values["num_support_points_per_block"],
                    outer_map_size=values["outer_map_size"],
                    hessian_parallelization=values["hessian_parallelization"],
                    jacobian_parallelization=values["jacobian_parallelization"],
                )
                for group_name, values in groups.items()
            },
        )
        diff_co_options = DiffCoOptions(robots=[robot_1_options, robot_2_options])
        return diff_co_options
class VisionOptions(TypedDict):
    robots: list[DiffCoRobotOptions]

    @staticmethod
    def from_yaml(path: pathlib.Path) -> "VisionOptions":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        svm_params = data["gaze_params"]
        groups = svm_params["groups"]
        robot_1_options = DiffCoRobotOptions(
            name="robot_1",
            groups={
                group_name: DiffCoGroupOptions(
                    name=group_name,
                    # num_obstacle_categories=values["num_obstacle_categories"],
                    inner_map_size=values["inner_map_size"],
                    num_support_points_per_block=values["num_support_points_per_block"],
                    outer_map_size=values["outer_map_size"],
                    hessian_parallelization=values["hessian_parallelization"],
                    jacobian_parallelization=values["jacobian_parallelization"],
                )
                for group_name, values in groups.items()
            },
        )
        
        diff_co_options = VisionOptions(robots=[robot_1_options,])
        return diff_co_options




def make_system(yaml_path, meshcat, CODEGEN_FOLDER, camera = False):
    info = plant_from_yaml(
        yaml_path, camera=camera, meshcat=meshcat, point_cloud_visualizer=False
    )
    plants_info = info["plants_info"]
    yaml = info["yaml"]
    svm_params = yaml["svm_params"]
    gaze_params = yaml["gaze_params"]
    """model instances"""

    complete_plant = plants_info["plant"]
    """all the robots and obstacles for visualization etc"""
    complete_diagram = plants_info["diagram"]

    carried_object_opt = plants_info["carried_object_opt"]
    robot_opt_tuples = plants_info["robot_opt_tuple"]
    """only the robot plants"""

    carried_object = RobotOptimisationInfo(
        diagram=carried_object_opt.diagram,
        plant=carried_object_opt.plant,
        plant_name="carried_object",
        viz_model_instance=carried_object_opt.viz_model_instance,
        wrapper=MultiBodyPlantWrapper(
            carried_object_opt.plant,
            diagram=carried_object_opt.diagram,
            temp_folder=CODEGEN_FOLDER / "carried_object",
        ),
    )
    robots = []
    for i in range(0, 2):
        wrapper = MultiBodyPlantWrapper(
            robot_opt_tuples[i].plant,
            diagram=robot_opt_tuples[i].diagram,
            temp_folder=CODEGEN_FOLDER / f"robot_{i}",
        )

        robot_opt_info = RobotOptimisationInfo()
        robot_opt_info.plant = robot_opt_tuples[i].plant
        robot_opt_info.diagram = robot_opt_tuples[i].diagram
        robot_opt_info.wrapper = wrapper
        robot_opt_info.viz_model_instance = robot_opt_tuples[i].viz_model_instance
        robot_opt_info.plant_name = f"robot_{i}"
        robot_opt_info.collision_model = GeometricalModel(
            "collision", wrapper, svm_params["groups"]
        )
        robot_opt_info.vision_model = GeometricalModel(
            "vision", wrapper, gaze_params["groups"]
        )

        robot_base_pose = (
            robot_opt_info.plant.GetFrameByName("panda_link0")
            .CalcPose(
                robot_opt_info.plant.CreateDefaultContext(),
                robot_opt_info.plant.GetFrameByName("world"),
            )
            .GetAsMatrix4()
        )
        object_frame_to_object_EE_frame = (
            carried_object.plant.GetFrameByName(f"carried_object_EE_{i+1}_frame")
            .CalcPose(
                carried_object.plant.CreateDefaultContext(),
                carried_object.plant.GetFrameByName("carried_object"),
            )
            .GetAsMatrix4()
        )
        robot_opt_info.inverse_kinematics = FrankaInverseKinematics(
            base_pose=robot_base_pose,
            grasping_transform=object_frame_to_object_EE_frame,
        )

        robots.append(robot_opt_info)

    viz_helper = VisualizerHelper(complete_plant, complete_diagram)
    return {
        "complete_plant": complete_plant,
        "robots": robots,
        "carried_object": carried_object,
        "yaml": yaml,
        "visualization_helper": viz_helper,
        'camera_params': info["camera_params"]
    }


@dataclass
class RobotOptimisationInfo:
    plant: pydrake.all.MultibodyPlant = None
    diagram: pydrake.all.Diagram = None
    wrapper: MultiBodyPlantWrapper = None
    collision_model: GeometricalModel = None
    vision_model: GeometricalModel = None
    viz_model_instance: ModelInstanceIndex = None
    # forward_kinematic_kernel: ForwardKinematicKernel = None

    plant_name: str = None
    # forward_kinematic_diff_co: ForwardKinematics = None
    control_points: T.Union[ca.MX, ca.SX] = None
    position_lower_limits: T.Union[ca.MX, ca.SX] = None
    position_upper_limits: T.Union[ca.MX, ca.SX] = None

    # bspline: BSpline = None
    def __copy__(self):
        ob = RobotOptimisationInfo(
            plant=self.plant,
            diagram=self.diagram,
            wrapper=self.wrapper,
            collision_model=self.collision_model,
            vision_model = self.vision_model,
            viz_model_instance=self.viz_model_instance,
            plant_name=self.plant_name,
            control_points=self.control_points,
            position_lower_limits=self.position_lower_limits,
            position_upper_limits=self.position_upper_limits,
            # bspline = self.bspline
        )
        for k, v in self.__dict__.items():
            if not hasattr(ob, k):
                setattr(ob, k, v)
        return ob

    @property
    def bspline(self) -> BSpline:
        return self._bspline

    @bspline.setter
    def bspline(self, value):
        self._bspline = value
        self.bspline_velocity = value.create_derivative_spline()
        self.bspline_acceleration = self.bspline_velocity.create_derivative_spline()

    def get_random_configuration(self, num_samples: int = 1):
        return np.random.default_rng().uniform(
            self.plant.GetPositionLowerLimits(),
            self.plant.GetPositionUpperLimits(),
            (num_samples, self.plant.num_positions()),
        )


@dataclass
class OptimizationData:
    parameters: T.Dict[str, np.ndarray] = field(default_factory=dict)
    parameter_shapes: T.Dict[str, np.ndarray] = field(default_factory=dict)
    variable_shapes: T.Dict[str, np.ndarray] = field(default_factory=dict)
    initial_guess: T.Dict[str, np.ndarray] = field(default_factory=dict)
    result: T.Dict[str, np.ndarray] = field(default_factory=dict)
    variable_slices: T.Dict[str, slice] = field(default_factory=dict)
    parameter_slices: T.Dict[str, slice] = field(default_factory=dict)
    constraint_slices: T.Dict[str, slice] = field(default_factory=dict)
    constraint_subnames: T.Dict[str, list[str]] = field(default_factory=dict)
    x0: np.ndarray = None
    p: np.ndarray = None
    lbx: np.ndarray = None
    ubx: np.ndarray = None
    lbg: np.ndarray = None
    ubg: np.ndarray = None
    lam_x0: np.ndarray = None
    lam_g0: np.ndarray = None
    x: np.ndarray = None
    f: np.ndarray = None
    g: np.ndarray = None
    lam_x: np.ndarray = None
    lam_g: np.ndarray = None
    lam_p: np.ndarray = None
    opti: ca.Opti = None
    solver: ca.Function = None
    lbg_ubg_function: ca.Function = None
    lbx_ubx_function: ca.Function = None
    last_solve_time = -1.0
    tol = 1e-4
    map_size = 1

    @staticmethod
    def create(
        opti: ca.Opti,
        variables: list[tuple[str, ca.MX]],
        parameters: list[tuple[str, ca.MX]],
        constraint_slices: dict[str, slice],
        constraint_subnames: dict[str, list[str]],
        map_size=1,
    ):
        # TODO warm start lambda_g
        optimization_data = OptimizationData()
        optimization_data.map_size = map_size
        baked_copy = opti.advanced.baked_copy()
        optimization_data.opti = baked_copy

        # inputs ['x0', 'p', 'lbx', 'ubx', 'lbg', 'ubg', 'lam_x0', 'lam_g0']

        optimization_data.x0 = np.zeros((baked_copy.x.shape[0], map_size), order="F")
        optimization_data.p = np.zeros((baked_copy.p.shape[0], map_size), order="F")
        optimization_data.lbx = np.zeros((baked_copy.x.shape[0], map_size), order="F")
        optimization_data.ubx = np.zeros((baked_copy.x.shape[0], map_size), order="F")
        optimization_data.lbg = np.zeros((baked_copy.g.shape[0], map_size), order="F")
        optimization_data.ubg = np.zeros((baked_copy.g.shape[0], map_size), order="F")
        optimization_data.lam_x0 = np.zeros(
            (baked_copy.x.shape[0], map_size), order="F"
        )
        optimization_data.lam_g0 = np.zeros(
            (baked_copy.g.shape[0], map_size), order="F"
        )

        # outputs ['x', 'f', 'g', 'lam_x', 'lam_g', 'lam_p']
        optimization_data.x = np.zeros((baked_copy.x.shape[0], map_size), order="F")
        optimization_data.f = np.zeros((1, map_size), order="F")
        optimization_data.g = np.zeros((baked_copy.g.shape[0], map_size), order="F")
        optimization_data.lam_x = np.zeros((baked_copy.x.shape[0], map_size), order="F")
        optimization_data.lam_g = np.zeros((baked_copy.g.shape[0], map_size), order="F")
        optimization_data.lam_p = np.zeros((baked_copy.p.shape[0], map_size), order="F")
        for var in variables:
            var_meta = baked_copy.get_meta(var[1])
            optimization_data.variable_slices[var[0]] = slice(
                var_meta.start, var_meta.stop
            )
            optimization_data.variable_shapes[var[0]] = var[1].shape

        start = 0
        for par in parameters:
            # optimization_data.parameters[par[0]] = None
            par_meta = baked_copy.get_meta(par[1])
            stop = start + par_meta.n * par_meta.m
            optimization_data.parameter_slices[par[0]] = slice(start, stop)
            optimization_data.parameter_shapes[par[0]] = par[1].shape
            start = stop
        optimization_data.constraint_slices = constraint_slices
        optimization_data.constraint_subnames = constraint_subnames
        # for name, constraint in named_constraints:
        #     if name in contraints_subnames:
        #         subnames = contraints_subnames[name]
        #     else:
        #         subnames = [f'c1_{i}' for i in range(constraint.shape[0])]
        #     # meta.n,meta.start,meta.stop
        #     meta = baked_copy.get_meta_con(constraint)
        #     n,start,stop = meta.n,meta.start,meta.stop
        #     optimization_data.constraint_slices[name] = slice(start, stop)
        #     optimization_data.constraint_subnames[name] = subnames


        optimization_data.setup_dictionaries()
        return optimization_data

    def setup_dictionaries(self):
        for name, s in self.variable_slices.items():
            if self.map_size > 1:
                self.result[name] = [None] * self.map_size
                self.initial_guess[name] = [None] * self.map_size
                for i in range(self.map_size):
                    self.initial_guess[name][i] = (
                        self.x0[s, i]
                        .reshape(
                            self.variable_shapes[name][1], self.variable_shapes[name][0]
                        )
                        .T
                    )
                    self.result[name][i] = (
                        self.x[s, i]
                        .reshape(
                            self.variable_shapes[name][1], self.variable_shapes[name][0]
                        )
                        .T
                    )
            else:
                self.initial_guess[name] = (
                    self.x0[s]
                    .reshape(
                        self.variable_shapes[name][1], self.variable_shapes[name][0]
                    )
                    .T
                )
                self.result[name] = (
                    self.x[s]
                    .reshape(
                        self.initial_guess[name].shape[1],
                        self.initial_guess[name].shape[0],
                    )
                    .T
                )
        for name, s in self.parameter_slices.items():
            if self.map_size > 1:
                self.parameters[name] = [None] * self.map_size
                for i in range(self.map_size):
                    self.parameters[name][i] = (
                        self.p[s, i]
                        .reshape(
                            self.parameter_shapes[name][1],
                            self.parameter_shapes[name][0],
                        )
                        .T
                    )
            else:
                self.parameters[name] = (
                    self.p[s]
                    .reshape(
                        self.parameter_shapes[name][1], self.parameter_shapes[name][0]
                    )
                    .T
                )

    def set_parameter(self, name, value, index=None):
        s = self.parameter_slices[name]
        if self.map_size > 1:
            if index is None:
                for i in range(self.map_size):
                    self.p[s, i] = value.T.reshape(*self.p[s, i].shape)
            else:
                self.p[s, index] = value.T.reshape(*self.p[s, index].shape)
            return

        self.p[s] = value.T.reshape(-1, 1)

    def set_initial_guess(self, name, value, index=None):
        s = self.variable_slices[name]
        if self.map_size > 1:
            # self.x0[s, index] = value.T.reshape(*self.x0[s, index].shape)
            if index is None:
                for i in range(self.map_size):
                    self.x0[s, i] = value.T.reshape(*self.x0[s, i].shape)
            else:
                self.x0[s, index] = value.T.reshape(*self.x0[s, index].shape)
            return

        self.x0[s] = value.T.reshape(-1, 1)

    def set_variable_bound(self, name, lb, ub, index=None):
        s = self.variable_slices[name]
        if self.map_size > 1:
            if index is None:
                for i in range(self.map_size):
                    self.lbx[s, i] = lb.T.reshape(*self.lbx[s, i].shape)
                    self.ubx[s, i] = ub.T.reshape(*self.ubx[s, i].shape)
            else:
                self.lbx[s, index] = lb.T.reshape(*self.lbx[s, index].shape)
                self.ubx[s, index] = ub.T.reshape(*self.ubx[s, index].shape)
            return
        self.lbx[s] = lb.T.reshape(-1, 1)
        self.ubx[s] = ub.T.reshape(-1, 1)

    def solve(self, warm=False, recompute_bounds=True):
        t0 = time.perf_counter()
        if recompute_bounds:
            self.bounds_eval()
            if self.lbx_ubx_function:
                self.bounds_eval_var()
        self.solver_eval()
        if warm:
            self.lam_x0[:] = self.lam_x
            self.lam_g0[:] = self.lam_g
            self.x0[:] = self.x
        self.last_solve_time = time.perf_counter() - t0

    def within_bounds(self):
        if self.map_size > 1:
            t = []
            for i in range(self.map_size):
                t.append(
                    np.all(self.lbx[:, i] - self.tol <= self.x[:, i])
                    and np.all(self.x[:, i] <= self.ubx[:, i] + self.tol)
                    and np.all(self.lbg[:, i] - self.tol <= self.g[:, i])
                    and np.all(self.g[:, i] <= self.ubg[:, i] + self.tol)
                )
            return t

        return (
            np.all(self.lbx - self.tol <= self.x)
            and np.all(self.x <= self.ubx + self.tol)
            and np.all(self.lbg - self.tol <= self.g)
            and np.all(self.g <= self.ubg + self.tol)
        )
    def get_constraint(self, name):
        if self.map_size > 1:
            t = []
            for i in range(self.map_size):
                t.append(self.g[self.constraint_slices[name], i])
            return t
        return self.g[self.constraint_slices[name]]
    def get_lbg_violation_by_name(self, name):
        if self.map_size > 1:
            t = []
            for i in range(self.map_size):
                t.append(np.clip(self.lbg[self.constraint_slices[name], i] - self.g[self.constraint_slices[name], i], 0.0, np.inf))
            return t
        return np.clip(self.lbg[self.constraint_slices[name]] - self.g[self.constraint_slices[name]], 0.0, np.inf)
    def get_ubg_violation_by_name(self, name):
        if self.map_size > 1:
            t = []
            for i in range(self.map_size):
                t.append(np.clip(self.g[self.constraint_slices[name], i] - self.ubg[self.constraint_slices[name], i], 0.0, np.inf))
            return t
        return np.clip(self.g[self.constraint_slices[name]] - self.ubg[self.constraint_slices[name]], 0.0, np.inf)
    def print_lbg_violation(self):
        for name, s in self.constraint_slices.items():
            print(f"{name} lb violation:")
            # g_ = self.get_constraint(name)
            lbg_violation = self.get_lbg_violation_by_name(name)
            for i,subname in enumerate(self.constraint_subnames[name]):
                print(f"{subname}: {lbg_violation[i]}")

    def print_ubg_violation(self):
        for name, s in self.constraint_slices.items():
            print(f"{name} ub violation:")
            ubg_violation = self.get_ubg_violation_by_name(name)
            for i,subname in enumerate(self.constraint_subnames[name]):
                print(f"{subname}: {ubg_violation[i]}")
    def print_g_violation(self, name=None):
        if name is None:
            for name, s in self.constraint_slices.items():
                print(f"{name} g violation:")
                
                lbg_violation = self.get_lbg_violation_by_name(name)
                ubg_violation = self.get_ubg_violation_by_name(name)
                for i,subname in enumerate(self.constraint_subnames[name]):
                    print(f"{subname}: {lbg_violation[i]} {ubg_violation[i]}")
        else:
            lbg_violation = self.get_lbg_violation_by_name(name)
            ubg_violation = self.get_ubg_violation_by_name(name)
            for i,subname in enumerate(self.constraint_subnames[name]):
                print(f"{subname}: {lbg_violation[i]} {ubg_violation[i]}")
    def print_violated_g(self):
        for name, s in self.constraint_slices.items():
            lbg_violation = self.get_lbg_violation_by_name(name)
            ubg_violation = self.get_ubg_violation_by_name(name)
            g = self.get_constraint(name)
            lbg = self.lbg[self.constraint_slices[name]]
            ubg = self.ubg[self.constraint_slices[name]]
            print(f"{name}:")
            for i,subname in enumerate(self.constraint_subnames[name]):
                if lbg_violation[i] > self.tol or ubg_violation[i] > self.tol:
                    print(f"{subname}: {lbg[i]} {g[i]} {ubg[i]}")
    def print_g(self, name = None):
        if name is None:
            for name, s in self.constraint_slices.items():
                print(f"{name}:")
                g_ = self.get_constraint(name)
                for i,subname in enumerate(self.constraint_subnames[name]):
                    print(f"{subname}: {g_[i]}")
        else:
            print(f"{name}:")
            g_ = self.get_constraint(name)
            for i,subname in enumerate(self.constraint_subnames[name]):
                print(f"{subname}: {g_[i]}")
    def lbx_violation(self, name=None):
        if name is None:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.lbx[:, i] - self.x[:, i], 0.0, np.inf))
                return t
            return np.clip(self.lbx - self.x, 0.0, np.inf)
        else:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.lbx[self.variable_slices[name], i] - self.x[self.variable_slices[name], i], 0.0, np.inf))
                return t
            return np.clip(self.lbx[self.variable_slices[name]] - self.x[self.variable_slices[name]], 0.0, np.inf)

    def ubx_violation(self, name=None):
        if name is None:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.x[:, i] - self.ubx[:, i], 0.0, np.inf))
                return t
            return np.clip(self.x - self.ubx, 0.0, np.inf)
        else:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.x[self.variable_slices[name], i] - self.ubx[self.variable_slices[name], i], 0.0, np.inf))
                return t
            return np.clip(self.x[self.variable_slices[name]] - self.ubx[self.variable_slices[name]], 0.0, np.inf)

    def lbg_violation(self, name=None):
        if name is None:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.lbg[:, i] - self.g[:, i], 0.0, np.inf))
                return t
            return np.clip(self.lbg - self.g, 0.0, np.inf)
        else:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.lbg[self.constraint_slices[name], i] - self.g[self.constraint_slices[name], i], 0.0, np.inf))
                return t
            return np.clip(self.lbg[self.constraint_slices[name]] - self.g[self.constraint_slices[name]], 0.0, np.inf)

    def ubg_violation(self, name=None):
        if name is None:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.g[:, i] - self.ubg[:, i], 0.0, np.inf))
                return t
            return np.clip(self.g - self.ubg, 0.0, np.inf)
        else:
            if self.map_size > 1:
                t = []
                for i in range(self.map_size):
                    t.append(np.clip(self.g[self.constraint_slices[name], i] - self.ubg[self.constraint_slices[name], i], 0.0, np.inf))
                return t
            return np.clip(self.g[self.constraint_slices[name]] - self.ubg[self.constraint_slices[name]], 0.0, np.inf)

    def bufferize_solver_and_bounds_g(
        self, solver, lbg_ubg_function, lbx_ubx_function=None
    ):
        [buf, f_eval_solver] = solver.buffer()
        self.solver = solver
        self.lbg_ubg_function = lbg_ubg_function
        self.lbx_ubx_function = lbx_ubx_function
        self.buffer_solver = buf
        # inputs ['x0', 'p', 'lbx', 'ubx', 'lbg', 'ubg', 'lam_x0', 'lam_g0']
        buf.set_arg(0, memoryview(self.x0))
        buf.set_arg(1, memoryview(self.p))
        buf.set_arg(2, memoryview(self.lbx))
        buf.set_arg(3, memoryview(self.ubx))
        buf.set_arg(4, memoryview(self.lbg))
        buf.set_arg(5, memoryview(self.ubg))
        buf.set_arg(6, memoryview(self.lam_x0))
        buf.set_arg(7, memoryview(self.lam_g0))
        # outputs ['x', 'f', 'g', 'lam_x', 'lam_g', 'lam_p']
        buf.set_res(0, memoryview(self.x))
        buf.set_res(1, memoryview(self.f))
        buf.set_res(2, memoryview(self.g))
        buf.set_res(3, memoryview(self.lam_x))
        buf.set_res(4, memoryview(self.lam_g))
        buf.set_res(5, memoryview(self.lam_p))
        [buf_bounds, f_eval_bounds] = lbg_ubg_function.buffer()
        self.buffer_bounds = buf_bounds
        buf_bounds.set_arg(0, memoryview(self.p))
        buf_bounds.set_res(0, memoryview(self.lbg))
        buf_bounds.set_res(1, memoryview(self.ubg))

        if lbx_ubx_function:
            [buf_bounds_var, f_eval_bounds_var] = lbx_ubx_function.buffer()
            self.buffer_bounds_var = buf_bounds_var
            buf_bounds_var.set_arg(0, memoryview(self.p))
            buf_bounds_var.set_res(0, memoryview(self.lbx))
            buf_bounds_var.set_res(1, memoryview(self.ubx))
            self.bounds_eval_var = f_eval_bounds_var
        self.bounds_eval = f_eval_bounds
        self.solver_eval = f_eval_solver

    def get_x0_p(self):
        return self.x0.copy(), self.p.copy()

    def get_x_p(self):
        return self.x.copy(), self.p.copy()

    def get_x(self):
        return self.x.copy()

    def get_p(self):
        return self.p.copy()

    def get_lam_x_g_p(self):
        return self.lam_x.copy(), self.lam_g.copy(), self.lam_p.copy()

    def get_x_as_dict_copy(self, x=None):
        if x is None:
            return copy.deepcopy(self.result)
        result = {}
        if isinstance(x, ca.DM):
            x = x.full().reshape(-1)
        else:
            x = x.copy()
        for name, s in self.variable_slices.items():

            if self.map_size > 1:
                result[name] = np.empty([self.map_size, *self.variable_shapes[name]])
                for i in range(self.map_size):
                    result[name][i] = (
                        x[s, i]
                        .reshape(
                            self.variable_shapes[name][1], self.variable_shapes[name][0]
                        )
                        .T
                    )
            else:
                result[name] = (
                    x[s]
                    .reshape(
                        self.variable_shapes[name][1], self.variable_shapes[name][0]
                    )
                    .T
                )
        return result

    def get_p_as_dict_copy(self, p=None):
        if p is None:
            return copy.deepcopy(self.parameters)

        result = {}
        if isinstance(p, ca.DM):
            p = p.full().reshape(-1)
        else:
            p = p.copy()
        for name, s in self.parameter_slices.items():
            if self.map_size > 1:
                result[name] = np.empty([self.map_size, *self.parameter_shapes[name]])
                for i in range(self.map_size):
                    result[name][i] = (
                        p[s, i]
                        .reshape(
                            self.parameter_shapes[name][1],
                            self.parameter_shapes[name][0],
                        )
                        .T
                    )
            else:
                result[name] = (
                    p[s]
                    .reshape(
                        self.parameter_shapes[name][1], self.parameter_shapes[name][0]
                    )
                    .T
                )
        return result

    def __copy__(self):
        ob = OptimizationData()
        ob.opti = self.opti
        ob.parameters = copy.deepcopy(self.parameters)
        ob.parameter_shapes = copy.deepcopy(self.parameter_shapes)
        ob.variable_shapes = copy.deepcopy(self.variable_shapes)
        ob.initial_guess = copy.deepcopy(self.initial_guess)
        ob.result = copy.deepcopy(self.result)
        ob.variable_slices = copy.deepcopy(self.variable_slices)
        ob.parameter_slices = copy.deepcopy(self.parameter_slices)
        ob.x0 = self.x0.copy()
        ob.p = self.p.copy()
        ob.lbx = self.lbx.copy()
        ob.ubx = self.ubx.copy()
        ob.lbg = self.lbg.copy()
        ob.ubg = self.ubg.copy()
        ob.lam_x0 = self.lam_x0.copy()
        ob.lam_g0 = self.lam_g0.copy()
        ob.x = self.x.copy()
        ob.f = self.f.copy()
        ob.g = self.g.copy()
        ob.lam_x = self.lam_x.copy()
        ob.lam_g = self.lam_g.copy()
        ob.lam_p = self.lam_p.copy()
        ob.tol = self.tol
        ob.map_size = self.map_size
        # ob.solver = self.solver
        if self.solver:
            ob.solver = self.solver
            ob.lbg_ubg_function = self.lbg_ubg_function
            ob.lbx_ubx_function = self.lbx_ubx_function
            ob.bufferize_solver_and_bounds_g(
                self.solver, self.lbg_ubg_function, self.lbx_ubx_function
            )
        ob.setup_dictionaries()
        return ob


@dataclass
class OptimizationInfo:
    robots: list[RobotOptimisationInfo] = field(default_factory=list)
    named_variables: list[tuple[str, ca.MX]] = field(default_factory=list)
    named_parameters: list[tuple[str, ca.MX]] = field(default_factory=list)
    named_constraints: list[tuple[str, ca.MX]] = field(default_factory=list)
    contraints_subnames: dict[str, list[str]] = field(default_factory=dict)
    carried_object: RobotOptimisationInfo = None
    robot_1_EE_frame_name: str = "EE_frame"
    robot_2_EE_frame_name: str = "EE_frame"
    carried_object_EE_1_frame_name: str = "carried_object_EE_1_frame"
    carried_object_EE_2_frame_name: str = "carried_object_EE_2_frame"
    carried_object_frame_name: str = "carried_object"
    opti: ca.Opti = field(default_factory=ca.Opti)
    num_control_points: int = None
    order: int = None
    _carried_object_EE_1_frame: pydrake.all.Frame = None
    _carried_object_EE_2_frame: pydrake.all.Frame = None
    _carried_object_frame: pydrake.all.Frame = None
    _robot_1_EE_frame: pydrake.all.Frame = None
    _robot_2_EE_frame: pydrake.all.Frame = None
    optimization_data: dict[T.Union[str, int], OptimizationData] = field(
        default_factory=dict
    )

    @property
    def carried_object_EE_1_frame(self):
        return self.carried_object.plant.GetFrameByName(
            self.carried_object_EE_1_frame_name
        )

    @property
    def carried_object_EE_2_frame(self):
        return self.carried_object.plant.GetFrameByName(
            self.carried_object_EE_2_frame_name
        )

    @property
    def carried_object_frame(self):
        return self.carried_object.plant.GetFrameByName(self.carried_object_frame_name)

    @property
    def robot_1_EE_frame(self):
        return self.robots[0].plant.GetFrameByName(self.robot_1_EE_frame_name)

    @property
    def robot_2_EE_frame(self):
        return self.robots[1].plant.GetFrameByName(self.robot_2_EE_frame_name)

    def make_optimization_data(self, map_size=1):
        optimization_data = OptimizationData.create(
            self.opti, self.named_variables, self.named_parameters, self.constraint_slices, self.constraint_subnames,map_size
        )
        return optimization_data

    # def __init__()

    def variable(self, name, *args, **kwargs):
        opti_var = self.opti.variable(*args, **kwargs)
        self.named_variables.append((name, opti_var))
        return opti_var

    def parameter(self, name, *args, **kwargs):
        opti_var = self.opti.parameter(*args, **kwargs)
        self.named_parameters.append((name, opti_var))
        return opti_var

    def subject_to(self, name, constraint, subnames = None):
        self.opti.subject_to(constraint)
        self.named_constraints.append((name, constraint))
        if subnames is not None:
            # assert isinstance(subnames, (list, tuple))
            assert len(subnames) == constraint.shape[0]
            self.contraints_subnames[name] = subnames

    def setup_optimization(
        self, num_control_points, order, diff_co_options, gaze_options
    ):
        self.num_control_points = num_control_points
        self.order = order
        self.diff_co_options = diff_co_options
        self.gaze_options:VisionOptions = gaze_options
        self.robots_by_name = {"robot_1": self.robots[0], "robot_2": self.robots[1]}
        self.robots[0].control_points = self.variable(
            "robot_1_control_points", 7, self.num_control_points
        )
        self.robots[1].control_points = self.variable(
            "robot_2_control_points", 7, self.num_control_points
        )
        self.carried_object.control_points = self.variable(
            "carried_object_control_points",
            self.carried_object.plant.num_positions(),
            self.num_control_points,
        )
        self.robots[0].bspline = BSpline(
            ca.vertcat(
                self.robots[0].control_points, ca.DM.zeros(2, self.num_control_points)
            ),
            self.order,
        )
        self.robots[1].bspline = BSpline(
            ca.vertcat(
                self.robots[1].control_points, ca.DM.zeros(2, self.num_control_points)
            ),
            self.order,
        )
        self.carried_object.bspline = BSpline(
            self.carried_object.control_points, self.order
        )
        self.duration = self.variable("duration", 1, 1)
        self.t_min = self.parameter("t_min", 1, 1)
        self.t_max = self.parameter("t_max", 1, 1)
        self.duration_cost = self.parameter("duration_cost", 1, 1)

        # self.robot_1_initial_configuration = self.parameter(
        #     "robot_1_initial_configuration", 7, 1
        # )
        # self.robot_2_initial_configuration = self.parameter(
        #     "robot_2_initial_configuration", 7, 1
        # )
        # self.object_initial_configuration = self.parameter(
        #     "object_initial_configuration", 7, 1
        # )

        # self.robot_1_initial_configuration_lbg = self.parameter(
        #     "robot_1_initial_configuration_lbg", 7, 1
        # )
        # self.robot_2_initial_configuration_lbg = self.parameter(
        #     "robot_2_initial_configuration_lbg", 7, 1
        # )
        # self.object_initial_configuration_lbg = self.parameter(
        #     "object_initial_configuration_lbg", 7, 1
        # )
        # self.robot_1_initial_configuration_ubg = self.parameter(
        #     "robot_1_initial_configuration_ubg", 7, 1
        # )
        # self.robot_2_initial_configuration_ubg = self.parameter(
        #     "robot_2_initial_configuration_ubg", 7, 1
        # )
        # self.object_initial_configuration_ubg = self.parameter(
        #     "object_initial_configuration_ubg", 7, 1
        # )

        self.robot_1_initial_velocity = self.parameter("robot_1_initial_velocity", 7, 1)
        self.robot_2_initial_velocity = self.parameter("robot_2_initial_velocity", 7, 1)
        self.object_initial_velocity = self.parameter("object_initial_velocity", 7, 1)
        self.robot_1_initial_velocity_lbg = self.parameter(
            "robot_1_initial_velocity_lbg", 7, 1
        )
        self.robot_2_initial_velocity_lbg = self.parameter(
            "robot_2_initial_velocity_lbg", 7, 1
        )
        self.object_initial_velocity_lbg = self.parameter(
            "object_initial_velocity_lbg", 7, 1
        )
        self.robot_1_initial_velocity_ubg = self.parameter(
            "robot_1_initial_velocity_ubg", 7, 1
        )
        self.robot_2_initial_velocity_ubg = self.parameter(
            "robot_2_initial_velocity_ubg", 7, 1
        )
        self.object_initial_velocity_ubg = self.parameter(
            "object_initial_velocity_ubg", 7, 1
        )
        self.robot_1_initial_position = self.parameter("robot_1_initial_position", 7, 1)
        self.robot_2_initial_position = self.parameter("robot_2_initial_position", 7, 1)
        self.replan_connection_cost = self.parameter("replan_connection_cost", 1, 1)
        # self.robot_1_initial_acceleration = self.parameter(
        #     "robot_1_initial_acceleration", 7, 1
        # )
        # self.robot_2_initial_acceleration = self.parameter(
        #     "robot_2_initial_acceleration", 7, 1
        # )
        # self.object_initial_acceleration = self.parameter(
        #     "object_initial_acceleration", 7, 1
        # )
        # self.robot_1_initial_acceleration_lbg = self.parameter(
        #     "robot_1_initial_acceleration_lbg", 7, 1
        # )
        # self.robot_2_initial_acceleration_lbg = self.parameter(
        #     "robot_2_initial_acceleration_lbg", 7, 1
        # )
        # self.object_initial_acceleration_lbg = self.parameter(
        #     "object_initial_acceleration_lbg", 7, 1
        # )
        # self.robot_1_initial_acceleration_ubg = self.parameter(
        #     "robot_1_initial_acceleration_ubg", 7, 1
        # )
        # self.robot_2_initial_acceleration_ubg = self.parameter(
        #     "robot_2_initial_acceleration_ubg", 7, 1
        # )
        # self.object_initial_acceleration_ubg = self.parameter(
        #     "object_initial_acceleration_ubg", 7, 1
        # )

        self.robot_1_terminal_velocity = self.parameter(
            "robot_1_terminal_velocity", 7, 1
        )
        self.robot_2_terminal_velocity = self.parameter(
            "robot_2_terminal_velocity", 7, 1
        )
        self.robot_1_terminal_velocity_lbg = self.parameter(
            "robot_1_terminal_velocity_lbg", 7, 1
        )
        self.robot_2_terminal_velocity_lbg = self.parameter(
            "robot_2_terminal_velocity_lbg", 7, 1
        )
        self.robot_1_terminal_velocity_ubg = self.parameter(
            "robot_1_terminal_velocity_ubg", 7, 1
        )
        self.robot_2_terminal_velocity_ubg = self.parameter(
            "robot_2_terminal_velocity_ubg", 7, 1
        )

        self.object_start_position: ca.MX = self.parameter(
            "object_start_position", 3, 1
        )
        self.object_end_position: ca.MX = self.parameter("object_end_position", 3, 1)
        self.object_start_rotation_matrix: ca.MX = self.parameter(
            "object_start_rotation_matrix", 3, 3
        )
        self.object_end_rotation_matrix: ca.MX = self.parameter(
            "object_end_rotation_matrix", 3, 3
        )
        self.grasping_rotation_z_EE_1: ca.MX = self.parameter(
            "grasping_rotation_z_EE_1", 1, 1
        )
        self.grasping_rotation_z_EE_2: ca.MX = self.parameter(
            "grasping_rotation_z_EE_2", 1, 1
        )

        self.grasping_x_EE_1: ca.MX = self.variable("grasping_x_EE_1", 1, 1)
        self.grasping_x_EE_2: ca.MX = self.variable("grasping_x_EE_2", 1, 1)
        self.grasping_point_EE_1: ca.MX = ca.veccat(self.grasping_x_EE_1, 0, 0)
        self.grasping_point_EE_2: ca.MX = ca.veccat(self.grasping_x_EE_2, 0, 0)
        self.grasping_rotation_matrix_EE_1: ca.MX = ca.vertcat(
            ca.horzcat(
                ca.cos(self.grasping_rotation_z_EE_1),
                -ca.sin(self.grasping_rotation_z_EE_1),
                0,
            ),
            ca.horzcat(
                ca.sin(self.grasping_rotation_z_EE_1),
                ca.cos(self.grasping_rotation_z_EE_1),
                0,
            ),
            ca.horzcat(0, 0, 1),
        )
        self.grasping_rotation_matrix_EE_2: ca.MX = ca.vertcat(
            ca.horzcat(
                ca.cos(self.grasping_rotation_z_EE_2),
                -ca.sin(self.grasping_rotation_z_EE_2),
                0,
            ),
            ca.horzcat(
                ca.sin(self.grasping_rotation_z_EE_2),
                ca.cos(self.grasping_rotation_z_EE_2),
                0,
            ),
            ca.horzcat(0, 0, 1),
        )

        self.acceleration_cost_weight: ca.MX = self.parameter(
            "acceleration_cost_weight", 1, 1
        )
        self.robot_1_lower_limits = self.robots[0].position_lower_limits = (
            self.parameter("robot_1_lower_limits", 7, 1)
        )
        self.robot_1_upper_limits = self.robots[0].position_upper_limits = (
            self.parameter("robot_1_upper_limits", 7, 1)
        )
        self.robot_2_lower_limits = self.robots[1].position_lower_limits = (
            self.parameter("robot_2_lower_limits", 7, 1)
        )
        self.robot_2_upper_limits = self.robots[1].position_upper_limits = (
            self.parameter("robot_2_upper_limits", 7, 1)
        )
        self.object_lower_limits = self.carried_object.position_lower_limits = (
            self.parameter(
                "object_lower_limits", self.carried_object.plant.num_positions(), 1
            )
        )
        self.object_upper_limits = self.carried_object.position_upper_limits = (
            self.parameter(
                "object_upper_limits", self.carried_object.plant.num_positions(), 1
            )
        )

        self.robot_1_lower_limits_velocity = self.robots[
            0
        ].position_lower_limits_velocity = self.parameter(
            "robot_1_lower_limits_velocity", 7, 1
        )
        self.robot_1_upper_limits_velocity = self.robots[
            0
        ].position_upper_limits_velocity = self.parameter(
            "robot_1_upper_limits_velocity", 7, 1
        )
        self.robot_2_lower_limits_velocity = self.robots[
            1
        ].position_lower_limits_velocity = self.parameter(
            "robot_2_lower_limits_velocity", 7, 1
        )
        self.robot_2_upper_limits_velocity = self.robots[
            1
        ].position_upper_limits_velocity = self.parameter(
            "robot_2_upper_limits_velocity", 7, 1
        )
        self.object_lower_limits_velocity = (
            self.carried_object.position_lower_limits_velocity
        ) = self.parameter(
            "object_lower_limits_velocity", self.carried_object.plant.num_positions(), 1
        )
        self.object_upper_limits_velocity = (
            self.carried_object.position_upper_limits_velocity
        ) = self.parameter(
            "object_upper_limits_velocity", self.carried_object.plant.num_positions(), 1
        )

        self.grasping_EE_1_bounds = self.parameter("grasping_EE_1_bounds", 2, 1)
        self.grasping_EE_2_bounds = self.parameter("grasping_EE_2_bounds", 2, 1)

        self.start_position_lower_bounds = self.parameter(
            "start_position_lower_bounds", 3, 1
        )
        self.end_position_lower_bounds = self.parameter(
            "end_position_lower_bounds", 3, 1
        )
        self.start_position_upper_bounds = self.parameter(
            "start_position_upper_bounds", 3, 1
        )
        self.end_position_upper_bounds = self.parameter(
            "end_position_upper_bounds", 3, 1
        )
        self.start_angle_bounds = self.parameter("start_angle_bounds", 1, 1)
        self.end_angle_bounds = self.parameter("end_angle_bounds", 1, 1)

        self.svm_weights = {}
        self.svm_support_vectors = {}
        self.svm_polynomial_weights = {}
        self.svm_lbg = {}
        self.svm_ubg = {}
        self.svm_slack = {}
        self.slack_cost_weight = self.parameter("slack_cost_weight", 1, 1)
        if self.diff_co_options:
            for robot_options in self.diff_co_options["robots"]:
                robot_options: DiffCoRobotOptions

                self.svm_weights[robot_options["name"]] = {}
                self.svm_support_vectors[robot_options["name"]] = {}
                self.svm_polynomial_weights[robot_options["name"]] = {}
                self.svm_lbg[robot_options["name"]] = {}
                self.svm_ubg[robot_options["name"]] = {}
                self.svm_slack[robot_options["name"]] = {}
                robot: RobotOptimisationInfo = self.robots_by_name[
                    robot_options["name"]
                ]
                for group_options in robot_options["groups"].values():
                    group_options: DiffCoGroupOptions
                    group_name = group_options["name"]
                    fk_function = (
                        robot.collision_model.forward_kinematics_groups_casadi[
                            group_name
                        ]
                    )
                    self.svm_weights[robot_options["name"]][group_name] = (
                        self.parameter(
                            f"{robot_options['name']}_{group_name}_svm_weights",
                            group_options["num_obstacle_categories"],
                            group_options["num_support_points_per_block"]
                            * group_options["inner_map_size"]
                            * group_options["outer_map_size"],
                        )
                    )
                    self.svm_support_vectors[robot_options["name"]][group_name] = (
                        self.parameter(
                            f"{robot_options['name']}_{group_name}_svm_support_vectors",
                            fk_function.numel_out(),
                            group_options["num_support_points_per_block"]
                            * group_options["inner_map_size"]
                            * group_options["outer_map_size"],
                        )
                    )
                    self.svm_polynomial_weights[robot_options["name"]][
                        group_name  # type: ignore
                    ] = self.parameter(
                        f"{robot_options['name']}_{group_name}_svm_polynomial_weights",
                        fk_function.numel_out() + 1,
                        group_options["num_obstacle_categories"],
                    )
                    self.svm_lbg[robot_options["name"]][group_name] = self.parameter(
                        f"{robot_options['name']}_{group_name}_svm_lbg",
                        group_options["num_obstacle_categories"],
                        1,
                    )
                    self.svm_ubg[robot_options["name"]][group_name] = self.parameter(
                        f"{robot_options['name']}_{group_name}_svm_ubg",
                        group_options["num_obstacle_categories"],
                        1,
                    )
                    self.svm_slack[robot_options["name"]][group_name] = self.variable(
                        f"{robot_options['name']}_{group_name}_svm_slack",
                        group_options["num_obstacle_categories"],
                        1,
                    )
            # self.score_lbg: ca.MX = self.parameter("score_lbg", 1, 1)
            # self.score_ubg: ca.MX = self.parameter("score_ubg", 1, 1)
        self.manipulability_weight = self.parameter("manipulability_weight", 1, 1)
        if self.gaze_options:
            self.vision_weights = {}
            self.vision_support_vectors = {}
            self.vision_polynomial_weights = {}
            self.vision_weights[robot_options["name"]] = {}
            self.vision_support_vectors[robot_options["name"]] = {}
            self.vision_polynomial_weights[robot_options["name"]] = {}
            
            self.vision_cost_weight: ca.MX = self.parameter("vision_cost_weight", 1, 1)
            for robot_options in self.gaze_options["robots"]:
                robot_options: DiffCoRobotOptions

                self.vision_weights[robot_options["name"]] = {}
                self.vision_support_vectors[robot_options["name"]] = {}
                self.vision_polynomial_weights[robot_options["name"]] = {}
                
                robot: RobotOptimisationInfo = self.robots_by_name[
                    robot_options["name"]
                ]
                for group_options in robot_options["groups"].values():
                    group_options: DiffCoGroupOptions
                    group_name = group_options["name"]
                    fk_function = (
                        robot.collision_model.forward_kinematics_groups_casadi[
                            group_name
                        ]
                    )
                    self.vision_weights[robot_options["name"]][group_name] = (
                        self.parameter(
                            f"{robot_options['name']}_{group_name}_vision_weights",
                            1,
                            group_options["num_support_points_per_block"]
                            * group_options["inner_map_size"]
                            * group_options["outer_map_size"],
                        )
                    )
                    self.vision_support_vectors[robot_options["name"]][group_name] = (
                        self.parameter(
                            f"{robot_options['name']}_{group_name}_vision_support_vectors",
                            fk_function.numel_out(),
                            group_options["num_support_points_per_block"]
                            * group_options["inner_map_size"]
                            * group_options["outer_map_size"],
                        )
                    )
                    self.vision_polynomial_weights[robot_options["name"]][
                        group_name  # type: ignore
                    ] = self.parameter(
                        f"{robot_options['name']}_{group_name}_vision_polynomial_weights",
                        fk_function.numel_out() + 1,
                        1,
                    )
                    
        self.decision_variables: ca.MX = ca.veccat(
            *[
                var
                for var in self.opti.advanced.symvar()
                if not self.opti.advanced.is_parametric(var)
            ]
        )
        self.parameters: ca.MX = ca.veccat(
            *[
                var
                for var in self.opti.advanced.symvar()
                if self.opti.advanced.is_parametric(var)
            ]
        )

    def make_lbx_ubx_function(self):
        lbx_out = -ca.MX.inf(self.decision_variables.shape[0], 1)
        ubx_out = ca.MX.inf(self.decision_variables.shape[0], 1)
        start = 0

        lower_limits = {
            "robot_1_control_points": ca.repmat(
                self.robots[0].position_lower_limits, 1, self.num_control_points
            ),
            "robot_2_control_points": ca.repmat(
                self.robots[1].position_lower_limits, 1, self.num_control_points
            ),
            "carried_object_control_points": ca.repmat(
                self.carried_object.position_lower_limits, 1, self.num_control_points
            ),
            "grasping_x_EE_1": self.grasping_EE_1_bounds[0],
            "grasping_x_EE_2": self.grasping_EE_2_bounds[0],
            "duration": self.t_min,
        }
        for robot_name in self.svm_weights:
            robot_slack = self.svm_slack[robot_name]
            for group_name in robot_slack:
                lower_limits |= {f"{robot_name}_{group_name}_svm_slack":0.}
        upper_limits = {
            "robot_1_control_points": ca.repmat(
                self.robots[0].position_upper_limits, 1, self.num_control_points
            ),
            "robot_2_control_points": ca.repmat(
                self.robots[1].position_upper_limits, 1, self.num_control_points
            ),
            "carried_object_control_points": ca.repmat(
                self.carried_object.position_upper_limits, 1, self.num_control_points
            ),
            "grasping_x_EE_1": self.grasping_EE_1_bounds[1],
            "grasping_x_EE_2": self.grasping_EE_2_bounds[1],
            "duration": self.t_max,
        }
        for robot_name in self.svm_weights:
            robot_slack = self.svm_slack[robot_name]
            for group_name in robot_slack:
                upper_limits |= {f"{robot_name}_{group_name}_svm_slack":1.}
        start = 0
        for par in self.named_variables:

            par_meta = self.opti.advanced.get_meta(par[1])
            stop = start + par_meta.n * par_meta.m
            if par[0] in lower_limits:
                lbx_out[start:stop] = ca.vec(lower_limits[par[0]])
            if par[0] in upper_limits:
                ubx_out[start:stop] = ca.vec(upper_limits[par[0]])
            start = stop
        self.lbx_ubx_function = ca.Function(
            "lbx_ubx_func",
            {"parameters": self.parameters} | {"lbx": lbx_out, "ubx": ubx_out},
            ["parameters"],
            ["lbx", "ubx"],
            {"always_inline": True},
        )
        return self.lbx_ubx_function

    def initial_configuration_velocity_constraint(self):
        g = []
        ubg = []
        lbg = []
        s = 1 / self.num_control_points * 2
        s = 0
        # g.append(
        #     self.robots[0].bspline.evaluate(s)[:7].reshape((7, 1))
        #     - self.robot_1_initial_configuration
        # )
        # g.append(
        #     self.robots[1].bspline.evaluate(s)[:7].reshape((7, 1))
        #     - self.robot_2_initial_configuration
        # )
        # g.append(
        #     self.carried_object.bspline.evaluate(s)[:7].reshape((7, 1))
        #     - self.object_initial_configuration
        # )
        # lbg.append(self.robot_1_initial_configuration_lbg)
        # lbg.append(self.robot_2_initial_configuration_lbg)
        # lbg.append(self.object_initial_configuration_lbg)
        # ubg.append(self.robot_1_initial_configuration_ubg)
        # ubg.append(self.robot_2_initial_configuration_ubg)
        # ubg.append(self.object_initial_configuration_ubg)

        g.append(
            self.robots[0].bspline_velocity.evaluate(s)[:7].reshape((7, 1))
            / self.duration
            - self.robot_1_initial_velocity
        )
        g.append(
            self.robots[1].bspline_velocity.evaluate(s)[:7].reshape((7, 1))
            / self.duration
            - self.robot_2_initial_velocity
        )
        g.append(
            self.carried_object.bspline_velocity.evaluate(s)[:7].reshape((7, 1))
            / self.duration
            - self.object_initial_velocity
        )
        lbg.append(self.robot_1_initial_velocity_lbg)
        lbg.append(self.robot_2_initial_velocity_lbg)
        lbg.append(self.object_initial_velocity_lbg)
        ubg.append(self.robot_1_initial_velocity_ubg)
        ubg.append(self.robot_2_initial_velocity_ubg)
        ubg.append(self.object_initial_velocity_ubg)

        g.append(
            self.robots[0].bspline_velocity.evaluate(1)[:7].reshape((7, 1))
            / self.duration
            - self.robot_1_terminal_velocity
        )
        g.append(
            self.robots[1].bspline_velocity.evaluate(1)[:7].reshape((7, 1))
            / self.duration
            - self.robot_2_terminal_velocity
        )
        lbg.append(self.robot_1_terminal_velocity_lbg)
        lbg.append(self.robot_2_terminal_velocity_lbg)
        ubg.append(self.robot_1_terminal_velocity_ubg)
        ubg.append(self.robot_2_terminal_velocity_ubg)

        # g.append(
        #     self.robots[0].bspline_acceleration.evaluate(s)[:7].reshape((7, 1))
        #     / self.duration**2
        #     - self.robot_1_initial_acceleration
        # )
        # g.append(
        #     self.robots[1].bspline_acceleration.evaluate(s)[:7].reshape((7, 1))
        #     / self.duration**2
        #     - self.robot_2_initial_acceleration
        # )
        # g.append(
        #     self.carried_object.bspline_acceleration.evaluate(s)[:7].reshape((7, 1))
        #     / self.duration**2
        #     - self.object_initial_acceleration
        # )
        # lbg.append(self.robot_1_initial_acceleration_lbg)
        # lbg.append(self.robot_2_initial_acceleration_lbg)
        # lbg.append(self.object_initial_acceleration_lbg)
        # ubg.append(self.robot_1_initial_acceleration_ubg)
        # ubg.append(self.robot_2_initial_acceleration_ubg)
        # ubg.append(self.object_initial_acceleration_ubg)

        g = ca.veccat(*g)
        lbg = ca.veccat(*lbg)
        ubg = ca.veccat(*ubg)
        constraint = self.opti.bounded(lbg, g, ubg)
        # self.opti.subject_to(constraint)
        # subnames = [
        #     *["robot_1_initial_configuration_"],
        #     "robot_2_initial_configuration",
        #     "object_initial_configuration",
        #     "robot_1_initial_velocity",
        #     "robot_2_initial_velocity",
        #     "object_initial_velocity",
        #     "robot_1_terminal_velocity",
        #     "robot_2_terminal_velocity",
        #     "robot_1_initial_acceleration",
        #     "robot_2_initial_acceleration",
        #     "object_initial_acceleration",
        # ]
        self.subject_to("initial_configuration_velocity_constraint", constraint, None)

        g_bounds_function = ca.Function(
            "g_initial_config",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {"out": g},
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        jacobian_bounds = ca.Function(
            "jacobian_initial_config",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {"out": ca.matrix_expand((ca.jacobian(g, self.decision_variables)))},
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        lam_g_bounds = ca.MX.sym("lam_g", g.numel(), 1)
        hessian_bounds = ca.Function(
            "hessian_initial_config",
            {
                "dec_variables": self.decision_variables,
                "parameters": self.parameters,
                "lam_g_in": lam_g_bounds,
            }
            | {
                "out": ca.matrix_expand(
                    ca.triu(
                        ca.hessian(ca.dot(g, lam_g_bounds), self.decision_variables)[0]
                    )
                )
            },
            ["dec_variables", "parameters", "lam_g_in"],
            ["out"],
            {"always_inline": True},
        )

        return {
            "constraint": constraint,
            "g": g_bounds_function,
            "jacobian": jacobian_bounds,
            "hessian": hessian_bounds,
        }

    def position_bounds_constraint(self):
        g_position = []
        lbg_position = []
        ubg_position = []
        for aaa in self.robots + [self.carried_object]:
            for i in range(0, self.num_control_points):
                control_point = aaa.control_points[:, i]
                # aaa.control_points
                lower_limits = aaa.position_lower_limits
                upper_limits = aaa.position_upper_limits
                g, lbg, ubg = KinematicOptimization.position_bound_constraint(
                    control_point, lower_limits, upper_limits
                )
                g_position.append(g)
                lbg_position.append(lbg)
                ubg_position.append(ubg)
        g_position.append(self.grasping_x_EE_1)
        # lbg_position.append(0.0)
        # ubg_position.append(0.1)
        lbg_position.append(self.grasping_EE_1_bounds[0])
        ubg_position.append(self.grasping_EE_1_bounds[1])
        g_position.append(self.grasping_x_EE_2)
        lbg_position.append(self.grasping_EE_2_bounds[0])
        ubg_position.append(self.grasping_EE_2_bounds[1])
        # lbg_position.append(-0.1)
        # ubg_position.append(0.0)
        g_position = ca_utils.veccat(g_position)
        lbg_position = ca_utils.veccat(lbg_position)
        ubg_position = ca_utils.veccat(ubg_position)
        g_bounds_function = ca.Function(
            "g_bounds",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {"out": g_position},
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        position_bound_constraint = self.opti.bounded(
            lbg_position, g_position, ubg_position
        )
        # self.opti.subject_to(position_bound_constraint)
        self.subject_to("position_bounds_constraint", position_bound_constraint)

        jacobian_bounds = ca.Function(
            "jacobian_bounds",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {
                "out": ca.matrix_expand(
                    (ca.jacobian(g_position, self.decision_variables))
                )
            },
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        return {
            "constraint": position_bound_constraint,
            "g": g_bounds_function,
            "jacobian": jacobian_bounds,
        }

    def velocity_bounds_constraint(self):
        g_position = []
        lbg_position = []
        ubg_position = []
        for aaa in self.robots + [self.carried_object]:
            for i in range(0, aaa.bspline_velocity.number_of_control_points):

                control_point = aaa.bspline_velocity.control_points[:7, i]

                lower_limits = aaa.position_lower_limits_velocity
                upper_limits = aaa.position_upper_limits_velocity
                g, lbg, ubg = KinematicOptimization.velocity_bound_constraint(
                    control_point, lower_limits, upper_limits
                )
                g_position.append(g)
                lbg_position.append(lbg)
                ubg_position.append(ubg)

        g_position = ca_utils.veccat(g_position) / self.duration
        lbg_position = ca_utils.veccat(lbg_position)
        ubg_position = ca_utils.veccat(ubg_position)
        g_bounds_function = ca.Function(
            "g_vel_bounds",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {"out": g_position},
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        velocity_bound_constraint = self.opti.bounded(
            lbg_position, g_position, ubg_position
        )
        # self.opti.subject_to(velocity_bound_constraint)
        # subnames = []
        self.subject_to("velocity_bounds_constraint", velocity_bound_constraint)

        jacobian_bounds = ca.Function(
            "jacobian_vel_bounds",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {
                "out": ca.matrix_expand(
                    (ca.jacobian(g_position, self.decision_variables))
                )
            },
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        lam_g_bounds = ca.MX.sym("lam_g", g_position.numel(), 1)
        hessian_bounds = ca.Function(
            "hessian_vel_bounds",
            {
                "dec_variables": self.decision_variables,
                "parameters": self.parameters,
                "lam_g_in": lam_g_bounds,
            }
            | {
                "out": ca.matrix_expand(
                    ca.triu(
                        ca.hessian(
                            ca.dot(g_position, lam_g_bounds), self.decision_variables
                        )[0]
                    )
                )
            },
            ["dec_variables", "parameters", "lam_g_in"],
            ["out"],
            {"always_inline": True},
        )
        return {
            "constraint": velocity_bound_constraint,
            "g": g_bounds_function,
            "jacobian": jacobian_bounds,
            "hessian": hessian_bounds,
        }

    def quaternion_constraint(self):
        g_quaternion = []
        for i in range(0, self.num_control_points):
            g_quaternion.append(
                ca.sumsqr(self.carried_object.bspline.control_points[0:4, i])
            )
        g_quaternion = ca_utils.veccat(g_quaternion)
        g_quaternion_func = ca.Function(
            "g_quaternion",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {"out": g_quaternion},
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        lam_g_quaternion = ca.MX.sym("lam_g", g_quaternion.numel(), 1)
        quaternion_hessian = ca.Function(
            "quaternion_hessian",
            {
                "dec_variables": self.decision_variables,
                "parameters": self.parameters,
                "lam_g_in": lam_g_quaternion,
            }
            | {
                "out": ca.matrix_expand(
                    ca.triu(
                        ca.hessian(
                            ca.dot(g_quaternion, lam_g_quaternion),
                            self.decision_variables,
                        )[0]
                    )
                )
            },
            ["dec_variables", "parameters", "lam_g_in"],
            ["out"],
            {"always_inline": True},
        )

        constraint_quaternion = self.opti.bounded(1, g_quaternion, 1)
        # self.opti.subject_to(constraint_quaternion)
        self.subject_to("quaternion_constraint", constraint_quaternion)

        jacobian_quaternion = ca.Function(
            "jacobian_quaternion",
            {"dec_variables": self.decision_variables, "parameters": self.parameters}
            | {
                "out": ca.matrix_expand(
                    (ca.jacobian(g_quaternion, self.decision_variables))
                )
            },
            ["dec_variables", "parameters"],
            ["out"],
            {"always_inline": True},
        )
        return {
            "constraint": constraint_quaternion,
            "g": g_quaternion_func,
            "jacobian": jacobian_quaternion,
            "hessian": quaternion_hessian,
        }

    def get_grasping_constraint_function(self):

        q_obj = ca.MX.sym("q_obj", self.carried_object.wrapper.num_positions())
        frame_pose_object_1 = self.carried_object.wrapper.calc_frame_pose_in_frame(
            q_obj,
            self.carried_object_EE_1_frame,
            self.carried_object.plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        frame_pose_object_2 = self.carried_object.wrapper.calc_frame_pose_in_frame(
            q_obj,
            self.carried_object_EE_2_frame,
            self.carried_object.plant.world_frame(),
            clean_small_coeffs=1e-6,
        )

        q_robot_1 = ca.MX.sym("q_robot_1", self.robots[0].wrapper.num_positions())
        grasping_point_1 = ca.MX.sym("grasping_point", 3)
        grasping_rotation_matrix_1 = ca.MX.sym("grasping_rotation_matrix", 3, 3)
        frame_pose_robot_1 = self.robots[0].wrapper.calc_frame_pose_in_frame(
            q_robot_1,
            self.robot_1_EE_frame,
            self.robots[0].plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        # frame_1_in_2 = frame_pose_robot_1 @ casadi_4x4_inverse(frame_pose_object_1)
        frame_1 = frame_pose_object_1
        frame_2 = frame_pose_robot_1
        rotation_1 = grasping_rotation_matrix_1
        rotation_2 = ca.DM.eye(3)
        frame_2_in_1 = casadi_frame_A_in_frame_B(frame_2, frame_1)
        g_translation, lbg_translation, ubg_translation = (
            KinematicOptimization.fwd_kin_translation_constraint(
                frame_1, grasping_point_1, frame_2, [0, 0, 0], [0, 0, 0], [0, 0, 0]
            )
        )
        g_orientation, lbg_orientation, ubg_orientation = (
            KinematicOptimization.fwd_kin_orientation_constraint(
                rotation_1, rotation_2, frame_2_in_1, 0
            )
        )
        g = veccat(g_translation, g_orientation)
        lbg = veccat(lbg_translation, lbg_orientation)
        ubg = veccat(ubg_translation, ubg_orientation)

        q_robot_2 = ca.MX.sym("q_robot_2", self.robots[1].wrapper.num_positions())
        grasping_point_2 = ca.MX.sym("grasping_point_2", 3)
        grasping_rotation_matrix_2 = ca.MX.sym("grasping_rotation_matrix_2", 3, 3)
        frame_pose_robot_2 = self.robots[1].wrapper.calc_frame_pose_in_frame(
            q_robot_2,
            self.robot_2_EE_frame,
            self.robots[1].plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        frame_1 = frame_pose_object_2
        frame_2 = frame_pose_robot_2
        rotation_1 = grasping_rotation_matrix_2
        rotation_2 = ca.DM.eye(3)
        frame_2_in_1 = casadi_frame_A_in_frame_B(frame_2, frame_1)
        g_translation, lbg_translation, ubg_translation = (
            KinematicOptimization.fwd_kin_translation_constraint(
                frame_1, grasping_point_2, frame_2, [0, 0, 0], [0, 0, 0], [0, 0, 0]
            )
        )
        g_orientation, lbg_orientation, ubg_orientation = (
            KinematicOptimization.fwd_kin_orientation_constraint(
                rotation_1, rotation_2, frame_2_in_1, 0
            )
        )
        g = veccat(g, g_translation, g_orientation)
        lbg = veccat(lbg, lbg_translation, lbg_orientation)
        ubg = veccat(ubg, ubg_translation, ubg_orientation)

        x = veccat(q_obj, q_robot_1, q_robot_2, grasping_point_1, grasping_point_2)
        p = veccat(grasping_rotation_matrix_1, grasping_rotation_matrix_2)
        # return ca.Function('grasping_constraint_function',
        #                                                 {
        #                                                     'q_obj':q_obj,
        #                                                     'q_robot_1':q_robot_1,
        #                                                     'q_robot_2':q_robot_2,
        #                                                     'grasping_point_1':grasping_point_1,
        #                                                     'grasping_point_2':grasping_point_2,
        #                                                     'grasping_rotation_matrix_1':grasping_rotation_matrix_1,
        #                                                     'grasping_rotation_matrix_2':grasping_rotation_matrix_2,
        #                                                     'g': g,
        #                                                     'lbg':lbg,
        #                                                     'ubg':ubg
        #                                                 },
        #                                                 ['q_obj','q_robot_1','q_robot_2','grasping_point_1','grasping_point_2','grasping_rotation_matrix_1', 'grasping_rotation_matrix_2'],
        #                                                 ['g','lbg','ubg'],

        #                                                 {
        #                                                     'post_expand':True,
        #                                                     'post_expand_options':{'cse':True,
        #                                                                         'is_diff_in':[True,True,True,True,True,False,False],
        #                                                                         'is_diff_out':[True,False,False],}
        #                                                 }
        #                                             )
        g_function = ca.Function(
            "grasping_constraint_function",
            {
                "x": x,
                "p": p,
                "g": g,
            },
            ["x", "p"],
            [
                "g",
            ],
            {
                "post_expand": True,
                "post_expand_options": {
                    "cse": True,
                    "is_diff_in": [True, False],
                    "is_diff_out": [
                        True,
                    ],
                },
            },
        )
        lbg_function = ca.Function(
            "grasping_constraint_function",
            {
                "p": p,
                "lbg": lbg,
            },
            ["p"],
            [
                "lbg",
            ],
        )
        ubg_function = ca.Function(
            "grasping_constraint_function",
            {
                "p": p,
                "ubg": ubg,
            },
            ["p"],
            [
                "ubg",
            ],
        )
        return {"g": g_function, "lbg": lbg_function, "ubg": ubg_function}

    def get_object_pose_constraint_function(
        self,
    ):
        q_MX = ca.MX.sym("q", self.carried_object.wrapper.num_positions())
        translation_MX = ca.MX.sym("translation_MX", 3)
        rotation_MX = ca.MX.sym("rotation_MX", 3, 3)
        EE_frame_pose = self.carried_object.wrapper.calc_frame_pose_in_frame(
            q_MX,
            self.carried_object_frame,
            self.carried_object.plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        g_translation, lbg_translation, ubg_translation = (
            KinematicOptimization.fwd_kin_translation_constraint(
                EE_frame_pose,
                [0, 0, 0],
                ca.DM.eye(4),
                translation_MX,
                [0, 0, 0],
                [0, 0, 0],
            )
        )
        g_orientation, lbg_orientation, ubg_orientation = (
            KinematicOptimization.fwd_kin_orientation_constraint(
                rotation_MX, ca.DM.eye(3), EE_frame_pose, 0
            )
        )
        g = ca.cse(veccat(g_translation, g_orientation))
        lbg = veccat(lbg_translation, lbg_orientation)
        ubg = veccat(ubg_translation, ubg_orientation)
        x = veccat(
            q_MX,
        )
        p = veccat(translation_MX, rotation_MX)
        g_function = ca.Function(
            "object_pose_constraint_function",
            {
                "x": x,
                "p": p,
                "g": g,
            },
            ["x", "p"],
            [
                "g",
            ],
            {
                "post_expand": True,
                "post_expand_options": {
                    "cse": True,
                    "is_diff_in": [True, False],
                    "is_diff_out": [
                        True,
                    ],
                },
            },
        )
        lbg_function = ca.Function(
            "object_pose_constraint_function",
            {
                "p": p,
                "lbg": lbg,
            },
            ["p"],
            [
                "lbg",
            ],
        )
        ubg_function = ca.Function(
            "object_pose_constraint_function",
            {
                "p": p,
                "ubg": ubg,
            },
            ["p"],
            [
                "ubg",
            ],
        )
        return {"g": g_function, "lbg": lbg_function, "ubg": ubg_function}
        return ca.Function(
            "fwd_kin_pose_constraint_func",
            {"x": x, "p": p, "g": g, "lbg": lbg, "ubg": ubg},
            ["x", "p"],
            ["g", "lbg", "ubg"],
            {
                "post_expand": True,
                "post_expand_options": {
                    "cse": True,
                    "is_diff_in": [True, False],
                    "is_diff_out": [True, False, False],
                },
            },
        )

    def get_object_pose_constraint_function(
        self,
    ):
        q_MX = ca.MX.sym("q", self.carried_object.wrapper.num_positions())
        translation_MX = ca.MX.sym("translation_MX", 3)
        rotation_MX = ca.MX.sym("rotation_MX", 3, 3)
        EE_frame_pose = self.carried_object.wrapper.calc_frame_pose_in_frame(
            q_MX,
            self.carried_object_frame,
            self.carried_object.plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        g_translation, lbg_translation, ubg_translation = (
            KinematicOptimization.fwd_kin_translation_constraint(
                EE_frame_pose,
                [0, 0, 0],
                ca.DM.eye(4),
                translation_MX,
                [0, 0, 0],
                [0, 0, 0],
            )
        )
        g_orientation, lbg_orientation, ubg_orientation = (
            KinematicOptimization.fwd_kin_orientation_constraint(
                rotation_MX, ca.DM.eye(3), EE_frame_pose, 0
            )
        )
        g = ca.cse(veccat(g_translation, g_orientation))
        lbg = veccat(lbg_translation, lbg_orientation)
        ubg = veccat(ubg_translation, ubg_orientation)
        x = veccat(
            q_MX,
        )
        p = veccat(translation_MX, rotation_MX)
        g_function = ca.Function(
            "object_pose_constraint_function",
            {
                "x": x,
                "p": p,
                "g": g,
            },
            ["x", "p"],
            [
                "g",
            ],
            {
                "post_expand": True,
                "post_expand_options": {
                    "cse": True,
                    "is_diff_in": [True, False],
                    "is_diff_out": [
                        True,
                    ],
                },
            },
        )
        lbg_function = ca.Function(
            "object_pose_constraint_function",
            {
                "p": p,
                "lbg": lbg,
            },
            ["p"],
            [
                "lbg",
            ],
        )
        ubg_function = ca.Function(
            "object_pose_constraint_function",
            {
                "p": p,
                "ubg": ubg,
            },
            ["p"],
            [
                "ubg",
            ],
        )
        return {"g": g_function, "lbg": lbg_function, "ubg": ubg_function}
        return ca.Function(
            "fwd_kin_pose_constraint_func",
            {"x": x, "p": p, "g": g, "lbg": lbg, "ubg": ubg},
            ["x", "p"],
            ["g", "lbg", "ubg"],
            {
                "post_expand": True,
                "post_expand_options": {
                    "cse": True,
                    "is_diff_in": [True, False],
                    "is_diff_out": [True, False, False],
                },
            },
        )

    def get_object_pose_constraint_function_2(
        self,
    ):
        # This could be simply some lbx ubx bound
        q_MX = ca.MX.sym("q", self.carried_object.wrapper.num_positions())
        translation_MX = ca.MX.sym("translation_MX", 3)
        rotation_MX = ca.MX.sym("rotation_MX", 3, 3)
        position_lower_bounds = ca.MX.sym("position_lower_bounds", 3)
        position_upper_bounds = ca.MX.sym("position_upper_bounds", 3)
        angle_upper_bound = ca.MX.sym("angle_upper_bound", 1)
        EE_frame_pose = self.carried_object.wrapper.calc_frame_pose_in_frame(
            q_MX,
            self.carried_object_frame,
            self.carried_object.plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        g_translation, lbg_translation, ubg_translation = (
            KinematicOptimization.fwd_kin_translation_constraint(
                EE_frame_pose,
                [0, 0, 0],
                ca.DM.eye(4),
                translation_MX,
                position_lower_bounds,
                position_upper_bounds,
            )
        )
        g_orientation, lbg_orientation, ubg_orientation = (
            KinematicOptimization.fwd_kin_orientation_constraint(
                rotation_MX, ca.DM.eye(3), EE_frame_pose, angle_upper_bound
            )
        )
        g = ca.cse(veccat(g_translation, g_orientation))
        lbg = veccat(lbg_translation, lbg_orientation)
        ubg = veccat(ubg_translation, ubg_orientation)
        x = veccat(
            q_MX,
        )
        p = veccat(
            translation_MX,
            rotation_MX,
            position_lower_bounds,
            position_upper_bounds,
            angle_upper_bound,
        )
        g_function = ca.Function(
            "object_pose_constraint_function",
            {
                "x": x,
                "p": p,
                "g": g,
            },
            ["x", "p"],
            [
                "g",
            ],
            {
                "post_expand": True,
                "post_expand_options": {
                    "cse": True,
                    "is_diff_in": [True, False],
                    "is_diff_out": [
                        True,
                    ],
                },
            },
        )
        lbg_function = ca.Function(
            "object_pose_constraint_function",
            {
                "p": p,
                "lbg": lbg,
            },
            ["p"],
            [
                "lbg",
            ],
        )
        ubg_function = ca.Function(
            "object_pose_constraint_function",
            {
                "p": p,
                "ubg": ubg,
            },
            ["p"],
            [
                "ubg",
            ],
        )
        return {"g": g_function, "lbg": lbg_function, "ubg": ubg_function}

    def IK_pose(self, pose, tries=20, gripper_translation_slack=0.1):
        q_middle: np.ndarray = 0.5 * (
            self.robots[0].plant.GetPositionLowerLimits()
            + self.robots[1].plant.GetPositionUpperLimits()
        )
        q_diagonal = np.ones_like(q_middle)
        cost_matrix_Q = np.diag(q_diagonal)
        ik_result_1 = None
        for i in range(0, tries):
            try:
                ik_result_1 = self.inverse_kinematics_grasping_drake(
                    object_pose=pose,
                    plant=self.robots[0].plant,
                    robot_EE_frame=self.robot_1_EE_frame,
                    robot_grasp_rotation=RollPitchYaw([0, np.pi, 0]).ToRotationMatrix(),
                    gripper_translation_lower_bound=np.array([0.1, 0.0, 0.0]),
                    gripper_translation_upper_bound=np.array([0.15, 0.0, 0.0]),
                    q_middle=q_middle,
                    cost_matrix_Q=cost_matrix_Q * 10,
                    initial_guess=np.random.randn(7),
                )
                break
            except:
                pass
        if ik_result_1 is None:
            raise ValueError("IK failed.")

        q_middle: np.ndarray = 0.5 * (
            self.robots[0].plant.GetPositionLowerLimits()
            + self.robots[1].plant.GetPositionUpperLimits()
        )
        q_diagonal = np.ones_like(q_middle)
        cost_matrix_Q = np.diag(q_diagonal)
        ik_result_2 = None
        for i in range(0, tries):
            try:
                ik_result_2 = self.inverse_kinematics_grasping_drake(
                    object_pose=pose,
                    plant=self.robots[1].plant,
                    robot_EE_frame=self.robot_2_EE_frame,
                    robot_grasp_rotation=RotationMatrix.MakeXRotation(np.pi),
                    gripper_translation_lower_bound=np.array([0.1, 0.0, 0.0]),
                    gripper_translation_upper_bound=np.array([0.15, 0.0, 0.0]),
                    q_middle=q_middle,
                    cost_matrix_Q=cost_matrix_Q * 10,
                    initial_guess=np.random.randn(7),
                )  # # object_middle_pose = RigidTransform()
                break
            except:
                pass
        if ik_result_2 is None:
            raise ValueError("IK failed.")
        return ik_result_1, ik_result_2

    def inverse_kinematics_grasping_drake(
        self,
        plant,
        object_pose: RigidTransform,
        robot_EE_frame,
        with_joint_limits=True,
        constrain_EE_translation=True,
        constrain_EE_orientation=True,
        gripper_translation_lower_bound=np.array([0.0, 0.0, 0.0]),
        gripper_translation_upper_bound=np.array([0.0, 0.0, 0.0]),
        robot_grasp_rotation=RotationMatrix(),
        gripper_orientation_slack=0.0,
        q_middle=None,
        cost_matrix_Q=None,
        initial_guess=None,
    ):

        plant_context_optimization = plant.CreateDefaultContext()
        ik = pydrake.all.InverseKinematics(
            plant, plant_context_optimization, with_joint_limits=with_joint_limits
        )

        if constrain_EE_translation:
            ik.AddPositionConstraint(
                plant.world_frame(),
                object_pose.translation(),
                robot_EE_frame,
                gripper_translation_lower_bound,
                gripper_translation_upper_bound,
            )

        if constrain_EE_orientation:
            ik.AddOrientationConstraint(
                plant.world_frame(),
                robot_grasp_rotation,
                robot_EE_frame,
                object_pose.rotation(),
                gripper_orientation_slack,
            )

        # add cost for being away from the middle of the joint limits
        q = ik.q()

        ik.get_mutable_prog().AddQuadraticErrorCost(cost_matrix_Q, q_middle, q)

        if initial_guess is not None:
            ik.prog().SetInitialGuess(q, initial_guess)
        else:
            q0 = plant.GetPositions(plant_context_optimization)
            ik.prog().SetInitialGuess(q, q0)
        t0 = time.time()
        result = pydrake.all.Solve(ik.prog())
        if not result.is_success():
            raise ValueError("IK failed.")
        return result.GetSolution(q)

    def get_x0_p(self, optimization_data):
        # error here usually means some variable wasn't actually used
        baked_copy = self.opti.advanced.baked_copy()
        x0 = ca.DM(*baked_copy.x.shape)
        p = ca.DM(*baked_copy.p.shape)
        for k, v in optimization_data.initial_guess.as_dictionary().items():
            var_meta = baked_copy.get_meta(k)
            x0[var_meta.start : var_meta.stop] = ca.vec(v)
        i = 0
        for k, v in optimization_data.parameters.as_dictionary().items():
            var_meta = baked_copy.get_meta(k)
            start = i
            stop = start + var_meta.n * var_meta.m
            i = stop
            p[start:stop] = ca.vec(v)
        return x0, p

    def get_inner_samples_function(self, samples, lam_g_shape=None):
        X = []
        P = []
        for s in samples:
            x_s = []
            p_s = []
            x_s += [self.carried_object.bspline.evaluate(s)]
            x_s += [self.robots[0].bspline.evaluate(s)]
            x_s += [self.robots[1].bspline.evaluate(s)]
            x_s += [self.grasping_point_EE_1]
            x_s += [self.grasping_point_EE_2]
            for robot_name in self.svm_weights:
                robot_slack = self.svm_slack[robot_name]
                for group_name in robot_slack:
                    slack = self.svm_slack[robot_name][group_name]
                    x_s += [slack]
            # 
            
            p_s += [self.grasping_rotation_matrix_EE_1]
            p_s += [self.grasping_rotation_matrix_EE_2]
            X += [ca_utils.veccat(x_s)]
            P += [ca_utils.veccat(p_s)]
        X = ca.cse(ca_utils.horzcat(X))
        P = ca.cse(ca_utils.horzcat(P))
        # X,P = ca.Function('temp',[self.decision_variables,self.parameters],[X,P],).expand()(self.decision_variables,self.parameters)
        inner_out = {"x": X, "p": P}

        inner_args = {
            "dec_variables": self.decision_variables,
            "parameters": self.parameters,
        }
        if lam_g_shape:
            if self.diff_co_options:
                for robot_name in self.svm_weights:
                    robot_support_vectors = self.svm_support_vectors[robot_name]
                    robot_svm_weights = self.svm_weights[robot_name]
                    robot_polynomial_weights = self.svm_polynomial_weights[robot_name]
                    for group_name in robot_svm_weights:
                        fk_S = robot_support_vectors[group_name]
                        A = robot_svm_weights[group_name]
                        pol_A = robot_polynomial_weights[group_name]
                        inner_out = inner_out | {
                            "fk_S" + robot_name + "_" + group_name: fk_S,
                            "A" + robot_name + "_" + group_name: A,
                            "pol_A" + robot_name + "_" + group_name: pol_A,
                        }

                    # inner_out = inner_out | {"fk_S" + : fk_S, "A": A, "pol_A": pol_A}
                # fk_S = self.svm_support_vectors
                # A = self.svm_weights
                # pol_A = self.svm_polynomial_weights
                # inner_out = inner_out | {"fk_S": fk_S, "A": A, "pol_A": pol_A}
            lam_g = ca.MX.sym("lam_g", *lam_g_shape)
            inner_out = inner_out | {"lam_g": lam_g}
            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g} | inner_out,
                ["dec_variables", "parameters", "lam_g_in"],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                },
            )

            # jac_jac_inner = inner_function_lagrangian.wrap().jacobian().jacobian()
            jac_inner = inner_function_lagrangian.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function_lagrangian.numel_in(i1)
                    * inner_function_lagrangian.numel_out(o),
                    inner_function_lagrangian.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g} | inner_out,
                ["dec_variables", "parameters", "lam_g_in"],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )

            return inner_function_lagrangian
        else:
            if self.diff_co_options:
                for robot_name in self.svm_weights:
                    robot_support_vectors = self.svm_support_vectors[robot_name]
                    robot_svm_weights = self.svm_weights[robot_name]
                    robot_polynomial_weights = self.svm_polynomial_weights[robot_name]
                    robot_lbg = self.svm_lbg[robot_name]
                    robot_ubg = self.svm_ubg[robot_name]
                    for group_name in robot_svm_weights:
                        fk_S = robot_support_vectors[group_name]
                        A = robot_svm_weights[group_name]
                        pol_A = robot_polynomial_weights[group_name]
                        lbg = robot_lbg[group_name]
                        ubg = robot_ubg[group_name]
                        inner_out = inner_out | {
                            "fk_S" + robot_name + "_" + group_name: fk_S,
                            "A" + robot_name + "_" + group_name: A,
                            "pol_A" + robot_name + "_" + group_name: pol_A,
                            robot_name + "_" + group_name + "_lbg": lbg,
                            robot_name + "_" + group_name + "_ubg": ubg,
                        }
                        # 'score_lbg':self.score_lbg,'score_ubg':self.score_ubg,'pol_A':pol_A
            inner_function = ca.Function(
                "inner",
                inner_args | inner_out,
                [
                    "dec_variables",
                    "parameters",
                ],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                },
            )
            # jac_jac_inner = inner_function_lagrangian.wrap().jacobian().jacobian()
            jac_inner = inner_function.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function.name_in(),
                    inner_function.name_in(),
                    inner_function.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function.numel_in(i1) * inner_function.numel_out(o),
                    inner_function.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function = ca.Function(
                "inner",
                inner_args | inner_out,
                [
                    "dec_variables",
                    "parameters",
                ],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )
            return inner_function

    def get_inner_pose_function(self, samples, lam_g_shape=None):
        X = []
        P = []
        for s in samples:
            x_s = []
            p_s = []
            x_s += [self.carried_object.bspline.evaluate(s)]
            if s == 0:
                p_s += [self.object_start_position]
                p_s += [self.object_start_rotation_matrix]
            elif s == 1:
                p_s += [self.object_end_position]
                p_s += [self.object_end_rotation_matrix]
            else:
                raise ValueError(f"s must be 0 or 1, was {s}")
            X += [ca_utils.veccat(x_s)]
            P += [ca_utils.veccat(p_s)]
        X = ca.cse(ca_utils.horzcat(X))
        P = ca.cse(ca_utils.horzcat(P))

        inner_args = {
            "dec_variables": self.decision_variables,
            "parameters": self.parameters,
        }
        if lam_g_shape:
            lam_g = ca.MX.sym("lam_g", *lam_g_shape)

            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g, "x": X, "p": P, "lam_g": lam_g},
                ["dec_variables", "parameters", "lam_g_in"],
                ["x", "p", "lam_g"],
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                },
            )
            jac_inner = inner_function_lagrangian.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function_lagrangian.numel_in(i1)
                    * inner_function_lagrangian.numel_out(o),
                    inner_function_lagrangian.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g, "x": X, "p": P, "lam_g": lam_g},
                ["dec_variables", "parameters", "lam_g_in"],
                ["x", "p", "lam_g"],
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )
            return inner_function_lagrangian
        else:
            inner_function = ca.Function(
                "inner",
                inner_args
                | {
                    "x": X,
                    "p": P,
                },
                [
                    "dec_variables",
                    "parameters",
                ],
                [
                    "x",
                    "p",
                ],
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                },
            )
            jac_inner = inner_function.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function.name_in(),
                    inner_function.name_in(),
                    inner_function.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function.numel_in(i1) * inner_function.numel_out(o),
                    inner_function.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function = ca.Function(
                "inner",
                inner_args
                | {
                    "x": X,
                    "p": P,
                },
                [
                    "dec_variables",
                    "parameters",
                ],
                [
                    "x",
                    "p",
                ],
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )
            return inner_function

    def get_inner_pose_function_2(self, samples, lam_g_shape=None):
        X = []
        P = []
        for s in samples:
            x_s = []
            p_s = []
            x_s += [self.carried_object.bspline.evaluate(s)]
            if s == 0:
                p_s += [self.object_start_position]
                p_s += [self.object_start_rotation_matrix]
                p_s += [self.start_position_lower_bounds]
                p_s += [self.start_position_upper_bounds]
                p_s += [self.start_angle_bounds]

            elif s == 1:
                p_s += [self.object_end_position]
                p_s += [self.object_end_rotation_matrix]
                p_s += [self.end_position_lower_bounds]
                p_s += [self.end_position_upper_bounds]
                p_s += [self.end_angle_bounds]
            # self.start_position_lower_bounds = self.parameter("start_position_lower_bounds",3,1)
            # self.end_position_lower_bounds = self.parameter("end_position_lower_bounds",3,1)
            # self.start_position_upper_bounds = self.parameter("start_position_upper_bounds",3,1)
            # self.end_position_upper_bounds = self.parameter("end_position_upper_bounds",3,1)
            # self.start_angle_bounds = self.parameter("start_angle_bounds",1,1)
            # self.end_angle_bounds = self.parameter("end_angle_bounds",1,1)
            else:
                raise ValueError(f"s must be 0 or 1, was {s}")
            X += [ca_utils.veccat(x_s)]
            P += [ca_utils.veccat(p_s)]
        X = ca.cse(ca_utils.horzcat(X))
        P = ca.cse(ca_utils.horzcat(P))

        inner_args = {
            "dec_variables": self.decision_variables,
            "parameters": self.parameters,
        }
        if lam_g_shape:
            lam_g = ca.MX.sym("lam_g", *lam_g_shape)

            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g, "x": X, "p": P, "lam_g": lam_g},
                ["dec_variables", "parameters", "lam_g_in"],
                ["x", "p", "lam_g"],
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                },
            )
            jac_inner = inner_function_lagrangian.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function_lagrangian.numel_in(i1)
                    * inner_function_lagrangian.numel_out(o),
                    inner_function_lagrangian.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g, "x": X, "p": P, "lam_g": lam_g},
                ["dec_variables", "parameters", "lam_g_in"],
                ["x", "p", "lam_g"],
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )
            return inner_function_lagrangian
        else:
            inner_function = ca.Function(
                "inner",
                inner_args
                | {
                    "x": X,
                    "p": P,
                },
                [
                    "dec_variables",
                    "parameters",
                ],
                [
                    "x",
                    "p",
                ],
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                },
            )
            jac_inner = inner_function.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function.name_in(),
                    inner_function.name_in(),
                    inner_function.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function.numel_in(i1) * inner_function.numel_out(o),
                    inner_function.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function = ca.Function(
                "inner",
                inner_args
                | {
                    "x": X,
                    "p": P,
                },
                [
                    "dec_variables",
                    "parameters",
                ],
                [
                    "x",
                    "p",
                ],
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )
            return inner_function

    def get_inner_gaze_function(self, samples, lam_g_shape=None):
        X = []
        P = []
        for s in samples:
            x_s = []
            x_s += [self.robots[0].bspline.evaluate(s)]
            # self.gaze_cost_weight: ca.MX = self.parameter("gaze_cost_weight",1,1)
            # self.gaze_weights: ca.MX  = self.parameter("gaze_weights",self.gaze_options.num_obstacle_categories,self.gaze_options.map_size_1*self.gaze_options.map_size_2*self.gaze_options.num_support_vectors_unmapped)
            # self.gaze_support_vectors: ca.MX  = self.parameter("gaze_support_vectors",self.robots[0].forward_kinematic_diff_co.fk_casadi.numel_out(),self.gaze_options.map_size_1*self.gaze_options.map_size_2*self.gaze_options.num_support_vectors_unmapped)
            X += [ca_utils.veccat(x_s)]
        X = ca.cse(ca_utils.horzcat(X))
        # X,P = ca.Function('temp',[self.decision_variables,self.parameters],[X,P],).expand()(self.decision_variables,self.parameters)
        inner_out = {
            "x": X,
        }

        inner_args = {
            "dec_variables": self.decision_variables,
            "parameters": self.parameters,
        }
        if lam_g_shape:
            if self.gaze_options:
                for robot_name in self.vision_weights:
                    robot_support_vectors = self.vision_support_vectors[robot_name]
                    robot_svm_weights = self.vision_weights[robot_name]
                    robot_polynomial_weights = self.vision_polynomial_weights[robot_name]
                    for group_name in robot_svm_weights:
                        fk_S = robot_support_vectors[group_name]
                        A = robot_svm_weights[group_name]
                        pol_A = robot_polynomial_weights[group_name]
                        inner_out = inner_out | {
                            "fk_S" + robot_name + "_" + group_name: fk_S,
                            "A" + robot_name + "_" + group_name: A,
                            "pol_A" + robot_name + "_" + group_name: pol_A,
                        }
                inner_out = inner_out | {'cost_weight':self.vision_cost_weight}
                # fk_S = self.gaze_support_vectors
                # A = self.gaze_weights
                # cost_weight = self.gaze_cost_weight
                # inner_out = inner_out | {
                #     "fk_S": fk_S,
                #     "A": A,
                #     "cost_weight": cost_weight,
                # }
            lam_g = ca.MX.sym("lam_g", *lam_g_shape)
            inner_out = inner_out | {"lam_g": lam_g}
            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g} | inner_out,
                ["dec_variables", "parameters", "lam_g_in"],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                },
            )

            # jac_jac_inner = inner_function_lagrangian.wrap().jacobian().jacobian()
            jac_inner = inner_function_lagrangian.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_in(),
                    inner_function_lagrangian.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function_lagrangian.numel_in(i1)
                    * inner_function_lagrangian.numel_out(o),
                    inner_function_lagrangian.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function_lagrangian = ca.Function(
                "inner",
                inner_args | {"lam_g_in": lam_g} | inner_out,
                ["dec_variables", "parameters", "lam_g_in"],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [True, False, False],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )

            return inner_function_lagrangian
        else:
            if self.gaze_options:
                # fk_S = self.gaze_support_vectors
                # A = self.gaze_weights
                # cost_weight = self.gaze_cost_weight
                # inner_out = inner_out | {
                #     "fk_S": fk_S,
                #     "A": A,
                #     "cost_weight": cost_weight,
                # }
                for robot_name in self.vision_weights:
                    robot_support_vectors = self.vision_support_vectors[robot_name]
                    robot_svm_weights = self.vision_weights[robot_name]
                    robot_polynomial_weights = self.vision_polynomial_weights[robot_name]
                    for group_name in robot_svm_weights:
                        fk_S = robot_support_vectors[group_name]
                        A = robot_svm_weights[group_name]
                        pol_A = robot_polynomial_weights[group_name]
                        inner_out = inner_out | {
                            "fk_S" + robot_name + "_" + group_name: fk_S,
                            "A" + robot_name + "_" + group_name: A,
                            "pol_A" + robot_name + "_" + group_name: pol_A,
                        }
                inner_out = inner_out | {'cost_weight':self.vision_cost_weight}
            inner_function = ca.Function(
                "inner",
                inner_args | inner_out,
                [
                    "dec_variables",
                    "parameters",
                ],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                },
            )
            # jac_jac_inner = inner_function_lagrangian.wrap().jacobian().jacobian()
            jac_inner = inner_function.jacobian()
            jac_jac_inner_mx_in = {
                "out_" + name: ca.MX(*jac_inner.size_in(name))
                for name in jac_inner.name_in()
            } | {name: jac_inner.mx_in(name) for name in jac_inner.name_in()}

            combinations = [
                (i1, i2, o)
                for (i1, i2, o) in itertools.product(
                    inner_function.name_in(),
                    inner_function.name_in(),
                    inner_function.name_out(),
                )
            ]
            jac_jac_out = {
                f"jac_jac_{o}_{i1}_{i2}": ca.MX(
                    inner_function.numel_in(i1) * inner_function.numel_out(o),
                    inner_function.numel_in(i2),
                )
                for i1, i2, o in combinations
            }

            jac_jac_inner = ca.Function(
                "jac_" + jac_inner.name(),
                jac_jac_inner_mx_in | jac_jac_out,
                jac_jac_inner_mx_in.keys(),
                jac_jac_out.keys(),
                {"always_inline": True},
            )
            inner_function = ca.Function(
                "inner",
                inner_args | inner_out,
                [
                    "dec_variables",
                    "parameters",
                ],
                inner_out.keys(),
                {
                    "always_inline": True,
                    "is_diff_in": [
                        True,
                        False,
                    ],
                    "jacobian_options": {
                        "custom_jacobian": jac_jac_inner,
                    },
                },
            )
            return inner_function

    def get_final_hessian(
        self,
        cost_hessians,
        constraints,
        constraints_hessians,
    ):
        assert len(constraints) == len(constraints_hessians)
        baked_copy = self.opti.advanced.baked_copy()

        dec_variables = ca.MX.sym("x", baked_copy.x.numel(), 1)
        parameters = ca.MX.sym("p", baked_copy.p.numel(), 1)
        dec_variables = baked_copy.x
        parameters = baked_copy.p
        kwargs_in = {"dec_variables": dec_variables, "parameters": parameters}

        results = []
        lam_f = ca.MX.sym("lam_f", 1, 1)
        lam_g = ca.MX.sym("lam_g", baked_copy.lam_g.numel(), 1)
        for i, (constraint, hessian) in enumerate(
            zip(constraints, constraints_hessians)
        ):
            constraint_meta = baked_copy.get_meta_con(constraint)
            start = constraint_meta.start
            stop = constraint_meta.stop
            lam_g_h = lam_g[start:stop].reshape(hessian.size_in("lam_g_in"))
            result = hessian.convert_out(
                hessian.call(kwargs_in | {"lam_g_in": lam_g_h})
            )

            # assert len(result) == 1
            for r in result:

                num_split = r.shape[1] // r.shape[0]
                split = ca.horzsplit_n(r, num_split)
                assert split[0].shape == (dec_variables.numel(), dec_variables.numel())
                results += split

        for i, cost_hessian in enumerate(cost_hessians):
            assert cost_hessian.size1_out(0) == dec_variables.numel()
            result = cost_hessian.convert_out(
                cost_hessian.call(kwargs_in | {"lam_g_in": lam_f})
            )
            assert len(result) == 1
            num_split = result[0].shape[1] // result[0].shape[0]
            split = ca.horzsplit_n(result[0], num_split)
            assert split[0].shape == (dec_variables.numel(), dec_variables.numel())
            results += split
        # results = sum(results)
        temp_sx_in = [
            ca.SX.sym(f"temp_{i}", result.sparsity())
            for i, result in enumerate(results)
        ]
        f_sum = ca.Function(
            "f_sum",
            temp_sx_in,
            [sum(temp_sx_in)],
        )
        summed_results = f_sum(*results)
        embedded_hessian = ca.Function(
            "nlp_hess_l",
            [dec_variables, parameters, lam_f, lam_g],
            [summed_results],
            ["x", "p", "lam_f", "lam_g"],
            ["nlp_hess"],
            {"always_inline": True},
        )
        return embedded_hessian

    def get_final_jacobian_and_g(
        self,
        constraints,
        g_functions,
        constraints_jacobians,
    ):
        assert len(constraints) == len(constraints_jacobians)
        baked_copy = self.opti.advanced.baked_copy()

        dec_variables = ca.MX.sym("x", baked_copy.x.numel(), 1)
        parameters = ca.MX.sym("p", baked_copy.p.numel(), 1)
        dec_variables = baked_copy.x
        parameters = baked_copy.p
        kwargs_in = {"dec_variables": dec_variables, "parameters": parameters}

        results_jac = []
        results_g = []
        slices = []
        for i, (constraint, g, jac) in enumerate(
            zip(constraints, g_functions, constraints_jacobians)
        ):
            constraint_meta = baked_copy.get_meta_con(constraint)
            start = constraint_meta.start
            stop = constraint_meta.stop
            slices.append((start, stop))
            result_jac = jac.convert_out(jac.call(kwargs_in))

            results_g += g.convert_out(g.call(kwargs_in))
            # assert len(result) == 1
            size_2_total = sum([r.shape[0] for r in result_jac])
            assert size_2_total == stop - start
            results_jac += [result_jac]

        temp_sx_jac_in = [
            [
                ca.SX.sym(f"temp_{i}", jac.sparsity())
                for j, jac in enumerate(list_of_jacs)
            ]
            for i, list_of_jacs in enumerate(results_jac)
        ]
        temp_sx_g_in = [
            ca.SX.sym(f"temp_{i}", result.sparsity())
            for i, result in enumerate(results_g)
        ]

        result_jac = ca.SX(baked_copy.g.numel(), dec_variables.numel())
        for list_of_jacs, (start, stop) in zip(temp_sx_jac_in, slices):
            step = (stop - start) // len(list_of_jacs)
            for i, jac in enumerate(list_of_jacs):
                result_jac[start + step * i : start + step * (i + 1), :] += jac
            assert stop == start + step * (i + 1)
            # result_jac[start:stop,:] += r_j
        result_g = ca.SX(baked_copy.g.numel(), 1)
        for r_g, (start, stop) in zip(temp_sx_g_in, slices):
            result_g[start:stop] = ca.vec(r_g)
        f_sum_jac_g = ca.Function(
            "f_sum",
            temp_sx_g_in
            + [jac for list_of_jacs in temp_sx_jac_in for jac in list_of_jacs],
            [result_g, result_jac],
        )
        f_sum_g = ca.Function(
            "f_sum",
            temp_sx_g_in,
            [result_g],
        )

        final_nlp_jac_g = f_sum_jac_g.call(
            results_g + [jac for list_of_jacs in results_jac for jac in list_of_jacs]
        )
        final_nlp_g = f_sum_g.call(results_g)
        embedded_jac = ca.Function(
            "nlp_jac_g",
            [
                dec_variables,
                parameters,
            ],
            final_nlp_jac_g,
            ["x", "p"],
            ["g", "jac_g_x"],
            {"always_inline": True},
        )
        embedded_g = ca.Function(
            "nlp_g",
            [
                dec_variables,
                parameters,
            ],
            final_nlp_g,
            ["x", "p"],
            ["g"],
            {"always_inline": True},
        )
        return embedded_g, embedded_jac

    def get_final_grad_f(self, jac_f_functions, f_values):
        baked_copy = self.opti.advanced.baked_copy()

        dec_variables = ca.MX.sym("x", baked_copy.x.numel(), 1)
        parameters = ca.MX.sym("p", baked_copy.p.numel(), 1)
        dec_variables = baked_copy.x
        parameters = baked_copy.p
        kwargs_in = {"dec_variables": dec_variables, "parameters": parameters}

        results_jac = []
        results_f = []
        for i, (jac) in enumerate(
            jac_f_functions,
        ):
            results_jac += jac.convert_out(jac.call(kwargs_in))
        temp_sx_in = [
            ca.SX.sym(f"temp_{i}", result.sparsity())
            for i, result in enumerate(results_jac)
        ]
        f_sum = ca.Function(
            "f_sum",
            temp_sx_in,
            [sum([ca.vec(sx_in) for sx_in in temp_sx_in])],
        )

        summed_results = f_sum(*results_jac)
        nlp_f = ca.Function(
            "nlp_f",
            [
                dec_variables,
                parameters,
            ],
            [sum(f_values)],
            ["x", "p"],
            ["f"],
            {"always_inline": True},
        )
        # nlp_f = ca.Function('nlp_f',{'dec_variables':opt_info_no_collision.decision_variables,'parameters':opt_info_no_collision.parameters,} | {'f':cost},['dec_variables','parameters'],['f'],{'always_inline':True})
        nlp_grad_f = ca.Function(
            "nlp_grad_f",
            [
                dec_variables,
                parameters,
            ],
            [sum(f_values), summed_results],
            ["x", "p"],
            ["f", "nlp_grad_f"],
            {"always_inline": True},
        )
        return nlp_f, nlp_grad_f

    def make_sample_constraints(
        self,
    ):
        modul = ca.SX

        q_obj = modul.sym("q_obj", self.carried_object.wrapper.num_positions())
        frame_pose_object_1 = self.carried_object.wrapper.calc_frame_pose_in_frame(
            q_obj,
            self.carried_object_EE_1_frame,
            self.carried_object.plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        frame_pose_object_2 = self.carried_object.wrapper.calc_frame_pose_in_frame(
            q_obj,
            self.carried_object_EE_2_frame,
            self.carried_object.plant.world_frame(),
            clean_small_coeffs=1e-6,
        )

        q_robot_1 = modul.sym("q_robot_1", self.robots[0].wrapper.num_positions())
        grasping_point_1 = modul.sym("grasping_point", 3)
        grasping_rotation_matrix_1 = modul.sym("grasping_rotation_matrix", 3, 3)
        frame_pose_robot_1 = self.robots[0].wrapper.calc_frame_pose_in_frame(
            q_robot_1,
            self.robot_1_EE_frame,
            self.robots[0].plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        # frame_1_in_2 = frame_pose_robot_1 @ casadi_4x4_inverse(frame_pose_object_1)
        frame_1 = frame_pose_object_1
        frame_2 = frame_pose_robot_1
        rotation_1 = grasping_rotation_matrix_1
        rotation_2 = ca.DM.eye(3)
        frame_2_in_1 = casadi_frame_A_in_frame_B(frame_2, frame_1)
        g_translation, lbg_translation, ubg_translation = (
            KinematicOptimization.fwd_kin_translation_constraint(
                frame_1, grasping_point_1, frame_2, [0, 0, 0], [0, 0, 0], [0, 0, 0]
            )
        )
        g_orientation, lbg_orientation, ubg_orientation = (
            KinematicOptimization.fwd_kin_orientation_constraint(
                rotation_1, rotation_2, frame_2_in_1, 0
            )
        )
        g = veccat(g_translation, g_orientation)
        lbg = veccat(lbg_translation, lbg_orientation)
        ubg = veccat(ubg_translation, ubg_orientation)

        q_robot_2 = modul.sym("q_robot_2", self.robots[1].wrapper.num_positions())
        grasping_point_2 = modul.sym("grasping_point_2", 3)
        grasping_rotation_matrix_2 = modul.sym("grasping_rotation_matrix_2", 3, 3)
        frame_pose_robot_2 = self.robots[1].wrapper.calc_frame_pose_in_frame(
            q_robot_2,
            self.robot_2_EE_frame,
            self.robots[1].plant.world_frame(),
            clean_small_coeffs=1e-6,
        )
        frame_1 = frame_pose_object_2
        frame_2 = frame_pose_robot_2
        rotation_1 = grasping_rotation_matrix_2
        rotation_2 = ca.DM.eye(3)
        frame_2_in_1 = casadi_frame_A_in_frame_B(frame_2, frame_1)
        g_translation, lbg_translation, ubg_translation = (
            KinematicOptimization.fwd_kin_translation_constraint(
                frame_1, grasping_point_2, frame_2, [0, 0, 0], [0, 0, 0], [0, 0, 0]
            )
        )
        g_orientation, lbg_orientation, ubg_orientation = (
            KinematicOptimization.fwd_kin_orientation_constraint(
                rotation_1, rotation_2, frame_2_in_1, 0
            )
        )
        g_grasping = veccat(g, g_translation, g_orientation)
        lbg_grasping = veccat(lbg, lbg_translation, lbg_orientation)
        ubg_grasping = veccat(ubg, ubg_translation, ubg_orientation)
        x_slack = []
        x_slack_dict = {}
        x_slack_index_dict = {}
        i = 0
        for robot_name in self.svm_weights:
            robot_slack = self.svm_slack[robot_name]
            x_slack_dict[robot_name] = {}
            x_slack_index_dict[robot_name] = {}
            for group_name in robot_slack:
                # slack = self.svm_slack[robot_name][group_name]
                x_slack += [modul.sym(f"slack_{robot_name}_group_name", 1)]
                x_slack_dict[robot_name][group_name] = x_slack[-1]
                x_slack_index_dict[robot_name][group_name] = i + q_obj.numel()+ q_robot_1.numel()+ q_robot_2.numel()+ grasping_point_1.numel()+ grasping_point_2.numel()
                i += 1
        x = veccat(q_obj, q_robot_1, q_robot_2, grasping_point_1, grasping_point_2,*x_slack)
        p = veccat(grasping_rotation_matrix_1, grasping_rotation_matrix_2)

        if not self.diff_co_options:
            print("Not using diff co")
            g_grasping_function = ca.Function(
                "sample_constraint_grasping",
                [x, p],
                [ca.cse(g_grasping)],
                [
                    "x",
                    "p",
                ],
                ["g"],
                {
                    "is_diff_in": [
                        True,
                        False,
                    ]
                },
            )
            lbg_grasping_function = ca.Function(
                "lbg_grasping",
                [],
                [
                    lbg_grasping,
                ],
                [],
                [
                    "lbg",
                ],
                {
                    "always_inline": True,
                },
            )
            ubg_grasping_function = ca.Function(
                "ubg_grasping",
                [],
                [
                    ubg_grasping,
                ],
                [],
                [
                    "ubg",
                ],
                {
                    "always_inline": True,
                },
            )
            lagrangian_sample = replace_hessian(get_lagrangian(g_grasping_function))
            return (
                g_grasping_function,
                lbg_grasping_function,
                ubg_grasping_function,
                lagrangian_sample,
            )
        else:

            # num_positions = self.diff_co_options.num_positions
            num_positions = 9
            # num_obstacle_categories = self.diff_co_options["num_obstacle_categories"]
            # map_size_2 = self.diff_co_options.map_size_2
            # num_support_vectors_unmapped = (
            #     self.diff_co_options.num_support_vectors_unmapped
            # )
            # map_size_1 = self.diff_co_options.map_size_1
            # hessian_parallelization = self.diff_co_options.hessian_parallelization
            # jacobian_parallelization = self.diff_co_options.jacobian_parallelization

            configuration_1 = x[q_obj.numel() : q_obj.numel() + q_robot_1.numel()]
            configuration_2 = x[
                q_obj.numel()
                + q_robot_1.numel() : q_obj.numel()
                + q_robot_1.numel()
                + q_robot_2.numel()
            ]

            # reshape_as_pytorch = True
            fk_functions = {}

            total_number_of_groups = 0
            for robot_name, robot in self.robots_by_name.items():
                total_number_of_groups += len(
                    robot.collision_model.forward_kinematics_groups_casadi
                )
            # g_grasping = ca.veccat(ca.SX(total_number_of_groups, 1), g_grasping)
            g_grasping_function = ca.Function(
                "sample_constraint_grasping_only",
                [x, p],
                [ca.cse(ca.veccat(ca.SX(total_number_of_groups, 1), g_grasping))],
                [
                    "x",
                    "p",
                ],
                ["g"],
                {
                    "is_diff_in": [
                        True,
                        False,
                    ]
                },
            )

            lam_g = ca.SX.sym("lam_g", total_number_of_groups + g_grasping.numel(), 1)
            lam_g_score = lam_g[:total_number_of_groups]
            l_grasping = ca.dot(lam_g[total_number_of_groups:], g_grasping)
            grasping_l_hessian_function = ca.Function(
                "grasping_l_hess",
                [x, p, lam_g],
                [ca.cse(ca.hessian(l_grasping, x)[0])],
                ["x", "p", "lam_g"],
                ["g_hess"],
            )

            group_i = 0
            score_functions = {}
            score_hessian_functions = {}
            score_jacobian_functions = {}
            for q, robot, robot_options in zip(
                (configuration_1, configuration_2),
                self.robots_by_name.values(),
                self.diff_co_options["robots"],
            ):
                robot_name = robot_options["name"]
                fk_functions[robot_name] = {}
                for (
                    group_name,
                    fk_function,
                ) in robot.collision_model.forward_kinematics_groups_casadi.items():
                    fk_functions[robot_name][group_name] = ca.Function(
                        "fk",
                        [q],
                        [fk_function(q).T.reshape((-1, 1))],
                        {"cse": True},
                    )
                    slack = x_slack_dict[robot_name][group_name]
                    fk_q = fk_functions[robot_name][group_name].call([q])[0]
                    group_options: DiffCoGroupOptions = robot_options["groups"][
                        group_name
                    ]
                    num_obstacle_categories = group_options["num_obstacle_categories"]
                    num_support_vectors_block = group_options[
                        "num_support_points_per_block"
                    ]

                    W_block = modul.sym(
                        "A", num_obstacle_categories, num_support_vectors_block
                    )
                    fk_S_block = modul.sym(
                        "fk_S",
                        fk_functions[robot_name][group_name].numel_out(),
                        num_support_vectors_block,
                    )
                    polyharmonic_kernel_function = make_polyharmonic_kernel_function(
                        fk_functions[robot_name][group_name].numel_out()
                    )
                    all_score = modul(total_number_of_groups + g_grasping.numel(), 1)
                    score = polyharmonic_kernel_function.map(
                        num_support_vectors_block, [True, False]
                    )(fk_q, fk_S_block) @ (W_block.T)
                    all_score[group_i] = score
                    group_score_function = ca.Function(
                        "g_score_" + robot_name + "_" + group_name,
                        [x, fk_S_block, W_block],
                        [ca.cse(all_score)],
                        [
                            "x",
                            "fk_S" + robot_name + "_" + group_name,
                            "A" + robot_name + "_" + group_name,
                        ],
                        ["g"],
                        {
                            "is_diff_in": [
                                True,
                                False,
                                False,
                            ]
                        },
                    )
                    group_score_function_repsum = group_score_function.map(
                        group_options["inner_map_size"]
                        * group_options["outer_map_size"],
                        [
                            True,
                            False,
                            False,
                        ],
                        [True],
                    )
                    mx_in = group_score_function_repsum.mx_in()
                    slack_MX = mx_in[0][x_slack_index_dict[robot_name][group_name]]
                    polynomial_weights_MX = ca.MX.sym(
                        "pol_weights",
                        fk_functions[robot_name][group_name].numel_out() + 1,
                        num_obstacle_categories,
                    )
                    # slack_MX = ca.MX.sym(f'slack_{robot_name}_{group_name}',1,1)
                    mx_out = group_score_function_repsum.call(mx_in)
                    if robot_name == "robot_1":
                        configuration_1_MX = mx_in[0][
                            q_obj.numel() : q_obj.numel() + q_robot_1.numel()
                        ]
                    else:
                        configuration_1_MX = mx_in[0][
                            q_obj.numel()
                            + q_robot_1.numel() : q_obj.numel()
                            + q_robot_1.numel()
                            + q_robot_2.numel()
                        ]
            #         x[
            #     q_obj.numel()
            #     + q_robot_1.numel() : q_obj.numel()
            #     + q_robot_1.numel()
            #     + q_robot_2.numel()
            # ]
                    fk_q_MX = fk_functions[robot_name][group_name](configuration_1_MX)

                    score_pol = ca.MX(total_number_of_groups + g_grasping.numel(), 1)
                    score_pol[group_i] += ca.dot(
                        polynomial_weights_MX,
                        ca.veccat(ca.DM(1), fk_q_MX),
                    ) + slack_MX
                    mx_out[0] += score_pol 
                    mx_in += [polynomial_weights_MX]
                    group_score_function_repsum = ca.Function(
                        "g_score",
                        mx_in,
                        mx_out,
                        [
                            "x",
                            "fk_S" + robot_name + "_" + group_name,
                            "A" + robot_name + "_" + group_name,
                            "pol_A" + robot_name + "_" + group_name,
                        ],
                        ["g"],
                        {
                            "is_diff_in": group_score_function.is_diff_in() + [False],
                            "always_inline": True,
                        },
                    )
                    score_functions[robot_name + "_" + group_name] = (
                        group_score_function_repsum
                    )

                    score_jacobian_function = (
                        group_score_function.jacobian()
                        .map(
                            group_options["inner_map_size"],
                            [
                                True,
                                False,
                                False,
                            ]
                            + [True] * (group_score_function.jacobian().n_in() - 3),
                            [True]
                            + [False] * (group_score_function.jacobian().n_out() - 1),
                        )
                        .map(
                            "jac_score_" + group_options["jacobian_parallelization"],
                            group_options["jacobian_parallelization"],
                            group_options["outer_map_size"],
                            [
                                0,
                            ],
                            [0],
                        )
                    )

                    mx_in = score_jacobian_function.mx_in()
                    slack_MX = mx_in[0][x_slack_index_dict[robot_name][group_name]]
                    mx_out = score_jacobian_function.call(mx_in)
                    mx_in += [polynomial_weights_MX]
                    # configuration_MX = mx_in[score_jacobian_function.index_in("x")][
                    #     q_obj.numel() : q_obj.numel() + q_robot_1.numel()
                    # ]
                    if robot_name == "robot_1":
                        configuration_MX = mx_in[score_jacobian_function.index_in("x")][
                            q_obj.numel() : q_obj.numel() + q_robot_1.numel()
                        ]
                    else:
                        configuration_MX = mx_in[score_jacobian_function.index_in("x")][
                            q_obj.numel()
                            + q_robot_1.numel() : q_obj.numel()
                            + q_robot_1.numel()
                            + q_robot_2.numel()
                        ]
                    fk_q_MX = fk_functions[robot_name][group_name](configuration_MX)

                    score_pol = ca.MX(total_number_of_groups + g_grasping.numel(), 1)
                    score_pol[group_i] += ca.dot(
                        polynomial_weights_MX,
                        ca.veccat(ca.DM(1), fk_q_MX),
                    ) + slack_MX
                    jac_score_pol = ca.jacobian(
                        score_pol, mx_in[score_jacobian_function.index_in("x")]
                    )

                    mx_out[
                        score_jacobian_function.index_out("jac_g_x")
                    ] += jac_score_pol

                    score_jacobian_function = ca.Function(
                        score_jacobian_function.name(),
                        mx_in,
                        mx_out,
                        score_jacobian_function.name_in()
                        + ["pol_A" + robot_name + "_" + group_name],
                        score_jacobian_function.name_out(),
                        {"always_inline": True},
                    )
                    score_jacobian_functions[robot_name + "_" + group_name] = (
                        score_jacobian_function
                    )
                    l_score = lam_g_score[group_i] * score

                    score_l_hessian_function = (
                        ca.Function(
                            "score_l_hess_" + robot_name + "_" + group_name,
                            [
                                x,
                                fk_S_block,
                                W_block,
                                ca.SX.sym(
                                    "pol_weights",
                                    fk_functions[robot_name][group_name].numel_out()
                                    + 1,
                                    num_obstacle_categories,
                                ),
                                lam_g,
                            ],
                            [ca.cse(ca.hessian(l_score, x)[0])],
                            [
                                "x",
                                "fk_S" + robot_name + "_" + group_name,
                                "A" + robot_name + "_" + group_name,
                                "pol_A" + robot_name + "_" + group_name,
                                "lam_g",
                            ],
                            ["g_hess"],
                        )
                        .map(
                            group_options["inner_map_size"],
                            [True, False, False, True, True],
                            [True],
                        )
                        .map(
                            "score_l_hess_" + group_options["hessian_parallelization"],
                            group_options["hessian_parallelization"],
                            group_options["outer_map_size"],
                            [0, 3, 4],
                            [0],
                        )
                    )

                    mx_in = score_l_hessian_function.mx_in()
                    slack_MX = mx_in[0][x_slack_index_dict[robot_name][group_name]]
                    mx_out = score_l_hessian_function.call(mx_in)
                    pol_W = mx_in[
                        score_l_hessian_function.index_in(
                            "pol_A" + robot_name + "_" + group_name
                        )
                    ]

                    # configuration_MX = mx_in[score_l_hessian_function.index_in("x")][
                    #     q_obj.numel() : q_obj.numel() + q_robot_1.numel()
                    # ]
                    if robot_name == "robot_1":
                        configuration_MX = mx_in[score_l_hessian_function.index_in("x")][
                            q_obj.numel() : q_obj.numel() + q_robot_1.numel()
                        ]
                    else:
                        configuration_MX = mx_in[score_l_hessian_function.index_in("x")][
                            q_obj.numel()
                            + q_robot_1.numel() : q_obj.numel()
                            + q_robot_1.numel()
                            + q_robot_2.numel()
                        ]
                    fk_q_MX = fk_functions[robot_name][group_name](configuration_MX)

                    score_pol = ca.MX(total_number_of_groups + g_grasping.numel(), 1)
                    score_pol[group_i] += ca.dot(
                        pol_W,
                        ca.veccat(ca.DM(1), fk_q_MX),
                    ) + slack_MX
                    hess_score_pol = ca.hessian(
                        ca.dot(
                            score_pol,
                            mx_in[score_l_hessian_function.index_in("lam_g")],
                        ),
                        mx_in[score_l_hessian_function.index_in("x")],
                    )[0]

                    score_hess = score_l_hessian_function.call(mx_in)[0] + (
                        hess_score_pol
                    )
                    score_l_hessian_function = ca.Function(
                        "sample_l_hess",
                        mx_in,
                        [score_hess],
                        score_l_hessian_function.name_in(),
                        ["g_hess"],
                        {
                            "cse": True,
                            "always_inline": True,
                        },
                    )
                    score_hessian_functions[robot_name + "_" + group_name] = (
                        score_l_hessian_function
                    )
                    group_i += 1
                    # Score + grasping constraint

            x_MX = ca.MX.sym("q", *x.shape)
            p_MX = ca.MX.sym("p", *p.shape)
            lam_g_MX = ca.MX.sym("lam_g", *lam_g.shape)
            variables_in = {"x": x_MX, "p": p_MX}
            g = g_grasping_function.call({"x": x_MX, "p": p_MX})["g"]
            lbgs = {}
            ubgs = {}
            self.score_functions = score_functions
            for name, score_function in score_functions.items():
                lbgs[name + "_lbg"] = ca.MX.sym("lb_" + name, 1)
                ubgs[name + "_ubg"] = ca.MX.sym("ub_" + name, 1)
                mx_in = score_function.convert_in(score_function.mx_in())
                mx_in["x"] = x_MX
                mx_in["p"] = p_MX
                mx_out = score_function.call(
                    {k: mx_in[k] for k in score_function.name_in()}
                )
                g += mx_out["g"]
                variables_in |= mx_in

            lbg = ca.veccat(*(list(lbgs.values()) + [lbg_grasping]))
            ubg = ca.veccat(*(list(ubgs.values()) + [ubg_grasping]))
            g_grasping_and_score_function = ca.Function(
                "g_score",
                variables_in | {"g": g},
                variables_in.keys(),
                [
                    "g",
                ],
                {
                    "is_diff_in": [True] + [False] * (len(variables_in) - 1),
                    "is_diff_out": [
                        True,
                    ],
                    "always_inline": True,
                    "cse": True,
                },
            )
            lbg_grasping_and_score_function = ca.Function(
                "lbg_grasping_and_score",
                lbgs | {"lbg": lbg},
                lbgs.keys(),
                [
                    "lbg",
                ],
                {
                    "always_inline": True,
                },
            )
            ubg_grasping_and_score_function = ca.Function(
                "ubg_grasping_and_score",
                ubgs | {"ubg": ubg},
                ubgs.keys(),
                [
                    "ubg",
                ],
                {
                    "always_inline": True,
                },
            )
            variables_in = {"x": x_MX, "p": p_MX}
            jac_g = g_grasping_function.jacobian().call({"x": x_MX, "p": p_MX})[
                "jac_g_x"
            ]
            for name, score_jacobian_function in score_jacobian_functions.items():
                mx_in = score_jacobian_function.convert_in(
                    score_jacobian_function.mx_in()
                )
                mx_in["x"] = x_MX
                mx_in["p"] = p_MX
                mx_out = score_jacobian_function.call(
                    {k: mx_in[k] for k in score_jacobian_function.name_in()}
                )
                jac_g += mx_out["jac_g_x"]
                variables_in |= mx_in

            jac_g_grasping_and_score_function = ca.Function(
                "jac_g_score",
                variables_in | {"jac_g_x": jac_g},
                variables_in.keys(),
                ["jac_g_x"],
                {
                    "is_diff_in": [
                        True if in_ == "x" else False for in_ in variables_in
                    ],
                    "always_inline": True,
                },
            )
            variables_in = {"x": x_MX, "p": p_MX, "lam_g": lam_g_MX}
            hess_l = grasping_l_hessian_function.call(variables_in)["g_hess"]
            for name, score_hessian_function in score_hessian_functions.items():
                mx_in = score_hessian_function.convert_in(
                    score_hessian_function.mx_in()
                )
                mx_in["x"] = x_MX
                mx_in["p"] = p_MX
                mx_in["lam_g"] = lam_g_MX
                mx_out = score_hessian_function.call(
                    {k: mx_in[k] for k in score_hessian_function.name_in()}
                )
                hess_l += mx_out["g_hess"]
                variables_in |= mx_in

            hess_l_grasping_and_score_function = ca.Function(
                "jac_g_score",
                variables_in | {"g_hess": hess_l},
                variables_in.keys(),
                ["g_hess"],
                {
                    "is_diff_in": [False for in_ in variables_in],
                    "always_inline": True,
                },
            )

            # replace jacobian in g_grasping_and_score_function with the mapped one
            mx_in = g_grasping_and_score_function.mx_in()
            jac_score_function_map = g_grasping_and_score_function.jacobian()
            jac_mx_in = jac_score_function_map.convert_in(
                jac_score_function_map.mx_in()
            )
            jac_result = {
                name: ca.MX(*jac_score_function_map.size_out(name))
                for name in jac_score_function_map.name_out()
            }
            jac_result["jac_g_x"] = jac_g_grasping_and_score_function.call(
                {
                    name: jac_mx_in[name]
                    for name in jac_g_grasping_and_score_function.name_in()
                }
            )["jac_g_x"]
            jac_f = ca.Function(
                jac_score_function_map.name() + "_replaced",
                jac_mx_in | jac_result,
                jac_score_function_map.name_in(),
                jac_score_function_map.name_out(),
                {
                    "always_inline": True,
                    "jac_penalty": 0,
                    "is_diff_in": [
                        True if in_ == "x" else False
                        for in_ in jac_score_function_map.name_in()
                    ],
                },
            )
            g_grasping_and_score_function = ca.Function(
                g_grasping_and_score_function.name() + "_replaced",
                mx_in,
                g_grasping_and_score_function.call(mx_in),
                g_grasping_and_score_function.name_in(),
                g_grasping_and_score_function.name_out(),
                {
                    "never_inline": True,
                    "custom_jacobian": jac_f,
                    "jac_penalty": 0,
                    "is_diff_in": g_grasping_and_score_function.is_diff_in(),
                    "is_diff_out": g_grasping_and_score_function.is_diff_out(),
                },
            )

            # create lagrangian for the sample constraint and replace its hessian
            lagrangian_sample = get_lagrangian(g_grasping_and_score_function)
            jac_l_function = lagrangian_sample.wrap().jacobian()
            jac_jac_l_function = jac_l_function.jacobian()

            mx_in = jac_jac_l_function.convert_in(jac_jac_l_function.mx_in())
            jac_jac_l_result = {
                name: ca.MX(*jac_jac_l_function.size_out(name))
                for name in jac_jac_l_function.name_out()
            }
            jac_jac_l_result["jac_jac_l_x_x"] = hess_l_grasping_and_score_function.call(
                {
                    name: mx_in[name]
                    for name in hess_l_grasping_and_score_function.name_in()
                }
            )["g_hess"]
            jac_jac_l_function = ca.Function(
                jac_jac_l_function.name(),
                mx_in | jac_jac_l_result,
                jac_jac_l_function.name_in(),
                jac_jac_l_function.name_out(),
            )

            mx_in = jac_l_function.convert_in(jac_l_function.mx_in())
            jac_l_result = {
                name: ca.MX(*jac_l_function.size_out(name))
                for name in jac_l_function.name_out()
            }
            jac_l_function = ca.Function(
                jac_l_function.name(),
                mx_in | jac_l_result,
                jac_l_function.name_in(),
                jac_l_function.name_out(),
                {
                    "custom_jacobian": jac_jac_l_function,
                },
            )
            lagrangian_sample = lagrangian_sample.wrap_as_needed(
                {
                    "is_diff_in": lagrangian_sample.is_diff_in(),
                    "custom_jacobian": jac_l_function,
                }
            )

            return (
                g_grasping_and_score_function,
                lbg_grasping_and_score_function,
                ubg_grasping_and_score_function,
                lagrangian_sample,
            )

            #  (polyharmonic_kernel(fk_vector,fk_support_vectors.view(fk_support_vectors.shape[0],-1))@polyharmonic_weights + constant + torch.dot(pol_weights.squeeze(),fk_vector.squeeze()))
            # polyharmonic_weights, constant, pol_weights = polyharmonic_weights[:N], polyharmonic_weights[N], polyharmonic_weights[N+1:]
            weights_matrix_sym = modul.sym(
                "A", num_obstacle_categories, (num_support_vectors_unmapped) * 2
            )  # gets tranposed later

            weights_matrix_sym_1 = weights_matrix_sym[:, :num_support_vectors_unmapped]
            weights_matrix_sym_2 = weights_matrix_sym[:, num_support_vectors_unmapped:]
            support_vectors_matrix_sym = modul.sym(
                "S", num_support_vectors_unmapped, num_positions
            )

            polyharmonic_kernel_function = make_polyharmonic_kernel_function(
                fk_function_1.numel_out()
            )
            fk_support_vectors_matrix_sym = modul.sym(
                "fk_S", fk_function_1.numel_out(), num_support_vectors_unmapped * 2
            )
            fk_support_vectors_matrix_sym_1 = fk_support_vectors_matrix_sym[
                :, :num_support_vectors_unmapped
            ]
            fk_support_vectors_matrix_sym_2 = fk_support_vectors_matrix_sym[
                :, num_support_vectors_unmapped:
            ]
            fk_q_sample_1 = fk_function_1(configuration_1)
            fk_q_sample_2 = fk_function_2(configuration_2)

            score_1 = polyharmonic_kernel_function.map(
                num_support_vectors_unmapped, [True, False]
            )(fk_q_sample_1, fk_support_vectors_matrix_sym_1) @ (weights_matrix_sym_1.T)
            # score_1_pol = ca.dot(polynomial_weights.T,ca.veccat(ca.DM(1),*fk_q_sample_1))
            score_2 = polyharmonic_kernel_function.map(
                num_support_vectors_unmapped, [True, False]
            )(fk_q_sample_2, fk_support_vectors_matrix_sym_2) @ (weights_matrix_sym_2.T)
            # score_2_pol = ca.dot(polynomial_weights.T,ca.veccat(ca.DM(1),*fk_q_sample_2))
            score = ca.vertcat(score_1, score_2)

            g_score_fake = ca.veccat(score, ca.SX(*g_grasping.shape))
            g_score_fake_function = ca.Function(
                "g_score",
                [x, fk_support_vectors_matrix_sym, weights_matrix_sym],
                [ca.cse(g_score_fake)],
                [
                    "x",
                    "fk_S",
                    "A",
                ],
                ["g"],
                {
                    "is_diff_in": [
                        True,
                        False,
                        False,
                    ]
                },
            )
            g_score_fake_function_repsum = g_score_fake_function.map(
                map_size_1 * map_size_2,
                [
                    True,
                    False,
                    False,
                ],
                [True],
            )

            mx_in = g_score_fake_function_repsum.mx_in()
            polynomial_weights_MX = ca.MX.sym(
                "pol_weights",
                fk_function_1.numel_out() * 2 + 2,
                num_obstacle_categories,
            )
            mx_out = g_score_fake_function_repsum.call(mx_in)
            configuration_1_MX = mx_in[0][
                q_obj.numel() : q_obj.numel() + q_robot_1.numel()
            ]
            configuration_2_MX = mx_in[0][
                q_obj.numel()
                + q_robot_1.numel() : q_obj.numel()
                + q_robot_1.numel()
                + q_robot_2.numel()
            ]
            fk_q_sample_1_MX = fk_function_1(configuration_1_MX)
            fk_q_sample_2_MX = fk_function_2(configuration_2_MX)
            score_pol = ca.vertcat(
                ca.dot(
                    polynomial_weights_MX[: polynomial_weights_MX.shape[0] // 2],
                    ca.veccat(ca.DM(1), fk_q_sample_1_MX),
                ),
                ca.dot(
                    polynomial_weights_MX[polynomial_weights_MX.shape[0] // 2 :],
                    ca.veccat(ca.DM(1), fk_q_sample_2_MX),
                ),
                ca.MX(*g_grasping.shape),
            )
            mx_out[0] += score_pol
            mx_in += [polynomial_weights_MX]
            g_score_fake_function_repsum = ca.Function(
                "g_score",
                mx_in,
                mx_out,
                ["x", "fk_S", "A", "pol_A"],
                ["g"],
                {
                    "is_diff_in": g_score_fake_function.is_diff_in() + [False],
                    "always_inline": True,
                },
            )
            # g_score_fake_function_repsum.name = lambda: 'g_score'

            self.g_score = g_score_fake_function_repsum

            score_jacobian_function = (
                g_score_fake_function.jacobian()
                .map(
                    map_size_1,
                    [
                        True,
                        False,
                        False,
                    ]
                    + [True] * (g_score_fake_function.jacobian().n_in() - 3),
                    [True] + [False] * (g_score_fake_function.jacobian().n_out() - 1),
                )
                .map(
                    "jac_score_" + jacobian_parallelization,
                    jacobian_parallelization,
                    map_size_2,
                    [
                        0,
                    ],
                    [0],
                )
            )
            mx_in = score_jacobian_function.mx_in()
            mx_out = score_jacobian_function.call(mx_in)
            mx_in += [polynomial_weights_MX]
            configuration_1_MX_ = mx_in[score_jacobian_function.index_in("x")][
                q_obj.numel() : q_obj.numel() + q_robot_1.numel()
            ]
            configuration_2_MX_ = mx_in[score_jacobian_function.index_in("x")][
                q_obj.numel()
                + q_robot_1.numel() : q_obj.numel()
                + q_robot_1.numel()
                + q_robot_2.numel()
            ]
            fk_q_sample_1_MX_ = fk_function_1(configuration_1_MX_)
            fk_q_sample_2_MX_ = fk_function_2(configuration_2_MX_)
            score_pol_ = ca.vertcat(
                ca.dot(
                    polynomial_weights_MX[: polynomial_weights_MX.shape[0] // 2],
                    ca.veccat(ca.DM(1), fk_q_sample_1_MX_),
                ),
                ca.dot(
                    polynomial_weights_MX[polynomial_weights_MX.shape[0] // 2 :],
                    ca.veccat(ca.DM(1), fk_q_sample_2_MX_),
                ),
                ca.MX(*g_grasping.shape),
            )
            jac_score_pol = ca.jacobian(
                score_pol_, mx_in[score_jacobian_function.index_in("x")]
            )

            mx_out[score_jacobian_function.index_out("jac_g_x")] += jac_score_pol

            score_jacobian_function = ca.Function(
                score_jacobian_function.name(),
                mx_in,
                mx_out,
                score_jacobian_function.name_in() + ["pol_A"],
                score_jacobian_function.name_out(),
                {"always_inline": True},
            )

            g_grasping_fake = ca.veccat(ca.SX(*score.shape), g_grasping)
            g_grasping_fake_function = ca.Function(
                "sample_constraint_grasping_only",
                [x, p],
                [ca.cse(g_grasping_fake)],
                [
                    "x",
                    "p",
                ],
                ["g"],
                {
                    "is_diff_in": [
                        True,
                        False,
                    ]
                },
            )
            lam_g = ca.SX.sym("lam_g", score.numel() + g_grasping.numel(), 1)
            l_score = ca.dot(lam_g[: score.numel()], score)
            l_grasping = ca.dot(lam_g[score.numel() :], g_grasping)

            score_l_hessian_function = (
                ca.Function(
                    "score_l_hess",
                    [
                        x,
                        fk_support_vectors_matrix_sym,
                        weights_matrix_sym,
                        ca.SX.sym(
                            "pol_weights",
                            fk_function_1.numel_out() * 2 + 2,
                            num_obstacle_categories,
                        ),
                        lam_g,
                    ],
                    [ca.cse(ca.hessian(l_score, x)[0])],
                    ["x", "fk_S", "A", "pol_A", "lam_g"],
                    ["g_hess"],
                )
                .map(map_size_1, [True, False, False, True, True], [True])
                .map(
                    "score_l_hess_" + hessian_parallelization,
                    hessian_parallelization,
                    map_size_2,
                    [0, 3, 4],
                    [0],
                )
            )

            grasping_l_hessian_function = ca.Function(
                "grasping_l_hess",
                [x, p, lam_g],
                [ca.cse(ca.hessian(l_grasping, x)[0])],
                ["x", "p", "lam_g"],
                ["g_hess"],
            )

            # Score + grasping constraint
            fk_support_vectors_matrix_sym_map = ca.MX.sym(
                "fk_S",
                fk_function_1.numel_out(),
                num_support_vectors_unmapped * map_size_2 * map_size_1 * 2,
            )
            weights_matrix_sym_map = ca.MX.sym(
                "A",
                num_obstacle_categories,
                num_support_vectors_unmapped * map_size_2 * map_size_1 * 2,
            )
            # polynomial_weights_MX = ca.MX.sym('pol_A',fk_function_1.numel_out()*2 + 2,num_obstacle_categories)
            x_MX = ca.MX.sym("q", *x.shape)
            p_MX = ca.MX.sym("p", *p.shape)
            lam_g_MX = ca.MX.sym("lam_g", *lam_g.shape)
            lb_score = ca.MX.sym("lbg_score", 1)
            ub_score = ca.MX.sym("ubg_score", 1)
            lb_score_grasping_constraint = ca.veccat(lb_score, lb_score, lbg_grasping)
            ub_score_grasping_constraint = ca.veccat(ub_score, ub_score, ubg_grasping)

            g_grasping = g_grasping_fake_function.call([x_MX, p_MX])[0]
            g_score = g_score_fake_function_repsum.call(
                [
                    x_MX,
                    fk_support_vectors_matrix_sym_map,
                    weights_matrix_sym_map,
                    polynomial_weights_MX,
                ]
            )[0]
            g_grasping_and_score_function = ca.Function(
                "g_score",
                [
                    x_MX,
                    p_MX,
                    fk_support_vectors_matrix_sym_map,
                    weights_matrix_sym_map,
                    polynomial_weights_MX,
                ],
                [
                    g_grasping + g_score,
                ],
                ["x", "p", "fk_S", "A", "pol_A"],
                [
                    "g",
                ],
                {
                    "is_diff_in": [True, False, False, False, False],
                    "is_diff_out": [
                        True,
                    ],
                    "always_inline": True,
                    "cse": True,
                },
            )

            configuration_1_MX_ = x_MX[
                q_obj.numel() : q_obj.numel() + q_robot_1.numel()
            ]
            configuration_2_MX_ = x_MX[
                q_obj.numel()
                + q_robot_1.numel() : q_obj.numel()
                + q_robot_1.numel()
                + q_robot_2.numel()
            ]
            fk_q_sample_1_MX_ = fk_function_1(configuration_1_MX_)
            fk_q_sample_2_MX_ = fk_function_2(configuration_2_MX_)
            score_pol_ = ca.vertcat(
                ca.dot(
                    polynomial_weights_MX[: polynomial_weights_MX.shape[0] // 2],
                    ca.veccat(ca.DM(1), fk_q_sample_1_MX_),
                ),
                ca.dot(
                    polynomial_weights_MX[polynomial_weights_MX.shape[0] // 2 :],
                    ca.veccat(ca.DM(1), fk_q_sample_2_MX_),
                ),
                ca.MX(*g_grasping.shape),
            )
            hess_score_pol = ca.hessian(ca.dot(lam_g_MX[:2], score_pol_[:2]), x_MX)[0]
            score_hess = (
                score_l_hessian_function.call(
                    [
                        x_MX,
                        fk_support_vectors_matrix_sym_map,
                        weights_matrix_sym_map,
                        polynomial_weights_MX,
                        lam_g_MX,
                    ]
                )[0]
                + hess_score_pol
            )

            grasping_hess = grasping_l_hessian_function.call([x_MX, p_MX, lam_g_MX])[0]
            sample_l_hess_function = ca.Function(
                "sample_l_hess",
                [
                    x_MX,
                    p_MX,
                    fk_support_vectors_matrix_sym_map,
                    weights_matrix_sym_map,
                    polynomial_weights_MX,
                    lam_g_MX,
                ],
                [ca.cse(score_hess + grasping_hess)],
                ["x", "p", "fk_S", "A", "pol_A", "lam_g"],
                ["g_hess"],
                {
                    "always_inline": True,
                },
            )

            jacobian_grasping = g_grasping_fake_function.jacobian().call(
                {"x": x_MX, "p": p_MX}
            )
            jacobian_score = score_jacobian_function.call(
                {
                    "x": x_MX,
                    "fk_S": fk_support_vectors_matrix_sym_map,
                    "A": weights_matrix_sym_map,
                    "pol_A": polynomial_weights_MX,
                }
            )

            jac_g_grasping_and_score_function = ca.Function(
                "jac_g_score",
                [
                    x_MX,
                    p_MX,
                    fk_support_vectors_matrix_sym_map,
                    weights_matrix_sym_map,
                    polynomial_weights_MX,
                ],
                [ca.cse(jacobian_score["jac_g_x"] + jacobian_grasping["jac_g_x"])],
                ["x", "p", "fk_S", "A", "pol_A"],
                ["jac_g_x"],
                {
                    "is_diff_in": g_grasping_and_score_function.is_diff_in(),
                    "always_inline": True,
                },
            )

            # replace jacobian in g_grasping_and_score_function with the mapped one
            mx_in = g_grasping_and_score_function.mx_in()
            jac_score_function_map = g_grasping_and_score_function.jacobian()
            jac_mx_in = jac_score_function_map.convert_in(
                jac_score_function_map.mx_in()
            )
            jac_result = {
                name: ca.MX(*jac_score_function_map.size_out(name))
                for name in jac_score_function_map.name_out()
            }
            jac_result["jac_g_x"] = jac_g_grasping_and_score_function.call(
                {
                    name: jac_mx_in[name]
                    for name in jac_g_grasping_and_score_function.name_in()
                }
            )["jac_g_x"]
            jac_f = ca.Function(
                jac_score_function_map.name() + "_replaced",
                jac_mx_in | jac_result,
                jac_score_function_map.name_in(),
                jac_score_function_map.name_out(),
                {
                    "always_inline": True,
                    "jac_penalty": 0,
                    "is_diff_in": [True, False, False, False, False, False],
                },
            )
            g_grasping_and_score_function = ca.Function(
                g_grasping_and_score_function.name() + "_replaced",
                mx_in,
                g_grasping_and_score_function.call(mx_in),
                g_grasping_and_score_function.name_in(),
                g_grasping_and_score_function.name_out(),
                {
                    "never_inline": True,
                    "custom_jacobian": jac_f,
                    "jac_penalty": 0,
                    "is_diff_in": g_grasping_and_score_function.is_diff_in(),
                    "is_diff_out": g_grasping_and_score_function.is_diff_out(),
                },
            )

            lbg_grasping_and_score_function = ca.Function(
                "lbg_grasping_and_score",
                [lb_score],
                [
                    lb_score_grasping_constraint,
                ],
                [
                    "score_lbg",
                ],
                [
                    "lbg",
                ],
                {
                    "always_inline": True,
                },
            )
            ubg_grasping_and_score_function = ca.Function(
                "ubg_grasping_and_score",
                [ub_score],
                [
                    ub_score_grasping_constraint,
                ],
                ["score_ubg"],
                [
                    "ubg",
                ],
                {
                    "always_inline": True,
                },
            )

            # create lagrangian for the sample constraint and replace its hessian
            lagrangian_sample = get_lagrangian(g_grasping_and_score_function)
            jac_l_function = lagrangian_sample.wrap().jacobian()
            jac_jac_l_function = jac_l_function.jacobian()

            mx_in = jac_jac_l_function.convert_in(jac_jac_l_function.mx_in())
            jac_jac_l_result = {
                name: ca.MX(*jac_jac_l_function.size_out(name))
                for name in jac_jac_l_function.name_out()
            }
            jac_jac_l_result["jac_jac_l_x_x"] = sample_l_hess_function.call(
                {name: mx_in[name] for name in sample_l_hess_function.name_in()}
            )["g_hess"]
            jac_jac_l_function = ca.Function(
                jac_jac_l_function.name(),
                mx_in | jac_jac_l_result,
                jac_jac_l_function.name_in(),
                jac_jac_l_function.name_out(),
            )

            mx_in = jac_l_function.convert_in(jac_l_function.mx_in())
            jac_l_result = {
                name: ca.MX(*jac_l_function.size_out(name))
                for name in jac_l_function.name_out()
            }
            jac_l_function = ca.Function(
                jac_l_function.name(),
                mx_in | jac_l_result,
                jac_l_function.name_in(),
                jac_l_function.name_out(),
                {
                    "custom_jacobian": jac_jac_l_function,
                },
            )
            lagrangian_sample = lagrangian_sample.wrap_as_needed(
                {
                    "is_diff_in": lagrangian_sample.is_diff_in(),
                    "custom_jacobian": jac_l_function,
                }
            )

            # create function that properly maps the support vectors and their weights to the inputs of the NLP
            support_vectors_matrix_sym = ca.MX.sym(
                "S",
                num_support_vectors_unmapped * map_size_1 * map_size_2,
                num_positions,
            )
            fk_support_vectors_function_1 = ca.Function(
                "fk_support_vectors",
                [support_vectors_matrix_sym],
                [
                    fk_function_1.map(support_vectors_matrix_sym.shape[0], "openmp", 4)(
                        support_vectors_matrix_sym.T
                    )
                ],
                {"cse": True},
            )
            fk_support_vectors_function_2 = ca.Function(
                "fk_support_vectors",
                [support_vectors_matrix_sym],
                [
                    fk_function_2.map(support_vectors_matrix_sym.shape[0], "openmp", 4)(
                        support_vectors_matrix_sym.T
                    )
                ],
                {"cse": True},
            )
            weights_matrix_sym_map_1 = ca.MX.sym(
                "A1",
                num_obstacle_categories,
                num_support_vectors_unmapped * map_size_1 * map_size_2,
            )
            weights_matrix_sym_map_2 = ca.MX.sym(
                "A2",
                num_obstacle_categories,
                num_support_vectors_unmapped * map_size_1 * map_size_2,
            )
            support_vectors_matrix_sym_1 = ca.MX.sym(
                "S1",
                num_support_vectors_unmapped * map_size_1 * map_size_2,
                num_positions,
            )
            support_vectors_matrix_sym_2 = ca.MX.sym(
                "S2",
                num_support_vectors_unmapped * map_size_1 * map_size_2,
                num_positions,
            )

            fk_sv_1 = ca.horzsplit_n(
                fk_support_vectors_function_1(support_vectors_matrix_sym_1),
                map_size_2 * map_size_1,
            )
            fk_sv_2 = ca.horzsplit_n(
                fk_support_vectors_function_2(support_vectors_matrix_sym_2),
                map_size_2 * map_size_1,
            )
            # intercalate both
            fk_total = [part for pair in zip(fk_sv_1, fk_sv_2) for part in pair]

            w_1 = ca.horzsplit_n(weights_matrix_sym_map_1, map_size_2 * map_size_1)
            w_2 = ca.horzsplit_n(weights_matrix_sym_map_2, map_size_2 * map_size_1)
            weights_total = [part for pair in zip(w_1, w_2) for part in pair]

            vectors_and_weights = ca.Function(
                "support_vectors_and_weights_collision",
                [
                    support_vectors_matrix_sym_1,
                    support_vectors_matrix_sym_2,
                    weights_matrix_sym_map_1,
                    weights_matrix_sym_map_2,
                ],
                [ca.horzcat(*fk_total), ca.horzcat(*weights_total)],
            )
            # self.support_vectors_and_weights = (vectors_and_weights)
            setattr(self, vectors_and_weights.name(), vectors_and_weights)

            fk_support_vectors_matrix_sym_1 = ca.MX.sym(
                "fk_S1", *fk_support_vectors_function_1.size_out(0)
            )
            fk_support_vectors_matrix_sym_2 = ca.MX.sym(
                "fk_S2", *fk_support_vectors_function_2.size_out(0)
            )
            fk_sv_1 = ca.horzsplit_n(
                fk_support_vectors_matrix_sym_1, map_size_2 * map_size_1
            )
            fk_sv_2 = ca.horzsplit_n(
                fk_support_vectors_matrix_sym_2, map_size_2 * map_size_1
            )
            # intercalate both
            fk_total = [part for pair in zip(fk_sv_1, fk_sv_2) for part in pair]
            w_1 = ca.horzsplit_n(weights_matrix_sym_map_1, map_size_2 * map_size_1)
            w_2 = ca.horzsplit_n(weights_matrix_sym_map_2, map_size_2 * map_size_1)
            weights_total = [part for pair in zip(w_1, w_2) for part in pair]
            fk_support_vectors_and_weights = ca.Function(
                "fk_support_vectors_and_weights_collision",
                [
                    fk_support_vectors_matrix_sym_1,
                    fk_support_vectors_matrix_sym_2,
                    weights_matrix_sym_map_1,
                    weights_matrix_sym_map_2,
                ],
                [ca.horzcat(*fk_total), ca.horzcat(*weights_total)],
            )
            # self.fk_support_vectors_and_weights = (fk_support_vectors_and_weights).
            setattr(
                self,
                fk_support_vectors_and_weights.name(),
                fk_support_vectors_and_weights,
            )
            return (
                g_grasping_and_score_function,
                lbg_grasping_and_score_function,
                ubg_grasping_and_score_function,
                lagrangian_sample,
            )

    def make_gaze_function(
        self,
    ):
        modul = ca.SX
        robot_name = "robot_1"
        group_name = "group_1"
        options = self.gaze_options['robots'][0]['groups'][group_name]
        
        # self.vision_weights[robot_name][group_name] = {}
        # self.vision_support_vectors[robot_name][group_name] = {}
        # self.vision_polynomial_weights[robot_name][group_name] = {}
        # self.gaze_options
        num_positions = 9
        # self.gaze_options['robots']: Dict[str, Robot]
        num_obstacle_categories = 1
        map_size_2 = options['outer_map_size']
        num_support_vectors_unmapped = options['num_support_points_per_block']
        map_size_1 = options['inner_map_size']
        hessian_parallelization = options['hessian_parallelization']
        jacobian_parallelization = options['jacobian_parallelization']

        x = ca.SX.sym("x", num_positions)
        cost_weight = ca.SX.sym("cost_weight", 1)
        reshape_as_pytorch = True
        if True:
            if reshape_as_pytorch:
                fk_function_1 = ca.Function(
                    "fk",
                    [x],
                    [self.robots[0].vision_model.forward_kinematics_groups_casadi[group_name](x).T.reshape((-1, 1))],
                    {"cse": True},
                )
            else:
                fk_function_1 = ca.Function(
                    "fk",
                    [x],
                    [self.robots[0].vision_model.forward_kinematics_groups_casadi[group_name](x).reshape((-1, 1))],
                    {"cse": True},
                )
        else:
            assert False, "Not implemented"
            fk_function = ca.Function(
                "fk",
                [configuration],
                [configuration],
                {"cse": True, "always_inline": True},
            )
        fk_function = self.robots[0].vision_model.forward_kinematics_groups_casadi[group_name]
        fk_functions = {robot_name:{}}
        q = x
        fk_functions[robot_name][group_name] = ca.Function(
                        "fk",
                        [q],
                        [fk_function(q).T.reshape((-1, 1))],
                        {"cse": True},
                    )
        
        # cost_weight = ca.SX('cost_weight')
        fk_q = fk_functions[robot_name][group_name].call([q])[0]
        group_options: DiffCoGroupOptions = self.gaze_options['robots'][0]['groups'][group_name]
        num_obstacle_categories = 1
        num_support_vectors_block = group_options[
            "num_support_points_per_block"
        ]
        W_block = modul.sym(
            "A", num_obstacle_categories, num_support_vectors_block
        )
        fk_S_block = modul.sym(
            "fk_S",
            fk_functions[robot_name][group_name].numel_out(),
            num_support_vectors_block,
        )
        polyharmonic_kernel_function = make_polyharmonic_kernel_function(
            fk_functions[robot_name][group_name].numel_out()
        )
        score = polyharmonic_kernel_function.map(
            num_support_vectors_block, [True, False]
        )(fk_q, fk_S_block) @ (W_block.T)
        group_score_function = ca.Function(
            "g_score_" + robot_name + "_" + group_name,
            [x, fk_S_block, W_block],
            [ca.cse(score)],
            [
                "x",
                "fk_S" + robot_name + "_" + group_name,
                "A" + robot_name + "_" + group_name,
            ],
            ["g"],
            {
                "is_diff_in": [
                    True,
                    False,
                    False,
                ]
            },
        )
        group_score_function_repsum = group_score_function.map(
            group_options["inner_map_size"]
            * group_options["outer_map_size"],
            [
                True,
                False,
                False,
            ],
            [True],
        )

        mx_in = group_score_function_repsum.mx_in()
        polynomial_weights_MX = ca.MX.sym(
            "pol_weights",
            fk_functions[robot_name][group_name].numel_out() + 1,
            num_obstacle_categories,
        )
        cost_weight_MX = ca.MX.sym('cost_weight')
        # slack_MX = ca.MX.sym(f'slack_{robot_name}_{group_name}',1,1)
        mx_out = group_score_function_repsum.call(mx_in)
        configuration_1_MX = mx_in[0]
        fk_q_MX = fk_functions[robot_name][group_name](configuration_1_MX)

        score_pol = ca.dot(
            polynomial_weights_MX,
            ca.veccat(ca.DM(1), fk_q_MX),
        )
        mx_out[0] += score_pol 
        mx_out[0] *= cost_weight_MX
        mx_in += [polynomial_weights_MX]
        mx_in += [cost_weight_MX]
        group_score_function_repsum = ca.Function(
            "vision_score",
            mx_in,
            mx_out,
            [
                "x",
                "fk_S" + robot_name + "_" + group_name,
                "A" + robot_name + "_" + group_name,
                "pol_A" + robot_name + "_" + group_name,
                "cost_weight",
            ],
            ["g"],
            {
                "is_diff_in": group_score_function.is_diff_in() + [False,False],
                "always_inline": True,
            },
        )
        score_jacobian_function = (
            group_score_function.jacobian()
            .map(
                group_options["inner_map_size"],
                [
                    True,
                    False,
                    False,
                ]
                + [True] * (group_score_function.jacobian().n_in() - 3),
                [True]
                + [False] * (group_score_function.jacobian().n_out() - 1),
            )
            .map(
                "jac_score_" + group_options["jacobian_parallelization"],
                group_options["jacobian_parallelization"],
                group_options["outer_map_size"],
                [
                    0,
                ],
                [0],
            )
        )

        mx_in = score_jacobian_function.mx_in()
        mx_out = score_jacobian_function.call(mx_in)
        mx_in += [polynomial_weights_MX]
        mx_in += [cost_weight_MX]
        
        configuration_MX = mx_in[score_jacobian_function.index_in("x")]
        fk_q_MX = fk_functions[robot_name][group_name](configuration_MX)

        score_pol = ca.dot(
            polynomial_weights_MX,
            ca.veccat(ca.DM(1), fk_q_MX),
        )
        jac_score_pol = ca.jacobian(
            score_pol, mx_in[score_jacobian_function.index_in("x")]
        )

        mx_out[
            score_jacobian_function.index_out("jac_g_x")
        ] += jac_score_pol
        mx_out[
            score_jacobian_function.index_out("jac_g_x")
        ] *= cost_weight_MX


        score_jacobian_function = ca.Function(
            score_jacobian_function.name(),
            mx_in,
            mx_out,
            score_jacobian_function.name_in()
            + ["pol_A" + robot_name + "_" + group_name] + ['cost_weight'],
            score_jacobian_function.name_out(),
            {"always_inline": True},
        )
        cost_weight_SX = ca.SX.sym('cost_weight')
        l_score = score

        score_l_hessian_function = (
            ca.Function(
                "score_l_hess_" + robot_name + "_" + group_name,
                [
                    x,
                    fk_S_block,
                    W_block,
                    ca.SX.sym(
                        "pol_weights",
                        fk_functions[robot_name][group_name].numel_out()
                        + 1,
                        num_obstacle_categories,
                    ),
                    cost_weight_SX,
                    ca.SX(1,1),
                ],
                [ca.cse(ca.hessian(l_score, x)[0])],
                [
                    "x",
                    "fk_S" + robot_name + "_" + group_name,
                    "A" + robot_name + "_" + group_name,
                    "pol_A" + robot_name + "_" + group_name,
                    'cost_weight',
                    "lam_g",
                ],
                ["g_hess"],
            )
            .map(
                group_options["inner_map_size"],
                [True, False, False, True, True,True],
                [True],
            ).map(
                "score_l_hess_" + group_options["hessian_parallelization"],
                group_options["hessian_parallelization"],
                group_options["outer_map_size"],
                [0, 3, 4,5],
                [0],
            )
        )
        mx_in = score_l_hessian_function.mx_in()
        
        mx_out = score_l_hessian_function.call(mx_in)
        pol_W = mx_in[
            score_l_hessian_function.index_in(
                "pol_A" + robot_name + "_" + group_name
            )
        ]
        configuration_MX = mx_in[score_l_hessian_function.index_in("x")]
        fk_q_MX = fk_functions[robot_name][group_name](configuration_MX)
        score_pol = ca.dot(
            pol_W,
            ca.veccat(ca.DM(1), fk_q_MX),
        )
        hess_score_pol = ca.hessian(
            ca.dot(
                score_pol,
                ca.MX.ones(score_pol.sparsity()),
            ),
            mx_in[score_l_hessian_function.index_in("x")],
        )[0]
        cost_weight_MX_ = mx_in[
            score_l_hessian_function.index_in(
                "cost_weight"
            )
        ]
        score_hess = cost_weight_MX_*( score_l_hessian_function.call(mx_in)[0] + (
            hess_score_pol
        ))
        score_l_hessian_function = ca.Function(
            "sample_l_hess",
            mx_in,
            [score_hess],
            score_l_hessian_function.name_in(),
            ["g_hess"],
            {
                "cse": True,
                "always_inline": True,
            },
        )
        mx_in = group_score_function_repsum.mx_in()
        jac_score_function_map = group_score_function_repsum.jacobian()
        jac_mx_in = jac_score_function_map.convert_in(
            jac_score_function_map.mx_in()
        )
        jac_result = {
            name: ca.MX(*jac_score_function_map.size_out(name))
            for name in jac_score_function_map.name_out()
        }
        jac_result["jac_g_x"] = score_jacobian_function.call(
            {
                name: jac_mx_in[name]
                for name in score_jacobian_function.name_in()
            }
        )["jac_g_x"]
        jac_f = ca.Function(
            jac_score_function_map.name() + "_replaced",
            jac_mx_in | jac_result,
            jac_score_function_map.name_in(),
            jac_score_function_map.name_out(),
            {
                "always_inline": True,
                "jac_penalty": 0,
                "is_diff_in": [
                    True if in_ == "x" else False
                    for in_ in jac_score_function_map.name_in()
                ],
            },
        )
        group_score_function_repsum = ca.Function(
            group_score_function_repsum.name() + "_replaced",
            mx_in,
            group_score_function_repsum.call(mx_in),
            group_score_function_repsum.name_in(),
            group_score_function_repsum.name_out(),
            {
                "never_inline": True,
                "custom_jacobian": jac_f,
                "jac_penalty": 0,
                "is_diff_in": group_score_function_repsum.is_diff_in(),
                "is_diff_out": group_score_function_repsum.is_diff_out(),
            },
        )
        lagrangian_sample = get_lagrangian(group_score_function_repsum)
        jac_l_function = lagrangian_sample.wrap().jacobian()
        jac_jac_l_function = jac_l_function.jacobian()

        mx_in = jac_jac_l_function.convert_in(jac_jac_l_function.mx_in())
        jac_jac_l_result = {
            name: ca.MX(*jac_jac_l_function.size_out(name))
            for name in jac_jac_l_function.name_out()
        }
        jac_jac_l_result["jac_jac_l_x_x"] = score_l_hessian_function.call(
            {
                name: mx_in[name]
                for name in score_l_hessian_function.name_in()
            }
        )["g_hess"]
        jac_jac_l_function = ca.Function(
            jac_jac_l_function.name(),
            mx_in | jac_jac_l_result,
            jac_jac_l_function.name_in(),
            jac_jac_l_function.name_out(),
        )

        mx_in = jac_l_function.convert_in(jac_l_function.mx_in())
        jac_l_result = {
            name: ca.MX(*jac_l_function.size_out(name))
            for name in jac_l_function.name_out()
        }
        jac_l_function = ca.Function(
            jac_l_function.name(),
            mx_in | jac_l_result,
            jac_l_function.name_in(),
            jac_l_function.name_out(),
            {
                "custom_jacobian": jac_jac_l_function,
            },
        )
        lagrangian_sample = lagrangian_sample.wrap_as_needed(
            {
                "is_diff_in": lagrangian_sample.is_diff_in(),
                "custom_jacobian": jac_l_function,
            }
        )
        score_function = group_score_function_repsum
        return score_function, lagrangian_sample
        weights_matrix_sym = modul.sym(
            "A", num_obstacle_categories, num_support_vectors_unmapped
        )  # gets tranposed later
        weights_matrix_sym_1 = weights_matrix_sym[:, :num_support_vectors_unmapped]
        support_vectors_matrix_sym = modul.sym(
            "S", num_support_vectors_unmapped, num_positions
        )

        polyharmonic_kernel_function = make_polyharmonic_kernel_function(
            fk_function_1.numel_out()
        )
        fk_support_vectors_matrix_sym = modul.sym(
            "fk_S", fk_function_1.numel_out(), num_support_vectors_unmapped
        )
        fk_support_vectors_matrix_sym_1 = fk_support_vectors_matrix_sym[
            :, :num_support_vectors_unmapped
        ]
        fk_q_sample_1 = fk_function_1(x)

        score = (
            cost_weight
            * polyharmonic_kernel_function.map(
                num_support_vectors_unmapped, [True, False]
            )(fk_q_sample_1, fk_support_vectors_matrix_sym_1)
            @ (weights_matrix_sym_1.T)
        )

        score_function = ca.Function(
            "sample_gaze_score",
            [x, fk_support_vectors_matrix_sym, weights_matrix_sym, cost_weight],
            [ca.cse(score)],
            ["x", "fk_S", "A", "cost_weight"],
            ["g"],
            {"is_diff_in": [True, False, False, False]},
        )
        score_function_repsum = score_function.map(
            map_size_1 * map_size_2, [True, False, False, True], [True]
        )

        jac_score_function = (
            score_function.jacobian()
            .map(
                map_size_1,
                [True, False, False, True]
                + [True] * (score_function.jacobian().n_in() - 4),
                [True] + [False] * (score_function.jacobian().n_out() - 1),
            )
            .map(
                "jac_score_" + jacobian_parallelization,
                jacobian_parallelization,
                map_size_2,
                [0, 3],
                [0],
            )
        )

        lam_f = ca.SX.sym("lam_f", 1, 1)
        l_score = lam_f * score

        score_l_hessian_function = (
            ca.Function(
                "score_l_hess",
                [
                    x,
                    fk_support_vectors_matrix_sym,
                    weights_matrix_sym,
                    cost_weight,
                    lam_f,
                ],
                [ca.cse(ca.hessian(l_score, x)[0])],
                ["x", "fk_S", "A", "cost_weight", "lam_g"],
                ["g_hess"],
            )
            .map(map_size_1, [True, False, False, True, True], [True])
            .map(
                "score_l_hess_" + hessian_parallelization,
                hessian_parallelization,
                map_size_2,
                [0, 3, 4],
                [0],
            )
        )

        mx_in = score_function_repsum.mx_in()
        jac_score_function_map = score_function_repsum.jacobian()
        jac_mx_in = jac_score_function_map.convert_in(jac_score_function_map.mx_in())
        jac_result = {
            name: ca.MX(*jac_score_function_map.size_out(name))
            for name in jac_score_function_map.name_out()
        }
        jac_result["jac_g_x"] = jac_score_function.call(
            {name: jac_mx_in[name] for name in jac_score_function.name_in()}
        )["jac_g_x"]
        jac_f = ca.Function(
            "jac_" + score_function_repsum.name() + "_replaced",
            jac_mx_in | jac_result,
            jac_score_function_map.name_in(),
            jac_score_function_map.name_out(),
            {
                "always_inline": True,
                "jac_penalty": 0,
                "is_diff_in": [True, False, False, False, False],
            },
        )
        score_function = ca.Function(
            score_function_repsum.name() + "_replaced",
            mx_in,
            score_function_repsum.call(mx_in),
            score_function_repsum.name_in(),
            score_function_repsum.name_out(),
            {
                "never_inline": True,
                "custom_jacobian": jac_f,
                "jac_penalty": 0,
                "is_diff_in": score_function_repsum.is_diff_in(),
                "is_diff_out": score_function_repsum.is_diff_out(),
            },
        )

        # create lagrangian for the sample constraint and replace its hessian
        lagrangian_sample = get_lagrangian(score_function)
        jac_l_function = lagrangian_sample.wrap().jacobian()
        jac_jac_l_function = jac_l_function.jacobian()

        mx_in = jac_jac_l_function.convert_in(jac_jac_l_function.mx_in())
        jac_jac_l_result = {
            name: ca.MX(*jac_jac_l_function.size_out(name))
            for name in jac_jac_l_function.name_out()
        }
        jac_jac_l_result["jac_jac_l_x_x"] = score_l_hessian_function.call(
            {name: mx_in[name] for name in score_l_hessian_function.name_in()}
        )["g_hess"]
        jac_jac_l_function = ca.Function(
            jac_jac_l_function.name(),
            mx_in | jac_jac_l_result,
            jac_jac_l_function.name_in(),
            jac_jac_l_function.name_out(),
        )

        mx_in = jac_l_function.convert_in(jac_l_function.mx_in())
        jac_l_result = {
            name: ca.MX(*jac_l_function.size_out(name))
            for name in jac_l_function.name_out()
        }
        jac_l_function = ca.Function(
            jac_l_function.name(),
            mx_in | jac_l_result,
            jac_l_function.name_in(),
            jac_l_function.name_out(),
            {
                "custom_jacobian": jac_jac_l_function,
            },
        )
        lagrangian_sample = lagrangian_sample.wrap_as_needed(
            {
                "is_diff_in": lagrangian_sample.is_diff_in(),
                "custom_jacobian": jac_l_function,
            }
        )

        # create function that properly maps the support vectors and their weights to the inputs of the NLP
        support_vectors_matrix_sym = ca.MX.sym(
            "S", num_support_vectors_unmapped * map_size_1 * map_size_2, num_positions
        )
        fk_support_vectors_function_1 = ca.Function(
            "fk_support_vectors",
            [support_vectors_matrix_sym],
            [
                fk_function_1.map(support_vectors_matrix_sym.shape[0], "openmp", 4)(
                    support_vectors_matrix_sym.T
                )
            ],
            {"cse": True},
        )
        weights_matrix_sym_map_1 = ca.MX.sym(
            "A1",
            num_obstacle_categories,
            num_support_vectors_unmapped * map_size_1 * map_size_2,
        )
        support_vectors_matrix_sym_1 = ca.MX.sym(
            "S1", num_support_vectors_unmapped * map_size_1 * map_size_2, num_positions
        )

        vectors_and_weights = ca.Function(
            "support_vectors_and_weights_gaze",
            [support_vectors_matrix_sym_1, weights_matrix_sym_map_1],
            [
                fk_support_vectors_function_1(support_vectors_matrix_sym_1),
                weights_matrix_sym_map_1,
            ],
        )
        setattr(self, vectors_and_weights.name(), (vectors_and_weights))

        # np.allclose(g_grasping_and_score_function.jacobian().call({'x':2,'p':3,'fk_S':4,'A':5})['jac_g_x'],jac_g_grasping_and_score_function.call({'x':2,'p':3,'fk_S':4,'A':5})['jac_g_x'])

        return score_function, lagrangian_sample

    def contraints_on_samples(
        self,
        inner_function,
        outer_function,
        lbg_func,
        ubg_func,
        samples,
        parallelization="serial",
        name = None,
        subnames = None
    ):
        # canon_g, canon_lbg, canon_ubg = get_canon_constraint_function(outer_function).values()
        canon_g = outer_function
        canon_lbg = lbg_func
        canon_ubg = ubg_func
        num_samples = len(samples)
        inner_args = {
            "dec_variables": self.decision_variables,
            "parameters": self.parameters,
        }
        # total_inner = get_inner_grasping_function(self,samples,)
        total_inner = inner_function
        mx_in = total_inner.mx_in()
        mx_out = ca.cse(total_inner.call(mx_in))

        lbg = canon_lbg.convert_out(
            canon_lbg.map(
                num_samples,
            ).call([mx_out[total_inner.index_out(par)] for par in canon_lbg.name_in()])
        )
        ubg = canon_ubg.convert_out(
            canon_ubg.map(
                num_samples,
            ).call([mx_out[total_inner.index_out(par)] for par in canon_ubg.name_in()])
        )
        g = canon_g.convert_out(
            canon_g.map(num_samples, parallelization).call(
                [mx_out[total_inner.index_out(par)] for par in canon_g.name_in()]
            )
        )

        g_total = ca.Function(
            outer_function.name(),
            total_inner.convert_in(mx_in) | g,
            total_inner.name_in(),
            ["g"],
            {"always_inline": True},
        )
        lbg_total = ca.Function(
            outer_function.name() + "_lbg",
            {"parameters": mx_in[1]} | lbg,
            total_inner.name_in(),
            ["lbg"],
            {"always_inline": True},
        ).expand()
        ubg_total = ca.Function(
            outer_function.name() + "_ubg",
            {"parameters": mx_in[1]} | ubg,
            total_inner.name_in(),
            ["ubg"],
            {"always_inline": True},
        ).expand()

        g_v = g_total.call(inner_args)["g"]
        lbg_v = lbg_total.call({"parameters": inner_args["parameters"]})["lbg"]
        ubg_v = ubg_total.call({"parameters": inner_args["parameters"]})["ubg"]
        constraint = self.opti.bounded(ca.vec(lbg_v), ca.vec(g_v), ca.vec(ubg_v))
        if name is None:
            name = outer_function.name() + "_constraint"

        # self.opti.subject_to(constraint)
        self.subject_to(constraint = constraint, name = name, subnames = subnames)
        return {"g": g_total, "canon_outer_g": canon_g, "constraint": constraint}

    def cost_on_samples(
        self, inner_function, outer_function, samples, parallelization="serial"
    ):
        # canon_g, canon_lbg, canon_ubg = get_canon_constraint_function(outer_function).values()
        canon_g = outer_function
        num_samples = len(samples)
        inner_args = {
            "dec_variables": self.decision_variables,
            "parameters": self.parameters,
        }
        # total_inner = get_inner_grasping_function(self,samples,)
        total_inner = inner_function
        mx_in = total_inner.mx_in()
        mx_out = ca.cse(total_inner.call(mx_in))

        f = outer_function.convert_out(
            outer_function.map(num_samples, parallelization).call(
                [mx_out[total_inner.index_out(par)] for par in outer_function.name_in()]
            )
        )

        f_total = ca.Function(
            "g_total",
            total_inner.convert_in(mx_in) | f,
            total_inner.name_in(),
            ["g"],
            {"always_inline": True},
        )

        f_v = f_total.call(inner_args)["g"]
        return {
            "f": f_v,
        }

    def pose_constraints(self, parallelization="serial"):
        get_object_pose_constraint_function = self.get_object_pose_constraint_function_2
        inner = self.get_inner_pose_function_2
        g_func, lbg_func, ubg_func = get_object_pose_constraint_function().values()
        pose_constraint_func = replace_hessian(g_func)
        samples = [0, 1]
        name = "object_pose_constraint"
        subnames = [
            "start_object_pose_translation_x",
            "start_object_pose_translation_y",
            "start_object_pose_translation_z",
            "start_object_pose_orientation",
            "end_object_pose_translation_x",
            "end_object_pose_translation_y",
            "end_object_pose_translation_z",
            "end_object_pose_orientation",
        ]
        result = self.contraints_on_samples(
            inner(
                samples,
            ),
            pose_constraint_func,
            lbg_func,
            ubg_func,
            samples,parallelization,
            name,
            subnames
        )
        g, canon_outer_g, constraint = (
            result["g"],
            result["canon_outer_g"],
            result["constraint"],
        )

        lagrangian = get_lagrangian(canon_outer_g)

        inner_functions_lagrangian = []
        inner_functions = []
        for s in samples:
            inner_functions_lagrangian.append(
                inner([s], lam_g_shape=(lagrangian.numel_in("lam_g"), 1))
            )
            inner_functions.append(inner([s]))
        final_jacobian = jac_chain_rule_with_mapped_outer_function(
            outer_function=canon_outer_g,
            inner_functions=inner_functions,
            parallelization=parallelization,
            return_summed_jacobian=False,
        )
        final_hessian = jac_jac_chain_rule_with_mapped_outer_function(
            outer_function=lagrangian,
            inner_functions=inner_functions_lagrangian,
            parallelization=parallelization,
            return_summed_hessian=False,
        )
        return {
            "hessian": final_hessian,
            "constraint": constraint,
            "g": g,
            "jacobian": final_jacobian,
        }

    def samples_constraints(self, num_samples, parallelization="serial"):
        # grasping_constraint_func = replace_hessian(self.get_grasping_constraint_function())
        g_function, lbg_function, ubg_function, lagrangian_sample = (
            self.make_sample_constraints()
        )
        name = "sample_constraint"
        samples = np.linspace(0, 1, num_samples)
        subnames = []
        for i in range(samples.size):
            if (self.diff_co_options is not None):
                for robot_options in  self.diff_co_options["robots"]:
                    robot_options:DiffCoRobotOptions
                    robot_name = robot_options['name']
                    for group_name in self.robots_by_name[robot_name].collision_model.forward_kinematics_groups_casadi.keys():
                        subnames.append(f'sample_{i}_{robot_name}_{group_name}')
            subnames += [f'sample_{i}_robot_1_grasp_translation_x',f'sample_{i}_robot_1_grasp_translation_y',f'sample_{i}_robot_1_grasp_translation_z',f'sample_{i}_robot_1_grasp_orientation',f'sample_{i}_robot_2_grasp_translation_x',f'sample_{i}_robot_2_grasp_translation_y',f'sample_{i}_robot_2_grasp_translation_z',f'sample_{i}_robot_2_grasp_orientation']
        result = self.contraints_on_samples(
            self.get_inner_samples_function(
                samples,
            ),
            g_function,
            lbg_function,
            ubg_function,
            samples,
            parallelization,
            name,
            subnames
        )
        g, canon_outer_g, constraint = (
            result["g"],
            result["canon_outer_g"],
            result["constraint"],
        )

        inner_functions_lagrangian = []
        inner_functions = []
        for s in samples:
            inner_functions_lagrangian.append(
                self.get_inner_samples_function(
                    [s], lam_g_shape=(lagrangian_sample.numel_in("lam_g"), 1)
                )
            )
            inner_functions.append(self.get_inner_samples_function([s]))
        final_jacobian = jac_chain_rule_with_mapped_outer_function(
            outer_function=canon_outer_g,
            inner_functions=inner_functions,
            parallelization=parallelization,
            return_summed_jacobian=False,
        )
        final_hessian = jac_jac_chain_rule_with_mapped_outer_function(
            outer_function=lagrangian_sample,
            inner_functions=inner_functions_lagrangian,
            parallelization=parallelization,
            return_summed_hessian=False,
        )
        return {
            "hessian": final_hessian,
            "constraint": constraint,
            "g": g,
            "jacobian": final_jacobian,
        }

    def gaze_cost(self, parallelization="serial"):
        # grasping_constraint_func = replace_hessian(self.get_grasping_constraint_function())
        f_function, lagrangian_sample = self.make_gaze_function()
        samples = [1]
        inner = self.get_inner_gaze_function

        result = self.cost_on_samples(
            inner(
                samples,
            ),
            f_function,
            samples,
        )
        f = result["f"]

        inner_functions_lagrangian = []
        inner_functions = []
        for s in samples:
            inner_functions_lagrangian.append(
                inner([s], lam_g_shape=(lagrangian_sample.numel_in("lam_g"), 1))
            )
            inner_functions.append(inner([s]))
        # TODO: those functions here are for the constraints so they expect a lam_g input and that the output is called g
        final_jacobian = jac_chain_rule_with_mapped_outer_function(
            outer_function=f_function,
            inner_functions=inner_functions,
            parallelization=parallelization,
            return_summed_jacobian=False,
        )
        final_hessian = jac_jac_chain_rule_with_mapped_outer_function(
            outer_function=lagrangian_sample,
            inner_functions=inner_functions_lagrangian,
            parallelization=parallelization,
            return_summed_hessian=False,
        )
        return {"hessian": final_hessian, "f": f, "jacobian": final_jacobian}


def get_lagrangian(function):
    return pipe(
        function.convert_in(function.mx_in())
        | {"lam_g": ca.MX.sym("lam_g", function.numel_out("g"), 1)},
        lambda mx_in: ca.Function(
            "canon",
            mx_in
            | {
                "l": ca.dot(
                    ca.vec(
                        function.call({k: v for k, v in mx_in.items() if k != "lam_g"})[
                            "g"
                        ]
                    ),
                    ca.vec(mx_in["lam_g"]),
                )
            },
            function.name_in() + ["lam_g"],
            ["l"],
            {
                "always_inline": True,
                "is_diff_in": function.is_diff_in() + [False],
            },
        ),
        # replace_hessian
    )


def make_polyharmonic_kernel_function(
    feature_size,
):
    sample_in = ca.SX.sym("x", feature_size)
    support_vector = ca.SX.sym(
        "support_vectors",
        feature_size,
        1,
    )
    # result = 0
    # for a, b in zip(ca.vertsplit(sample_in, 3), ca.vertsplit(support_vector, 3)):
    #     result += ca.norm_2(a - b)
    # result /= sample_in.size1() // 3
    result = ca.norm_2((sample_in - support_vector))

    f = ca.Function(
        "polyharmonic_kernel",
        [sample_in, support_vector],
        [result],
        {"is_diff_in": [True, False],
         'cse':True},
    )
    return f

# def make_polyharmonic_kernel_function(
#     feature_size,
# ):

#     func_name = "polyharmonic_kernel"
#     modul = ca.SX
#     sample_in = modul.sym("x", feature_size)
#     # support_vectors = modul.sym('support_vectors',feature_size,num_samples,)
#     support_vector = modul.sym(
#         "support_vectors",
#         feature_size,
#         1,
#     )
#     # sample_in = sample_in.monitor('sample_in')
#     # support_vector = support_vector.monitor('support_vector')
#     result = ca.norm_2((sample_in - support_vector))

#     f = ca.Function(
#         func_name, [sample_in, support_vector], [result], {"is_diff_in": [True, False]}
#     )
#     return f


def load_opt_info(
    compile_path,
    robots,
    carried_object,
    num_control_points,
    order,
    diff_co_options,
    gaze_options,
    solver_options,
):
    def deserialize_function(
        path,
    ):
        with open(str(path), "r") as text_file:
            return ca.Function.deserialize(text_file.read())

    loaded_opt_info = OptimizationInfo(
        robots=[r.__copy__() for r in robots],
        carried_object=carried_object.__copy__(),
    )
    loaded_opt_info.setup_optimization(
        num_control_points, order, diff_co_options, gaze_options
    )
    # get functions
    with open(compile_path / "compiled_functions.txt", "r") as text_file:
        compiled_functions = text_file.read().splitlines()
    for function_name in compiled_functions:
        setattr(
            loaded_opt_info,
            function_name,
            ca_utils.Compile(
                file_name=function_name,
                path=compile_path,
                function_name=function_name,
                get_cached_or_throw=True,
            ),
        )

    nlp_g_canon = deserialize_function(compile_path / "nlp_g_serialized")
    nlp_f_canon = deserialize_function(compile_path / "nlp_f_serialized")
    for name in [
        "nlp",
        "nlp_jac_g",
        "nlp_hess_l",
        "nlp_g",
        "nlp_f",
        "nlp_grad_f",
        "lbg_ubg_func",
        "lbx_ubx_func",
    ]:
        assert hasattr(loaded_opt_info, name)

    x = []
    p = []
    opti = loaded_opt_info.opti.advanced
    for var in opti.symvar():
        if opti.is_parametric(var):
            p.append(var)
        else:
            x.append(var)
    # p = p[0:-4]
    p = ca.veccat(*p)
    x = ca.veccat(*x)

    assert loaded_opt_info.nlp.size_in(0) == x.shape
    assert loaded_opt_info.nlp.size_in(1) == p.shape

    f = nlp_f_canon(x, p)
    g = nlp_g_canon(x, p)
    lbg, ubg = loaded_opt_info.lbg_ubg_func(p)
    opti.subject_to(opti.bounded(lbg, g, ubg))
    opti.minimize(f)

    def read_constraint_slices(filename):
        constraint_slices = {}
        with open(filename, 'r') as file:
            for line in file:
                name, values = line.strip().split(': ')
                start, stop = map(int, values.split(', '))
                constraint_slices[name] = slice(start, stop)
        return constraint_slices

    def read_constraint_subnames(filename):
        constraint_subnames = {}
        with open(filename, 'r') as file:
            for line in file:
                name, subnames_str = line.strip().split(': ')
                subnames = eval(subnames_str)
                constraint_subnames[name] = subnames
        return constraint_subnames

    constraint_slices = read_constraint_slices(str(compile_path / 'constraint_slices.txt'))
    constraint_subnames = read_constraint_subnames(str(compile_path / 'constraint_subnames.txt'))

    loaded_opt_info.constraint_slices = constraint_slices
    loaded_opt_info.constraint_subnames = constraint_subnames
    loaded_opt_info.opti = opti
    loaded_opt_info.solver, _ = get_solver(
        opti,
        solver_options,
        loaded_opt_info.nlp_hess_l,
        loaded_opt_info.nlp_jac_g,
        cache={
            "nlp": loaded_opt_info.nlp,
            "nlp_g": loaded_opt_info.nlp_g,
            "nlp_f": loaded_opt_info.nlp_f,
            "nlp_grad_f": loaded_opt_info.nlp_grad_f,
        },
    )

    return loaded_opt_info


def get_solver(opti, options, hess_lag=None, nlp_jac_g=None, cache=None):
    baked_copy = opti.advanced.baked_copy()

    lubg_func = ca.Function(
        "lbg_ubg",
        [baked_copy.p],
        [baked_copy.lbg, baked_copy.ubg],
        {
            "cse": True,
        },
    )
    nlp = {
        "x": baked_copy.x,
        "p": baked_copy.p,
        "g": ca.cse(baked_copy.g),
        "f": ca.cse(baked_copy.f),
    }
    if hess_lag:
        options |= {"hess_lag": hess_lag}
    if nlp_jac_g:
        options |= {"jac_g": nlp_jac_g}
    if cache:
        options |= {"cache": cache}
    options |= {"calc_lam_p":False}
    solver = ca.nlpsol("kin_opt", "ipopt", nlp, options)
    # planner.initial_plan_data.solver.generate()
    # files = [str((opt_with_collision_compile_path / file).resolve()) for file in ['nlp_f.so','nlp_g.so','nlp_grad_f.so','nlp_hess_l.so','nlp_jac_g.so','nlp.so']]

    # compile_command = f"""gcc -Ofast -fPIC -march=native -mavx2 -mfma -shared kin_opt.c -o solver3.so -L/home/user/.local/lib/python3.10/site-packages/casadi/ -lipopt -lm {' '.join(files)}"""
    # os.system(compile_command)
    # solver = ca.external("kin_opt", "./solver3.so")
    return solver, lubg_func
