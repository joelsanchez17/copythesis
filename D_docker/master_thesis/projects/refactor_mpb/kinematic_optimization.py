
# import multiprocessing
from toolz import memoize, pipe, accumulate, groupby, compose, compose_left
from toolz.curried import do
from collections import namedtuple

# import sympy
import casadi as ca
import time
import numpy as np
from utils.misc import *
import utils.my_casadi.misc as ca_utils
import typing as T
# from utils.my_drake.casadi.multibody_wrapper import CasadiMultiBodyPlantWrapper

from projects.refactor_mpb.multibody_wrapper import MultiBodyPlantWrapper
from utils.math.BSpline import BSpline

class KinematicOptimization:
    
    @staticmethod
    def fwd_kin_translation_constraint(frame_pose_1, point_1, frame_pose_2, point_2, lower_bound, upper_bound,):
        translation_1 = frame_pose_1[0:3, 3] + frame_pose_1[0:3, 0:3] @ point_1
        translation_2 = frame_pose_2[0:3, 3] + frame_pose_2[0:3, 0:3] @ point_2
        expr = translation_2 - translation_1
        
        return expr, ca_utils.veccat(lower_bound), ca_utils.veccat(upper_bound)
    
    @staticmethod
    def fwd_kin_orientation_constraint( rotation_1, rotation_2, frame_2_in_1, theta_bound):
        """
        Rotate frame 1 by rotation_1 and frame 2 by rotation_2; describe frame_1 in frame_2 (by frame_2_in_1 belonging to SE(3)); describe the rotation between those two rotated frames by axis-angle representation. The maximum angle around the axis is bounded by theta_bound.
        
        Returns [expression, lbg, ubg]
        """
        trace_R1_R2 = ca.trace(rotation_1.T@frame_2_in_1[0:3, 0:3]@rotation_2)
        return (trace_R1_R2 - (2*ca.cos(theta_bound) + 1)), 0, 4.
    @staticmethod
    def velocity_bound_constraint(variables, lower_bound,upper_bound,trajectory_duration = 1):
        lower_bound = ca_utils.veccat(lower_bound)
        upper_bound = ca_utils.veccat(upper_bound)
        variables = ca_utils.veccat(variables)
        return variables/trajectory_duration, lower_bound, upper_bound
    @staticmethod
    def position_bound_constraint(variables, lower_bound,upper_bound):
        lower_bound = ca_utils.veccat(lower_bound)
        upper_bound = ca_utils.veccat(upper_bound)
        variables = ca_utils.veccat(variables)
        return variables, lower_bound, upper_bound
    @staticmethod
    def quaternion_constraint(quaternion):
        quaternion = ca.veccat(quaternion)
        return ca.sumsqr(quaternion) , 1, 1
    @staticmethod
    def manipulability(jacobian):
        return ca.sqrt(ca.det(jacobian@jacobian.T))
    @staticmethod
    def distance_between_points_in_frames(frame_1_pose,point_1,frame_2_pose,point_2):
        translation_1 = frame_1_pose[0:3, 3] + frame_1_pose[0:3, 0:3] @ point_1
        translation_2 = frame_2_pose[0:3, 3] + frame_2_pose[0:3, 0:3] @ point_2
        return ca.norm_2(translation_1-translation_2)
    @staticmethod
    def distance_between_points_in_frames_constraint(frame_1_pose,point_1,frame_2_pose,point_2,distance):
        return KinematicOptimization.distance_between_points_in_frames(frame_1_pose,point_1,frame_2_pose,point_2), distance, distance
    