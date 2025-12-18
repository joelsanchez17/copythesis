import typing as T
import numpy as np
import casadi as ca
import pathlib, os
import time
import numba
import utils.my_casadi.misc as ca_utils

# sf
import tempfile


class FrankaInverseKinematics:
    casadi_obj_IK: ca.Function
    casadi_EE_IK: ca.Function
    # new version
    def __init__(self, base_pose: np.ndarray, grasping_transform: np.ndarray):
        self.base_pose = base_pose
        self.grasping_transform = grasping_transform

        # Convert numpy arrays to CasADi symbolic variables
        base_pose_sym = ca.MX(base_pose)
        grasping_transform_sym = ca.MX(grasping_transform)

        self.codegen_path = pathlib.Path(tempfile.mkdtemp())
        self.codegen_path.mkdir(parents=True, exist_ok=True)

        # Pass symbolic matrices here, not numpy arrays
        ###################################
        # base_pose_sym = ca.MX.sym("base_pose", 4, 4)
        # grasping_transform_sym = ca.MX.sym("grasping_transform", 4, 4)

        # # Pass these into your function
        # f = make_franka_IK_object_with_base_transform(base_pose_sym, grasping_transform_sym)
        ##############################
        self.casadi_obj_IK = make_franka_IK_object_with_base_transform(
            base_pose_sym, grasping_transform_sym
        )

        self.casadi_EE_IK = make_franka_IK_EE_with_base_transform(base_pose_sym)

        self.casadi_obj_IK_compiled = ca_utils.Compile(
            "IK_obj", self.codegen_path, function=self.casadi_obj_IK
        )
        self.casadi_EE_IK_compiled = ca_utils.Compile(
            "IK_EE", self.codegen_path, function=self.casadi_EE_IK
        )

        self.base_pose_inv = np.linalg.inv(self.base_pose)

    # def __init__(self, base_pose: np.ndarray, grasping_transform: np.ndarray):
    #     self.base_pose = base_pose
    #     self.grasping_transform = grasping_transform
    #     self.codegen_path = pathlib.Path(tempfile.mkdtemp())
    #     self.codegen_path.mkdir(parents=True, exist_ok=True)
    #     self.casadi_obj_IK = make_franka_IK_object_with_base_transform(
    #         base_pose, grasping_transform
    #     )
    #     self.casadi_EE_IK = make_franka_IK_EE_with_base_transform(base_pose)
    #     self.casadi_obj_IK_compiled = ca_utils.Compile(
    #         "IK_obj", self.codegen_path, function=self.casadi_obj_IK
    #     )
    #     self.casadi_EE_IK_compiled = ca_utils.Compile(
    #         "IK_EE", self.codegen_path, function=self.casadi_EE_IK
    #     )
    #     self.base_pose_inv = np.linalg.inv(self.base_pose)

    def batch_franka_IK_EE(self, O_T_EE_array, q7, q_actual_array):
        robot_base_inverse = self.base_pose_inv
        q_all = batch_franka_IK_EE_(
            robot_base_inverse, O_T_EE_array, q7, q_actual_array
        )
        return q_all[~np.isnan(q_all).any(axis=2)]


@numba.jit(nopython=True, parallel=False)
def rotation_matrix(axis, theta):
    if axis.size == 3:
        axis = axis.reshape(-1, 3)
    if isinstance(theta, float) or isinstance(theta, int):
        theta = np.full(axis.shape[0], theta)
    elif theta.size == 1:
        theta = np.full(axis.shape[0], theta)
    out = np.empty((axis.shape[0], 3, 3))
    for i in range(
        0,
        axis.shape[0],
    ):
        axis[i] = axis[i] / np.sqrt(np.dot(axis[i], axis[i]))
        a = np.cos(theta[i] / 2.0)
        b, c, d = -axis[i] * np.sin(theta[i] / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        out[i] = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
    return out


@numba.njit
def franka_IK_EE(robot_base_inverse, O_T_EE_array, q7, q_actual_array):

    #     robot_base_pose = robot_opt_info.plant.GetFrameByName('panda_link0').CalcPose(robot_opt_info.plant.CreateDefaultContext(),robot_opt_info.plant.GetFrameByName('world')).GetAsMatrix4()
    # object_frame_to_object_EE_frame = carried_object.plant.GetFrameByName(f'carried_object_EE_{i+1}_frame').CalcPose(carried_object.plant.CreateDefaultContext(),carried_object.plant.GetFrameByName('carried_object')).GetAsMatrix4()
    # robot_opt_info.obj_IK = make_franka_IK_object_with_base_transform(base_pose_in_world = robot_base_pose,grasping_transform = object_frame_to_object_EE_frame)
    # robot_opt_info.EE_IK = make_franka_IK_EE_with_base_transform(base_pose_in_world = robot_base_pose)
    q_all_NAN = np.full((4, 7), np.nan)
    q_NAN = np.full(7, np.nan)
    q_all = q_all_NAN.copy()

    O_T_EE = robot_base_inverse @ O_T_EE_array

    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d7e = 0.2104
    a4 = 0.0825
    a7 = 0.0880

    LL24 = 0.10666225
    LL46 = 0.15426225
    L24 = np.sqrt(LL24)
    L46 = np.sqrt(LL46)

    thetaH46 = 1.35916951803
    theta342 = 1.31542071191
    theta46H = 0.211626808766

    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    if q7 <= q_min[6] or q7 >= q_max[6]:
        return q_all_NAN
    else:
        for i in range(4):
            q_all[i][6] = q7

    R_EE = O_T_EE[:3, :3]
    z_EE = O_T_EE[:3, 2]
    p_EE = O_T_EE[:3, 3]
    p_7 = p_EE - d7e * z_EE

    x_EE_6 = np.array([np.cos(q7 - np.pi / 4), -np.sin(q7 - np.pi / 4), 0.0])
    x_6 = R_EE @ x_EE_6
    x_6 /= np.linalg.norm(x_6)
    p_6 = p_7 - a7 * x_6

    p_2 = np.array([0.0, 0.0, d1])
    V26 = p_6 - p_2

    LL26 = np.dot(V26, V26)
    L26 = np.sqrt(LL26)

    if L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24:
        return q_all_NAN

    theta246 = np.arccos((LL24 + LL46 - LL26) / (2.0 * L24 * L46))
    q4 = theta246 + thetaH46 + theta342 - 2.0 * np.pi
    if q4 <= q_min[3] or q4 >= q_max[3]:
        return q_all_NAN
    else:
        for i in range(4):
            q_all[i][3] = q4

    theta462 = np.arccos((LL26 + LL46 - LL24) / (2.0 * L26 * L46))
    theta26H = theta46H + theta462
    D26 = -L26 * np.cos(theta26H)

    Z_6 = np.cross(z_EE, x_6)
    Y_6 = np.cross(Z_6, x_6)
    R_6 = np.column_stack((x_6, Y_6 / np.linalg.norm(Y_6), Z_6 / np.linalg.norm(Z_6)))
    V_6_62 = R_6.T @ (-V26)

    Phi6 = np.arctan2(V_6_62[1], V_6_62[0])
    Theta6 = np.arcsin(D26 / np.sqrt(V_6_62[0] ** 2 + V_6_62[1] ** 2))

    q6 = np.zeros(2)
    q6[0] = np.pi - Theta6 - Phi6
    q6[1] = Theta6 - Phi6

    for i in range(2):
        if q6[i] <= q_min[5]:
            q6[i] += 2.0 * np.pi
        elif q6[i] >= q_max[5]:
            q6[i] -= 2.0 * np.pi

        if q6[i] <= q_min[5] or q6[i] >= q_max[5]:
            q_all[2 * i] = q_NAN
            q_all[2 * i + 1] = q_NAN
        else:
            q_all[2 * i][5] = q6[i]
            q_all[2 * i + 1][5] = q6[i]

    if np.isnan(q_all[0][5]) and np.isnan(q_all[2][5]):
        return q_all_NAN

    thetaP26 = 3.0 * np.pi / 2 - theta462 - theta246 - theta342
    thetaP = np.pi - thetaP26 - theta26H
    LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)

    z_5_all = np.zeros((4, 3))
    V2P_all = np.zeros((4, 3))

    for i in range(2):
        z_6_5 = np.array([np.sin(q6[i]), np.cos(q6[i]), 0.0])
        z_5 = R_6 @ z_6_5
        V2P = p_6 - LP6 * z_5 - p_2

        z_5_all[2 * i] = z_5
        z_5_all[2 * i + 1] = z_5
        V2P_all[2 * i] = V2P
        V2P_all[2 * i + 1] = V2P

        L2P = np.linalg.norm(V2P)

        if np.abs(V2P[2] / L2P) > 0.999:
            q_all[2 * i][0] = q_actual_array[0, 0]
            q_all[2 * i][1] = 0.0
            q_all[2 * i + 1][0] = q_actual_array[0, 0]
            q_all[2 * i + 1][1] = 0.0
        else:
            q_all[2 * i][0] = np.arctan2(V2P[1], V2P[0])
            q_all[2 * i][1] = np.arccos(V2P[2] / L2P)
            if q_all[2 * i][0] < 0:
                q_all[2 * i + 1][0] = q_all[2 * i][0] + np.pi
            else:
                q_all[2 * i + 1][0] = q_all[2 * i][0] - np.pi
            q_all[2 * i + 1][1] = -q_all[2 * i][1]

    for i in range(4):
        if (
            q_all[i][0] <= q_min[0]
            or q_all[i][0] >= q_max[0]
            or q_all[i][1] <= q_min[1]
            or q_all[i][1] >= q_max[1]
        ):
            q_all[i] = q_NAN
            continue

        z_3 = V2P_all[i] / np.linalg.norm(V2P_all[i])
        Y_3 = -np.cross(V26, V2P_all[i])
        y_3 = Y_3 / np.linalg.norm(Y_3)
        x_3 = np.cross(y_3, z_3)
        c1 = np.cos(q_all[i][0])
        s1 = np.sin(q_all[i][0])
        R_1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
        c2 = np.cos(q_all[i][1])
        s2 = np.sin(q_all[i][1])
        R_1_2 = np.array([[c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])
        R_2 = R_1 @ R_1_2
        x_2_3 = R_2.T @ x_3
        q_all[i][2] = np.arctan2(x_2_3[2], x_2_3[0])

        if q_all[i][2] <= q_min[2] or q_all[i][2] >= q_max[2]:
            q_all[i] = q_NAN
            continue

        VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5_all[i]
        c6 = np.cos(q_all[i][5])
        s6 = np.sin(q_all[i][5])
        R_5_6 = np.array([[c6, -s6, 0.0], [0.0, 0.0, -1.0], [s6, c6, 0.0]])
        R_5 = R_6 @ R_5_6.T
        V_5_H4 = R_5.T @ VH4

        q_all[i][4] = -np.arctan2(V_5_H4[1], V_5_H4[0])
        if q_all[i][4] <= q_min[4] or q_all[i][4] >= q_max[4]:
            q_all[i] = q_NAN
            continue

    return q_all


@numba.njit(parallel=True)
def batch_franka_IK_EE_(robot_base_inverse, O_T_EE_array, q7, q_actual_array):
    q_all = np.empty((O_T_EE_array.shape[0], 4, 7))
    for i in numba.prange(O_T_EE_array.shape[0]):
        q_all[i] = franka_IK_EE(
            robot_base_inverse, O_T_EE_array[i], q7[i], q_actual_array[i]
        )
    return q_all


def batch_franka_IK_EE(robot_base_inverse, O_T_EE_array, q7, q_actual_array):
    q_all = batch_franka_IK_EE_(robot_base_inverse, O_T_EE_array, q7, q_actual_array)

    return q_all[~np.isnan(q_all).any(axis=2)]


def make_casadi_franka_IK_EE():
    def nested_or(*args):
        if len(args) == 1:
            return args[0]
        else:
            return ca.logic_or(args[0], nested_or(*args[1:]))

    q7 = ca.MX.sym("q7")
    q_current = ca.MX.sym("q_current", 7)
    O_T_EE = ca.MX.sym("O_T_EE", 4, 4)

    q_all_NAN = ca.DM.nan(4, 7)
    q_NAN = ca.DM.nan(1, 7)
    q_all = ca.MX.zeros(4, 7)

    conditions_for_nan = []
    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d7e = 0.2104
    a4 = 0.0825
    a7 = 0.0880

    LL24 = a4**2 + d3**2
    LL46 = a4**2 + d5**2
    L24 = ca.sqrt(LL24)
    L46 = ca.sqrt(LL46)

    thetaH46 = ca.atan2(d5, a4)
    theta342 = ca.atan2(d3, a4)
    theta46H = ca.atan2(a4, d5)

    LL24 = 0.10666225
    LL46 = 0.15426225
    L24 = 0.326591870689
    L46 = 0.392762332715

    thetaH46 = 1.35916951803
    theta342 = 1.31542071191
    theta46H = thetaH64 = 0.211626808766
    theta243 = ca.atan2(d3, a4)
    thetaH64 = np.pi / 2 - thetaH46
    q_min = ca.DM([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

    q_max = ca.DM([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    conditions_for_nan.append(ca.logic_or(q7 <= q_min[6], q7 >= q_max[6]))
    for i in range(4):
        q_all[i, 6] = q7
    R_EE = O_T_EE[0:3, 0:3]
    z_EE = O_T_EE[0:3, 2]
    p_EE = O_T_EE[0:3, 3]
    p_7 = p_EE - d7e * z_EE

    x_EE_6 = ca.vertcat(ca.cos(q7 - ca.pi / 4), -ca.sin(q7 - ca.pi / 4), 0.0)
    x_6 = ca.mtimes(R_EE, x_EE_6)
    x_6 = x_6 / ca.norm_2(x_6)
    p_6 = p_7 - a7 * x_6

    p_2 = ca.DM([0.0, 0.0, d1])
    V26 = p_6 - p_2

    LL26 = ca.sumsqr(V26)
    L26 = ca.sqrt(LL26)
    conditions_for_nan.append(
        ca.logic_or(L24 + L46 < L26, ca.logic_or(L24 + L26 < L46, L26 + L46 < L24))
    )
    theta246 = ca.acos((LL24 + LL46 - LL26) / (2 * L24 * L46))
    q4 = theta246 + thetaH46 + theta243 - 2 * ca.pi
    conditions_for_nan.append(ca.logic_or(q4 <= q_min[3], q4 >= q_max[3]))
    for i in range(4):
        q_all[i, 3] = q4
    theta462 = ca.acos(
        (LL26 + LL46 - LL24) / (2 * L26 * L46)
    )  ##.monitor('theta264 = ca.acos((LL26 + LL46 - LL24)/(2*L26*L46))')
    theta26H = thetaH64 + theta462
    D26 = -L26 * ca.cos(theta26H)
    Z_6 = ca.cross(
        z_EE,
        x_6,
    )
    Y_6 = ca.cross(
        Z_6,
        x_6,
    )
    R_6 = ca.horzcat(x_6, Y_6 / ca.norm_2(Y_6), Z_6 / ca.norm_2(Z_6))
    V_6_62 = ca.mtimes(R_6.T, -V26)

    Phi6 = ca.atan2(V_6_62[1], V_6_62[0])
    Theta6 = ca.asin((D26 / ca.sqrt(V_6_62[0] * V_6_62[0] + V_6_62[1] * V_6_62[1])))

    q6 = [ca.pi - Theta6 - Phi6, Theta6 - Phi6]

    for i in range(2):

        q6[i] = ca.if_else(
            q6[i] >= q_max[5],
            q6[i] - 2 * ca.pi,
            ca.if_else(q6[i] <= q_min[5], q6[i] + 2 * ca.pi, q6[i]),
        )
        q_all[2 * i, 5] = q6[i]
        q_all[2 * i + 1, 5] = q6[i]
        q_all[2 * i, :] = ca.if_else(
            ca.logic_or(q6[i] <= q_min[5], q6[i] >= q_max[5]), q_NAN, q_all[2 * i, :]
        )
        q_all[2 * i + 1, :] = ca.if_else(
            ca.logic_or(q6[i] <= q_min[5], q6[i] >= q_max[5]),
            q_NAN,
            q_all[2 * i + 1, :],
        )
    conditions_for_nan.append(
        ca.logic_and(
            ca.logic_or(q6[0] <= q_min[5], q6[0] >= q_max[5]),
            ca.logic_or(q6[1] <= q_min[5], q6[1] >= q_max[5]),
        )
    )

    thetaP26 = 3 * ca.pi / 2 - theta462 - theta246 - theta342
    thetaP = ca.pi - thetaP26 - theta26H
    LP6 = L26 * ca.sin(thetaP26) / ca.sin(thetaP)

    z_5_all = []
    V2P_all = []

    for i in range(2):
        z_6_5 = ca.vertcat(ca.sin(q6[i]), ca.cos(q6[i]), 0.0)
        z_5 = ca.mtimes(R_6, z_6_5)
        V2P = p_6 - LP6 * z_5 - p_2

        z_5_all.extend([z_5, z_5])
        V2P_all.extend([V2P, V2P])

        L2P = ca.norm_2(V2P)
        c = ca.fabs(V2P[2] / L2P) > 0.99999999999
        q_all[2 * i, 0] = ca.atan2(V2P[1], V2P[0])
        q_all[2 * i, 1] = ca.acos(V2P[2] / L2P)
        q_all[2 * i + 1, 1] = -q_all[2 * i, 1]
        q_all[2 * i + 1, 0] = ca.if_else(
            q_all[2 * i, 0] < 0, q_all[2 * i, 0] + ca.pi, q_all[2 * i, 0] - ca.pi
        )

        q_all[2 * i, 0] = ca.if_else(c, q_current[0], q_all[2 * i, 0])
        q_all[2 * i, 1] = ca.if_else(c, 0.0, q_all[2 * i, 1])
        q_all[2 * i + 1, 0] = ca.if_else(c, q_current[0], q_all[2 * i + 1, 0])
        q_all[2 * i + 1, 1] = ca.if_else(c, 0.0, q_all[2 * i + 1, 1])

    for i in range(4):
        q_all = q_all
        c = ca.logic_or(
            q_all[i, 0] <= q_min[0],
            ca.logic_or(
                q_all[i, 0] >= q_max[0],
                ca.logic_or(q_all[i, 1] <= q_min[1], q_all[i, 1] >= q_max[1]),
            ),
        )
        z_3 = V2P_all[i] / ca.norm_2(V2P_all[i])
        Y_3 = -ca.cross(V2P_all[i], V26)
        y_3 = Y_3 / ca.norm_2(Y_3)
        x_3 = ca.cross(z_3, y_3)

        R_1 = ca.blockcat(
            [
                [ca.cos(q_all[i, 0]), -ca.sin(q_all[i, 0]), 0.0],
                [ca.sin(q_all[i, 0]), ca.cos(q_all[i, 0]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        R_1_2 = ca.blockcat(
            [
                [ca.cos(q_all[i, 1]), -ca.sin(q_all[i, 1]), 0.0],
                [0.0, 0.0, 1.0],
                [-ca.sin(q_all[i, 1]), -ca.cos(q_all[i, 1]), 0.0],
            ]
        )

        R_2 = ca.mtimes(R_1, R_1_2)
        x_2_3 = ca.mtimes(R_2.T, x_3)
        q_all[i, 2] = ca.atan2(x_2_3[2], x_2_3[0])

        c2 = ca.logic_or(q_all[i, 2] <= q_min[2], q_all[i, 2] >= q_max[2])

        VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5_all[i]
        R_5_6 = ca.blockcat(
            [
                [ca.cos(q_all[i, 5]), -ca.sin(q_all[i, 5]), 0.0],
                [0.0, 0.0, -1.0],
                [ca.sin(q_all[i, 5]), ca.cos(q_all[i, 5]), 0.0],
            ]
        )

        R_5 = ca.mtimes(R_6, R_5_6.T)
        V_5_H4 = ca.mtimes(R_5.T, VH4)
        q_all[i, 4] = -ca.atan2(V_5_H4[1], V_5_H4[0])
        c3 = ca.logic_or(q_all[i, 4] <= q_min[4], q_all[i, 4] >= q_max[4])
        q_all[i, :] = ca.if_else(c2, q_NAN, q_all[i, :])
        q_all[i, :] = ca.if_else(c3, q_NAN, q_all[i, :])
        q_all[i, :] = ca.if_else(c, q_NAN, q_all[i, :])
    q_all = ca.if_else(nested_or(*conditions_for_nan), q_all_NAN, q_all)
    f = ca.Function(
        "franka_IK_EE",
        [O_T_EE, q7, q_current],
        [q_all],
        ["O_T_EE", "q_7", "q_current"],
        ["q_all"],
    )
    return f
## old one

# def make_franka_IK_EE_with_base_transform(base_pose_in_world):
#     f = make_casadi_franka_IK_EE()
#     mx_in = f.convert_in(f.mx_in())
#     out = f.call(
#         {
#             "O_T_EE": (
#                 ca_utils.casadi_inverse_homogeneous_transform(base_pose_in_world)
#                 @ mx_in["O_T_EE"]
#             ),
#             "q_7": mx_in["q_7"],
#             "q_current": mx_in["q_current"],
#         }
#     )
#     out = out
#     return ca.Function("franka_IK_EE_with_base", mx_in | out, mx_in.keys(), out.keys())

# new version

def make_franka_IK_EE_with_base_transform(base_pose_in_world):
    f = make_casadi_franka_IK_EE()
    mx_in = f.convert_in(f.mx_in())  # dict of symbolic inputs

    # Compose new O_T_EE
    new_O_T_EE = ca_utils.casadi_inverse_homogeneous_transform(base_pose_in_world) @ mx_in["O_T_EE"]

    out = f.call(
        {
            "O_T_EE": new_O_T_EE,
            "q_7": mx_in["q_7"],
            "q_current": mx_in["q_current"],
        }
    )

    # Prepare input and output lists (for CasADi Function constructor)
    input_syms = list(mx_in.values())
    output_syms = list(out.values())

    # Names for inputs and outputs (optional but good)
    input_names = list(mx_in.keys())
    output_names = list(out.keys())

    return ca.Function("franka_IK_EE_with_base", input_syms, output_syms, input_names, output_names)

## casadi old

# def make_franka_IK_object_with_base_transform(base_pose_in_world, grasping_transform):
#     f = make_franka_IK_EE_with_base_transform(base_pose_in_world)

#     mx_in = f.convert_in(f.mx_in())
#     offset_transform = ca.MX.sym("offset_transform", 4, 4)
#     out = f.call(
#         {
#             "O_T_EE": ((mx_in["O_T_EE"] @ grasping_transform @ offset_transform)),
#             "q_7": mx_in["q_7"],
#             "q_current": mx_in["q_current"],
#         }
#     )
#     mx_in.update({"offset_transform": offset_transform})
#     mx_in.update(out)
#     return ca.Function("franka_IK_object_with_base", mx_in, mx_in.keys(), out.keys())

## new version compatible with casadi 3.6.5
def make_franka_IK_object_with_base_transform(base_pose_in_world, grasping_transform):
    f = make_franka_IK_EE_with_base_transform(base_pose_in_world)

    mx_in = f.convert_in(f.mx_in())
    offset_transform = ca.MX.sym("offset_transform", 4, 4)

    out = f.call(
        {
            "O_T_EE": (mx_in["O_T_EE"] @ grasping_transform @ offset_transform),
            "q_7": mx_in["q_7"],
            "q_current": mx_in["q_current"],
        }
    )

    # Combine inputs: original inputs + offset_transform
    input_syms = list(mx_in.values()) + [offset_transform]
    input_names = list(mx_in.keys()) + ["offset_transform"]

    # Outputs from the call
    output_syms = list(out.values())
    output_names = list(out.keys())

    return ca.Function("franka_IK_object_with_base", input_syms, output_syms, input_names, output_names)






@numba.njit
def get_IK_case(q_actual_array):
    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    a4 = 0.0825
    # FK for getting current case id
    c1_a = np.cos(q_actual_array[0])
    s1_a = np.sin(q_actual_array[0])
    c2_a = np.cos(q_actual_array[1])
    s2_a = np.sin(q_actual_array[1])
    c3_a = np.cos(q_actual_array[2])
    s3_a = np.sin(q_actual_array[2])
    c4_a = np.cos(q_actual_array[3])
    s4_a = np.sin(q_actual_array[3])
    c5_a = np.cos(q_actual_array[4])
    s5_a = np.sin(q_actual_array[4])
    c6_a = np.cos(q_actual_array[5])
    s6_a = np.sin(q_actual_array[5])

    As_a = np.zeros((7, 4, 4))
    As_a[0] = np.array(
        [
            [c1_a, -s1_a, 0.0, 0.0],
            [s1_a, c1_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, d1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    As_a[1] = np.array(
        [
            [c2_a, -s2_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-s2_a, -c2_a, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    As_a[2] = np.array(
        [
            [c3_a, -s3_a, 0.0, 0.0],
            [0.0, 0.0, -1.0, -d3],
            [s3_a, c3_a, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    As_a[3] = np.array(
        [
            [c4_a, -s4_a, 0.0, a4],
            [0.0, 0.0, -1.0, 0.0],
            [s4_a, c4_a, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    As_a[4] = np.array(
        [
            [1.0, 0.0, 0.0, -a4],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    As_a[5] = np.array(
        [
            [c5_a, -s5_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, d5],
            [-s5_a, -c5_a, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    As_a[6] = np.array(
        [
            [c6_a, -s6_a, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [s6_a, c6_a, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    Ts_a = np.zeros((7, 4, 4))
    Ts_a[0] = As_a[0]
    for j in range(1, 7):
        Ts_a[j] = Ts_a[j - 1] @ As_a[j]

    # identify q6 case
    V62_a = Ts_a[1, :3, 3] - Ts_a[6, :3, 3]
    V6H_a = Ts_a[4, :3, 3] - Ts_a[6, :3, 3]
    Z6_a = Ts_a[6, :3, 2]
    is_case6_0 = np.dot(np.cross(V6H_a, V62_a), Z6_a) <= 0

    # identify q1 case
    is_case1_1 = q_actual_array[1] < 0

    return is_case6_0 * 2 + is_case1_1


def get_IK_case_both(q_1, q_2):
    return get_IK_case(q_1) + get_IK_case(q_2) * 4
