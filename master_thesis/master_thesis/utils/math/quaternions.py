# import os, sys, pathlib
# def add_root_directory():
#     path = pathlib.Path(os.getcwd()).resolve()
#     while not (root_file := path / "root_toys").is_file():
#         path = path.parent
#         if path == path.parent:
#             return None
#     sys.path.insert(0, str(path))
# add_root_directory()

from typing import Union
# from pydrake.all import Quaternion
from pydrake.common.eigen_geometry import Quaternion
import numpy as np
import casadi as ca

def quat_exp(q: Union[np.ndarray, Quaternion], t: float) -> Union[np.ndarray, Quaternion]:
    if isinstance(q, np.ndarray):
        theta = np.arccos(q[0])
        if abs(theta) < 1e-6:
            new_u = np.zeros(3)
        else:
            new_u = q[1:] * np.sin(theta * t) / np.sin(theta)
        new_w = np.cos(t * theta)
        wxyz = np.hstack([new_w, new_u])
        wxyz = wxyz / np.linalg.norm(wxyz)
        return wxyz
    else:
        theta = np.arccos(q.w())
        if abs(theta) < 1e-6:
            new_u = np.zeros(3)
        else:
            new_u = q.wxyz()[1:] * np.sin(theta * t) / np.sin(theta)
        new_w = np.cos(t * theta)
        wxyz = np.hstack([new_w, new_u])
        wxyz = wxyz / np.linalg.norm(wxyz)
        return Quaternion(wxyz)
    
def quat_log(q: Union[np.ndarray, Quaternion]) -> np.ndarray:
    if isinstance(q, np.ndarray):
        theta = np.arccos(q[0])
        if abs(np.sin(theta)) < 1e-6:
            return np.zeros(3)
        else:
            return theta * q[1:] / np.sin(theta)
    else:
        theta = np.arccos(q.w())
        if abs(np.sin(theta)) < 1e-6:
            return np.zeros(3)
        else:
            return theta * (q.wxyz()[1:]) / np.sin(theta)

def hamiltonian_product(quaternion0: Union[np.ndarray, Quaternion], quaternion1: Union[np.ndarray, Quaternion]) -> np.ndarray:
    casadi = False
    if isinstance(quaternion0, np.ndarray):
        w0, x0, y0, z0 = quaternion0
    elif isinstance(quaternion0, (ca.MX, ca.SX, ca.DM)):
        casadi = True
        w0, x0, y0, z0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    else:
        w0, x0, y0, z0 = quaternion0.wxyz()

    if isinstance(quaternion1, np.ndarray):
        w1, x1, y1, z1 = quaternion1
    elif isinstance(quaternion1, (ca.MX, ca.SX, ca.DM)):
        casadi = True
        w1, x1, y1, z1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]
    else:
        w1, x1, y1, z1 = quaternion1.wxyz()

    if casadi:
        return ca.vertcat(-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)
    else:
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])
    


def quat_times_vector(q: Union[np.ndarray, Quaternion], v: np.ndarray) -> np.ndarray:
    if isinstance(q, np.ndarray):
        q_conj = q.copy()
        q_conj[1:] *= -1
        return hamiltonian_product(hamiltonian_product(q, np.hstack([0, v])), q_conj)[1:]
    else:
        q_conj = q.conjugate()
        return hamiltonian_product(hamiltonian_product(q.wxyz(), np.hstack([0, v])), q_conj.wxyz())[1:]

def angular_velocity_to_quaternion_dot(q: Union[np.ndarray, Quaternion], w: np.ndarray) -> np.ndarray:
    if isinstance(q, (ca.MX, ca.SX, ca.DM)) or isinstance(w, (ca.MX, ca.SX, ca.DM)):
        wx, wy, wz = w[0], w[1], w[2]
        omega_operator = lambda wx, wy, wz: ca.vertcat(ca.horzcat(0.0, -wx, -wy, -wz),
                                                        ca.horzcat(wx, 0.0, wz, -wy),
                                                        ca.horzcat(wy, -wz, 0.0, wx),
                                                        ca.horzcat(wz, wy, -wx, 0.0))
        return 1/2 * omega_operator(wx, wy, wz) @ q
    wx, wy, wz = w
    omega_operator = lambda wx, wy, wz: np.array([[0.0, -wx, -wy, -wz], [wx, 0.0, wz, -wy], [wy, -wz, 0.0, wx], [wz, wy, -wx, 0.0]])
    if isinstance(q, np.ndarray):
        return 1/2 * omega_operator(*w) @ q
    else:
        return 1/2 * omega_operator(*w) @ q.wxyz()

def slerp(t: float, q0: Union[np.ndarray, Quaternion], q1: Union[np.ndarray, Quaternion]) -> Union[np.ndarray, Quaternion]:
    if isinstance(q0, np.ndarray) and isinstance(q1, np.ndarray):
        q0_inv = np.hstack([q0[0], -q0[1], -q0[2], -q0[3]])
        q = hamiltonian_product(q0,quat_exp(hamiltonian_product(q0_inv, q1), t))
        return q
    else:
        q0_inv = q0.inverse()
        q = q0.multiply(quat_exp(q0_inv.multiply(q1),t))
        return q

def slerp_dot(t: float, q0: Union[np.ndarray, Quaternion], q1: Union[np.ndarray, Quaternion]) -> Union[np.ndarray, Quaternion]:
    if isinstance(q0, np.ndarray) and isinstance(q1, np.ndarray):
        q0_inv = np.hstack([q0[0], -q0[1], -q0[2], -q0[3]])
        q = hamiltonian_product(q0,quat_exp(hamiltonian_product(q0_inv, q1), t))
        return q * quat_log(hamiltonian_product(q0_inv, q1)) * 2
    else:
        q0_inv = q0.inverse()
        q = q0.multiply(quat_exp(q0_inv.multiply(q1),t))
        return q.multiply(quat_log(q0_inv.multiply(q1))) * 2
    
def quaternion_dot_numerical(q0: Union[np.ndarray, Quaternion], q1: Union[np.ndarray, Quaternion], dt: float) -> np.ndarray:
    if isinstance(q0, np.ndarray) and isinstance(q1, np.ndarray):
        q0_conj = np.hstack([q0[0], -q0[1], -q0[2], -q0[3]])
        return hamiltonian_product(q1, q0_conj)[1:] * 2 / dt
    else:
        q0_conj = q0.conjugate()
        return hamiltonian_product(q1.wxyz(), q0_conj.wxyz())[1:] * 2 / dt

def quaternion_dot_to_angular_velocity(q: Union[np.ndarray, Quaternion,ca.MX, ca.SX, ca.DM], qd: np.ndarray) -> np.ndarray:
    if isinstance(q, np.ndarray):
        q_conj = np.hstack([q[0], -q[1], -q[2], -q[3]])
        return hamiltonian_product((2 * q_conj), qd)[1:]
    elif isinstance(q, (ca.MX, ca.SX, ca.DM)):
        q_conj = ca.vertcat(q[0], -q[1], -q[2], -q[3])
        return hamiltonian_product((2 * q_conj), qd)[1:]
    else:
        q_conj = q.conjugate()
        return hamiltonian_product((2 * q_conj.wxyz()), qd)[1:]