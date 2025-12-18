import casadi as ca
import numpy as np

"""
TODO:
- quatenrion Logarithm map
- euler angle to rotation matrix
"""

def quaternion_to_euler_angles(q, seq, extrinsic=False):
    """
    Calculate the three angles of the Euler angle representation of a quaternion `[q_w, q_x, q_y, q_z]` using CasaDi.
    `seq = 'xyz'` etc or `seq = [i,j,k] ∈ {1,2,3}` where `i != j` and `j != k` or  and represents the order of the rotation axes (1 = x, 2 = y, 3 = z).
    
    From: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302
    """
    # 
    if isinstance(seq, str):
        seq = [1 if i=='x' else 2 if i=='y' else 3 if i=='z' else -1 for i in seq]
    if not extrinsic:
        seq = seq[::-1]
    assert seq[0] != seq[1] and seq[1] != seq[2]
    assert all([i in [1,2,3] for i in seq])
    i = seq[0] 
    j = seq[1]
    k = seq[2]
    not_proper = ca.if_else(i == k, False, True)
    # not_proper = False
    k = ca.if_else(ca.logic_not(not_proper), 6 - i - j, k)
    # Calculate the Levi-Civita symbol equivalent
    e = (i - j)*(j - k)*(k - i)/2
    # Compute a, b, c, d based on 'not_proper'
    a = ca.if_else(not_proper, q[0] - q[j], q[0])
    b = ca.if_else(not_proper, q[i] + q[k] * e, q[i])
    c = ca.if_else(not_proper, q[j] + q[0], q[j])
    d = ca.if_else(not_proper, q[k] * e - q[i], q[k]*e)
    
    # Calculate theta_plus and theta_minus
    theta_plus = ca.atan2(b, a)
    theta_minus = ca.atan2(d, c)
    
    # Compute theta_2 using acos
    theta_2 = ca.acos(2 * (a**2 + b**2) / (a**2 + b**2 + c**2 + d**2 + 1e-12) - 1)
    # print(theta_plus, theta_minus, theta_2)
    # Determine theta_1 and theta_3 based on theta_2
    theta_1 = ca.if_else(ca.fabs(theta_2) < 1e-12, 0, theta_plus - theta_minus)
    theta_3 = ca.if_else(ca.fabs(theta_2) < 1e-12, 2 * theta_plus - theta_1, theta_plus + theta_minus)
    
    theta_1 = ca.if_else(ca.logic_and(ca.fabs(theta_2) > ca.pi-1e-12,ca.fabs(theta_2) < ca.pi+1e-12), 0, theta_1)
    theta_3 = ca.if_else(ca.logic_and(ca.fabs(theta_2) > ca.pi-1e-12,ca.fabs(theta_2) < ca.pi+1e-12), 2*theta_minus + theta_1, theta_3)
    
    # Adjust theta_3 if 'not_proper'
    theta_3 = ca.if_else(not_proper, e * theta_3, theta_3)
    
    # Adjust theta_2 if 'not_proper'
    theta_2 = ca.if_else(not_proper, theta_2 - ca.pi/2, theta_2)
    
    # Wrap angles
    # theta_1 = ca.fmod(theta_1 + ca.pi, 2 * ca.pi) - ca.pi
    # theta_3 = ca.fmod(theta_3 + ca.pi, 2 * ca.pi) - ca.pi
    
    theta_1 = wrap_to_2_pi(theta_1)
    theta_2 = wrap_to_2_pi(theta_2)
    # theta_1 = ca.if_else(theta_1 < 0, theta_1 + 2 * ca.pi, theta_1)
    theta_3 = wrap_to_2_pi(theta_3)
    # theta_3 = ca.if_else(theta_3 < 0, theta_3 + 2 * ca.pi, theta_3)
    if not extrinsic:
        # reversal
        theta_1, theta_3 = theta_3, theta_1
    return ca.vertcat(theta_1, theta_2, theta_3)

def wrap_to_2_pi(angle):
    """
    Wrap an angle to the interval `[0, 2*pi]` using CasaDi.
    """
    return ca.fmod(angle, 2 * ca.pi)

def rot_matrix_to_quaternion(m):
    """
    Calculate a quaternion representation from a rotation matrix using CasaDi.

    Parameters:
    m (ca.[MX|SX|DM]): A 3x3 part of the rotation matrix.

    Returns:
    quaternion (ca.MX): `[w,x,y,z]`
    
    From: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    
    def _copysign_casadi(x, y):
        """Replicate the behavior of the standard library's math.copysign with CasADi."""
        return ca.if_else(y >= 0, ca.fabs(x), -ca.fabs(x))
    m00, m11, m22 = m[0, 0], m[1, 1], m[2, 2]
    w = ca.sqrt(ca.mmax(ca.vertcat(0, 1 + m00 + m11 + m22))) / 2
    x = ca.sqrt(ca.mmax(ca.vertcat(0, 1 + m00 - m11 - m22))) / 2
    y = ca.sqrt(ca.mmax(ca.vertcat(0, 1 - m00 + m11 - m22))) / 2
    z = ca.sqrt(ca.mmax(ca.vertcat(0, 1 - m00 - m11 + m22))) / 2
    x = _copysign_casadi(x, m[2, 1] - m[1, 2])
    y = _copysign_casadi(y, m[0, 2] - m[2, 0])
    z = _copysign_casadi(z, m[1, 0] - m[0, 1])
    return ca.vertcat(w,x,y,z)

def quat_to_rot_matrix(q):
    """
    
    """

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    
    return ca.vertcat(ca.horzcat(qw**2+qx**2-qy**2-qz**2, 2*qx*qy-2*qw*qz, 2*qx*qz+2*qw*qy),
                        ca.horzcat(2*qx*qy+2*qw*qz, qw**2-qx**2+qy**2-qz**2, 2*qy*qz-2*qw*qx),
                        ca.horzcat(2*qx*qz-2*qw*qy, 2*qy*qz+2*qw*qx, qw**2-qx**2-qy**2+qz**2))
    
rotation_z = lambda angle: ca.vertcat(ca.horzcat(ca.cos(angle), -ca.sin(angle), 0),
                                        ca.horzcat(ca.sin(angle), ca.cos(angle), 0),
                                        ca.horzcat(0, 0, 1))
rotation_y = lambda angle: ca.vertcat(ca.horzcat(ca.cos(angle), 0, ca.sin(angle)),
                                        ca.horzcat(0, 1, 0),
                                        ca.horzcat(-ca.sin(angle), 0, ca.cos(angle)))
rotation_x = lambda angle: ca.vertcat(ca.horzcat(1, 0, 0),
                                        ca.horzcat(0, ca.cos(angle), -ca.sin(angle)),
                                        ca.horzcat(0, ca.sin(angle), ca.cos(angle)))

def euler_derivatives_to_angular_velocity_matrix(order:str, variables:list):
    
    """
    Calculates the transformation matrix that transforms the euler angle derivatives to the angular velocity vector using CasADI.
    ```
    alpha = euler_angles
    
    T = euler_derivatives_to_angular_velocity_matrix('zyz',alpha)
    
    angular_velocity = T @ alpha_dot
    ```
    Intrisinc rotation, so rotations around the axes of the moving body.
    
    Example:
    ```
    psi = ca.SX.sym('psi')
    theta = ca.SX.sym('theta')
    phi = ca.SX.sym('phi')
    euler_derivatives_to_angular_velocity_matrix('zyz',[psi,theta,phi])
```
    """
    assert len(order) == 3
    assert order[0] != order[1] and order[1] != order[2]
    # assert len(variables) >= 3
    matrices = []
    vectors = []
    
    for i in range(0,3):
        # char = order[2-i]
        char = order[i]
        if char == 'x':
            vectors.append(ca.vertcat(1,0,0))
            matrices.append(rotation_x(variables[i]))
        if char == 'y':
            vectors.append(ca.vertcat(0,1,0))
            matrices.append(rotation_y(variables[i]))
        if char == 'z':
            vectors.append(ca.vertcat(0,0,1))
            matrices.append(rotation_z(variables[i]))
    return ca.horzcat(vectors[0],matrices[0]@vectors[1],matrices[0]@matrices[1]@vectors[2])

def analytical_jacobian_to_geometric_matrix(order:str, variables:list):
    """
    Returns `T_J`, such that `T_J @ J_a = J`, where `J_a` is the analytical Jacobian and `J` is the geometric Jacobian.
    
    The analytical jacobian uses the order `order` for the euler angles and the variables `variables`, which are the euler angles.
    """
    T = euler_derivatives_to_angular_velocity_matrix(order,variables)
    upper_rows = ca.horzcat(T,ca.DM.zeros(3,3))
    lower_rows = ca.horzcat(ca.DM.zeros(3,3),ca.DM.eye(3))
    return ca.vertcat(upper_rows,lower_rows)

def geometric_jacobian_to_analytical_matrix(order:str, variables:list):
    """
    Returns `T_J`, such that `T_J @ J = J_a`, where `J_a` is the analytical Jacobian and `J` is the geometric Jacobian.
    
    The analytical jacobian uses the order `order` for the euler angles and the variables `variables`, which are the euler angles.
    """
    raise NotImplementedError()
    T = euler_derivatives_to_angular_velocity_matrix(order,variables)
    upper_rows = ca.horzcat(T,ca.DM.zeros(3,3))
    lower_rows = ca.horzcat(ca.DM.zeros(3,3),ca.DM.eye(3))
    return ca.vertcat(upper_rows,lower_rows)

def rot_matrix_to_angle_axis(R):
    angle = ca.arccos(ca.fmin((ca.trace(R)-1)/2,1))
    axis = ca.if_else(ca.fabs(angle) < 1e-12,ca.vertcat(0,0,1),ca.vertcat(R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1])/(2*ca.sin(angle)))
    return angle,axis

def quaternion_to_angle_axis(q):
    """
    https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions
    """
    s = q[0]
    x = q[1:]
    angle = 2*ca.arccos(s)
    axis = ca.if_else(ca.fabs(angle) < 1e-12,ca.vertcat(0,0,0),x/ca.sin(angle/2))
    return angle,axis

def quat_dot_to_angular_velocity(q,q_dot):
    """
    Calculates the angular velocity vector from the quaternion derivative using CasADI.
    """
    H = ca.vertcat(ca.horzcat(-q[1], q[0], -q[3], q[2]),
                    ca.horzcat(-q[2], q[3], q[0], -q[1]),
                    ca.horzcat(-q[3], -q[2], q[1], q[0]))
    return 2*H@q_dot

def angular_velocity_to_quat_dot(q,w):
    """
    Calculates the quaternion derivative from the angular velocity vector using CasADI.
    """
    H = ca.vertcat( ca.horzcat(-q[1], q[0], -q[3], q[2]),
                    ca.horzcat(-q[2], q[3], q[0], -q[1]),
                    ca.horzcat(-q[3], -q[2], q[1], q[0]))
    return 1/2*H.T@w
def skew_symmetric(w):
    return ca.vertcat(ca.horzcat(0, -w[2], w[1]),
                    ca.horzcat(w[2], 0, -w[0]),
                    ca.horzcat(-w[1], w[0], 0))

def rodrigues_formula(w,theta):
    """
    Calculates Rodrigues' formula using CasADI, which is the rotation matrix for a rotation around the axis w with angle theta.
    """
    norm_w = ca.norm_2(w)
    axis = w/norm_w
    K = skew_symmetric(axis)
    return ca.DM.eye(3) + ca.sin(norm_w*theta)*K + (1-ca.cos(norm_w*theta))*K@K

def quat_conjugate(q):
    q_star = q.__copy__()
    q_star[1:] *= -1
    return q_star


def hamiltonian_product(quaternion0, quaternion1):
    shape = quaternion0.shape
    w0, x0, y0, z0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    w1, x1, y1, z1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]
    result_1 = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    result_2 = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    result_3 = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    result_4 = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    return ca.vertcat(result_1,result_2,result_3,result_4).reshape(shape)

def quat_times_vector(q, x):
    """
    Returns the quaternion action on the vector x that belongs to R^3.
    
    x' = q*x*q_conj
    """
    q_conj = quat_conjugate(q)
    return hamiltonian_product(hamiltonian_product(q, [0,x[0],x[1],x[2]], q_conj))[1:]

def quat_exponential_general(q,eps:float = 1e-12):
    """
    e^(q) where q is an element of H, the general quaternions.
    """
    w = q[0:1]
    v = q[1:]
    shape = q.shape
    angle = ca.norm_2(v)
    temp = ca.sin(angle)*v/angle
    result = ca.if_else(angle < eps, ca.vertcat(1,v[0],v[1],v[2]), ca.vertcat(ca.cos(angle),temp[0],temp[1],temp[2])).reshape(shape)
    return result*ca.exp(w)

def quat_exponential_pure(v,eps:float = 1e-12):
    """
    e^(v) where v is the imaginary part of a quaternion (i.e, the real part of the whole quaternion is zero)
    """
    shape = v.shape
    if shape[0] == 1:
        shape = (1,4)
    else:
        shape = (4,1)
    angle = ca.norm_2(v)
    u = v/angle
    temp = u*ca.sin(angle)
    result = ca.if_else(angle < eps, ca.vertcat(1.,0.,0.,0.), ca.vertcat(ca.cos(angle),temp[0],temp[1],temp[2]))
    return result.reshape(shape)

def quat_exponential_map_R3(q0,v,t):
    """
    Exponential map Exp(v) where v is in R^3.
    
    Quaternion Kinematics for the error-state Kalman Filter pg 20
    https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    """
    shape = q0.shape
    v = v/2
    return quat_exponential_map_H_p(q0,v,t).reshape(shape)

def quat_exponential_map_H_p(q0,v,t):
    """
    Exponential map Exp(v) where v is in H_p.
    
    Quaternion Kinematics for the error-state Kalman Filter pg 20
    https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    """
    shape = q0.shape
    V = v*t
    return hamiltonian_product(q0,quat_exponential_pure(V)).reshape(shape)




# testing
if __name__ == '__main__':
    import utils.my_casadi.math as ca_math
    import pydrake
    from pydrake.all import (
        MultibodyPlant,
        Simulator,
        DiagramBuilder,
        RollPitchYaw,
        SceneGraph,
        Quaternion
    )
    import casadi as ca
    dt = 0.0001
    builder =  DiagramBuilder()
    plant = MultibodyPlant(time_step=dt)
    scene_graph = SceneGraph()
    builder.AddSystem(scene_graph)
    builder.AddSystem(plant)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    sphere_instance = plant.AddModelInstance("floating_body")
    body_inertia = pydrake.multibody.tree.SpatialInertia(mass=1000.0, p_PScm_E=np.array([0., 0., 0.]), G_SP_E=pydrake.multibody.tree.UnitInertia.SolidSphere(0.1))
    body = plant.AddRigidBody("floating_body",sphere_instance, body_inertia)
    plant.Finalize()


    wx,wy,wz = 1,2,3
    w = np.array([wx,wy,wz])
    T = 2
    initial_rotation = RollPitchYaw(np.array([3.33,1.,2.]))
    initial_quaternion = initial_rotation.ToQuaternion().wxyz()
    initial_R_matrix = initial_rotation.ToRotationMatrix().matrix()

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context  = plant.GetMyContextFromRoot(diagram_context)
    initial_state = (np.concatenate((initial_quaternion,[0,0,0,wx,wy,wz,0,0,0])))
    plant_context.SetDiscreteState(initial_state)
    simulator = Simulator(diagram, diagram_context)

    simulator.AdvanceTo(T)

    result_quaternion = plant_context.get_discrete_state_vector().get_value()[0:4] / np.linalg.norm(plant_context.get_discrete_state_vector().get_value()[0:4])

    w_ca = ca.SX.sym('w',3)
    q_ca_1 = ca.SX.sym('q',4)
    q_ca_2 = ca.SX.sym('q',4)
    t_ca = ca.SX.sym('t')

    hamiltonian_product_ca = ca.Function('hamiltonian_product',[q_ca_1,q_ca_2],[ca_math.hamiltonian_product(q_ca_1,q_ca_2)])
    quaternion_dot_to_angular_velocity_ca = ca.Function('f',[q_ca_1,q_ca_2],[ca_math.quat_dot_to_angular_velocity(q_ca_1,q_ca_2)])
    angular_velocity_to_quaternion_dot_ca = ca.Function('f',[q_ca_1,w_ca],[ca_math.angular_velocity_to_quat_dot(q_ca_1,w_ca)])
    rodrigues_formula_ca = ca.Function('rodrigues_formula',[w_ca,t_ca],[ca_math.rodrigues_formula(w_ca,t_ca)])
    quat_exponential_map_R3_ca = ca.Function('quat_exponential_map_R3',[q_ca_1,w_ca,t_ca],[ca_math.quat_exponential_map_R3(q_ca_1,w_ca,t_ca)])
    quat_exponential_map_H_p = ca.Function('quat_exponential_map_H_p',[q_ca_1,w_ca,t_ca],[ca_math.quat_exponential_map_H_p(q_ca_1,w_ca,t_ca)])

    q_dot = plant.MapVelocityToQDot(plant_context, np.array([wx,wy,wz,0,0,0]))[0:4]

    print("w to qd")
    print("Calculated q_dot from w using angular_velocity_to_quaternion_dot_ca: " + str(angular_velocity_to_quaternion_dot_ca(result_quaternion,w).full()))

    print("Ground truth from drake" + str(q_dot))

    print("qd to w")
    print("Calculated q_dot from w using angular_velocity_to_quaternion_dot_ca: " + str(quaternion_dot_to_angular_velocity_ca(result_quaternion,q_dot).full()))

    print("Ground truth " + str(w))
    # 

    print("~"*100)
    print("Integration")
    print('Drake Ground Truth')
    print(Quaternion(result_quaternion))
    R = rodrigues_formula_ca(w,T)
    print('Calculated using rodrigues_formula_ca')
    print(Quaternion(R@initial_R_matrix))

    print("Using quat_exponential_map_R3_ca",quat_exponential_map_R3_ca(initial_quaternion,w,T))
    print("Using quat_exponential_map_H_p",quat_exponential_map_H_p(initial_quaternion,w/2,T))