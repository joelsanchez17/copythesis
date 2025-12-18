#adding this to the jupyter environment 
import sys
sys.path.append('/workspaces/master_thesis')
kjghfdhdsfh3 = 3
import os, importlib
import sys, pathlib
import psutil
import threadpoolctl
# Get the current process
import numba
# numba.config.THREADING_LAYER = 'forksafe'

sys.path += ["../diff_co_mpc/"]
workstation = True
# sys.path += [str(pathlib.Path(os.getcwd()) / '..' / "diff_co_mpc")]
if workstation:
    # LD_PRELOAD=<path>/libgomp.so
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libiomp5.so:'
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libmkl_def.so:/usr/lib/x86_64-linux-gnu/libmkl_avx2.so:/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libiomp5.so'
    os.environ["OMP_NUM_THREADS"] = "12"  #
    os.environ["MKL_NUM_THREADS"] = "12"  #    
    # os.environ['TORCH_LOGS'] = '+dynamic'
    os.environ["OMP_PLACES"] = "{12:23}"
    os.environ["OMP_PROC_BIND"] = "true"
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    p = psutil.Process()
    p.cpu_affinity(list(range(12,24)))
else:
    p = psutil.Process()
    # LD_PRELOAD=<path>/libgomp.so
    os.environ["OMP_NUM_THREADS"] = "3"  #
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREAD"] = "1"
    # TORCH_LOGS=dynamic
    os.environ['TORCH_LOGS'] = '+dynamic'
    os.environ["OMP_PLACES"] = "{0,2,4}"
    os.environ["OMP_PROC_BIND"] = "true"
    p.cpu_affinity([0,1,2,3,4,5])
import casadi as ca
from IPython.display import clear_output, display, SVG
from functools import partial
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
#====================================================
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"

import numpy as np
#====================================================

import torch
import itertools
import pathlib, os
from matplotlib import pyplot as plt
import time
import copy
import threadpoolctl

import importlib
from pydrake.all import (
    ModelInstanceIndex,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    StartMeshcat,
)
import threadpoolctl
from utils.my_casadi.misc import time_function, Jit, Compile


from utils.math.BSpline import BSpline

bspline = BSpline(np.array([[0, 0], [0.5, 0.5], [1, 1]]), 3)
bspline.fast_batch_evaluate(np.linspace(0,1,10))
bspline.fast_create_derivative_spline()

import utils.my_casadi.misc as ca_utils
from utils.my_casadi.misc import veccat

# from diff_co_mpc.casadi_custom import *
# from diff_co_mpc.global_planner import *
from mpc.planner_implementation.make import *
from mpc.optimisation import load_opt_info, DiffCoOptions, make_system
from misc.helper_functions import *

try:
    meshcat
    print("meshcat already defined", meshcat.web_url())
except NameError:
    meshcat = StartMeshcat()
try:
    ROS_INITIALIZED 
except:
    import sys
    sys.path += ["/opt/ros/noetic/lib/python3/dist-packages"]
    import rospy as ros
    ros.init_node('global_planner')
    ROS_INITIALIZED = True
    from actionlib import SimpleActionClient
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.msg import FollowJointTrajectoryAction, \
                                FollowJointTrajectoryGoal, FollowJointTrajectoryResult
TEMP_FOLDER = pathlib.Path(os.getcwd()) / '..' / "temp"
TEMP_FOLDER.mkdir(exist_ok=True)
CODEGEN_FOLDER = TEMP_FOLDER / "codegen"
CODEGEN_FOLDER.mkdir(exist_ok=True, parents=True)
yaml_path = pathlib.Path('..') / 'config'/ "plant_definition.yaml"

system = make_system(yaml_path, meshcat, CODEGEN_FOLDER)
complete_plant = system["complete_plant"]
robots = system["robots"]
viz_helper = system["visualization_helper"]
carried_object = system["carried_object"]
yaml = system["yaml"]

# EE_transform_1 = RigidTransform(
#     p=[0.1, 0, 0], rpy=RollPitchYaw([0, 0, np.pi])
# ).GetAsMatrix4()
# EE_transform_2 = RigidTransform(
#     p=[-0.0, 0, 0], rpy=RollPitchYaw([0, 0, 0])
# ).GetAsMatrix4()


def configuration_to_control_points(configuration, number_of_control_points):
    return np.tile(configuration.reshape(-1,1),(1,number_of_control_points))
def publish_trajectory(opt_data, lcm_subscription_handler, order = 3):
    message = lcmt_global_solve()
    message.utime = time.perf_counter_ns()
    message.bspline_robot_1.control_points = opt_data.result["robot_1_control_points"]
    message.bspline_robot_1.order = order
    message.bspline_robot_2.order = order
    message.bspline_robot_1.num_positions = opt_data.result["robot_1_control_points"].shape[0]
    message.bspline_robot_1.num_control_points = opt_data.result["robot_1_control_points"].shape[1]
    message.bspline_robot_2.control_points = opt_data.result["robot_2_control_points"]
    message.bspline_robot_2.num_positions = opt_data.result["robot_2_control_points"].shape[0]
    message.bspline_robot_2.num_control_points = opt_data.result["robot_2_control_points"].shape[1]
    message.bspline_object.control_points = opt_data.result["carried_object_control_points"]
    message.bspline_object.num_positions = opt_data.result["carried_object_control_points"].shape[0]
    message.bspline_object.num_control_points = opt_data.result["carried_object_control_points"].shape[1]
    lcm_subscription_handler.lcm.Publish("global_trajectory", message.encode())
def display_trajectory(opt_data,meshcat):
    bspline_1_col = BSpline(
        opt_data.result["robot_1_control_points"],
        opt_collision.order,
    )
    bspline_2_col = BSpline(
        opt_data.result["robot_2_control_points"],
        opt_collision.order,
    )
    bspline_obj_col = BSpline(
        opt_data.result["carried_object_control_points"],
        opt_collision.order,
    )
    

    meshcat.StartRecording(frames_per_second=30.0)

    for s in np.linspace(0,1,15):
        q_1 = bspline_1_col.evaluate(s)
        q_1 = np.concatenate([q_1,np.zeros(2)])
        q_2 = bspline_2_col.evaluate(s)
        q_2 = np.concatenate([q_2,np.zeros(2)])
        q_obj = bspline_obj_col.evaluate(s)
        
        viz_helper.set_position('robot_0',q_1)
        viz_helper.set_position('robot_1',q_2)
        viz_helper.set_position('carried_object',q_obj)
        

        viz_helper.diagram_context.SetTime(s)
        
        viz_helper.publish_diagram()

    meshcat.StopRecording()
    
    meshcat.PublishRecording()
def display_initial_guesses(parallel_data_object,show_all = False):
    meshcat.StartRecording(frames_per_second=30.0)
    ii = 0
    bspline_1_col = []
    bspline_2_col = []
    bspline_obj_col = []
    for i in range(parallel_data_object.map_size):
        bspline_1_col.append(BSpline(parallel_data_object.result['robot_1_control_points'][i],opt_collision.order))
        bspline_2_col.append(BSpline(parallel_data_object.result['robot_2_control_points'][i],opt_collision.order))
        bspline_obj_col.append(BSpline(parallel_data_object.result['carried_object_control_points'][i],opt_collision.order))
    for i in range(parallel_data_object.map_size):
        if parallel_data_object.within_bounds()[i] or show_all:
            for s in np.linspace(0,1,80):
                q_1 = bspline_1_col[i].evaluate(s)
                q_2 = bspline_2_col[i].evaluate(s)
                q_obj = bspline_obj_col[i].evaluate(s)
                q_1 = np.concatenate([q_1,np.zeros(2)])
                q_2 = np.concatenate([q_2,np.zeros(2)])
                
                
                viz_helper.set_position('robot_0',q_1)
                viz_helper.set_position('robot_1',q_2)
                viz_helper.set_position('carried_object',q_obj)              

                viz_helper.diagram_context.SetTime(s+ii)
                viz_helper.publish()
            ii+=1
            if parallel_data_object.within_bounds()[i]: print(ii)
    meshcat.StopRecording()
    meshcat.PublishRecording()
def collision_score_from_parallel_solves(num_samples,planner):
    def polyharmonic_kernel(
        x: torch.Tensor, y: torch.Tensor, alpha: int
    ) -> torch.Tensor:

        if alpha % 2 == 1:
            return torch.linalg.norm(x - y, axis=-1) ** alpha
        else:
            r = torch.linalg.norm(x - y, axis=-1)
            temp = (r**alpha) * torch.log(r)
            temp[torch.isnan(temp)] = 0.0

            return temp
    total_scores = {}
    for i in range(planner.parallel_solve_data.map_size):
        bspline_1 = (BSpline(planner.parallel_solve_data.result['robot_1_control_points'][i],opt_collision.order))
        bspline_2 = (BSpline(planner.parallel_solve_data.result['robot_2_control_points'][i],opt_collision.order))
        # bspline_obj_col = (BSpline(planner.parallel_solve_data.result['carried_object_control_points'][i],opt_collision.order))
        total_score = 0
        for s in np.linspace(0,1,num_samples):
            # print(s)
            # text= ''
            for robot in ['robot_1','robot_2']:
                for group_name in ['group_1','group_2','group_3','group_4']:
                    weights = torch.as_tensor(planner.lcm_subscription_handler.last_svm[robot][group_name][-1]['weights'])
                    support_vectors = torch.as_tensor(planner.lcm_subscription_handler.last_svm[robot][group_name][-1]['sv'])
                    polynomial_weights = torch.as_tensor(planner.lcm_subscription_handler.last_svm[robot][group_name][-1]['pol_weights']).squeeze()
                    q = torch.as_tensor(bspline_2.evaluate(s) if robot == 'robot_2' else bspline_1.evaluate(s))
                    col_model = planner.robot_1_collision_model if robot == 'robot_1' else planner.robot_2_collision_model
                    fk_q = col_model.forward_kinematics_groups_torch[group_name](q)
                    sample = fk_q.reshape(-1)
                    dist = polyharmonic_kernel(sample, support_vectors.T, 1)
                    score = torch.dot(weights.reshape(-1).to(torch.float64), dist.reshape(-1).to(torch.float64)) + polynomial_weights[0].reshape(-1).to(torch.float64) + torch.dot(polynomial_weights[1:].reshape(-1).to(torch.float64), sample.reshape(-1).to(torch.float64))
                    total_score += torch.clip(score,torch.tensor(0.),torch.inf)
                    # print(s,score)
                    # text += f'{group_name}: {score.item():.2f} '
                # text+= '\n'
            # print(text)
        total_scores[i] = total_score
    return total_scores
from pydrake.all import Sphere

# path = 'tensors.pth'
# loaded_tensors = torch.load(path)
# # Access the tensors
# obstacle_points = loaded_tensors['tensor1'].numpy()
# obstacle_radii = loaded_tensors['tensor2'].numpy()

# sphere_positions = obstacle_points
from multiprocessing import shared_memory
dtype = np.float32
array_shape = (10000, 3)
shared_memory_name = "point_cloud_shared_memory"

try:
    shm = shared_memory.SharedMemory(name=shared_memory_name, create=True, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
    shared_array = np.ndarray(array_shape, dtype=dtype, buffer=shm.buf)
    shared_array[:] = np.nan
    print("Created new shared memory.")
except FileExistsError:
    shm = shared_memory.SharedMemory(name=shared_memory_name)
    shared_array = np.ndarray(array_shape, dtype=dtype, buffer=shm.buf)
    print("Linked to existing shared memory.")
point_cloud = shared_array.copy()
point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1)]

point_cloud_size = point_cloud.shape[ 0 ]
obstacle_points = torch.as_tensor(point_cloud).to(device = 'cpu',dtype=torch.float32)
obstacle_radii = torch.tensor(0.04, dtype=torch.float32, device='cpu').expand(point_cloud_size, 1).to(device = 'cpu',dtype=torch.float32)
for i, q_sample in enumerate(obstacle_points):
    meshcat.SetObject (
        f"/obstacles/{i}", Sphere(obstacle_radii[i])
    )
    meshcat.SetProperty (
        f"/obstacles/{i}", "position", q_sample.tolist()
    )
class PointCloudViz:
    def __init__(self, meshcat):
        self.meshcat = meshcat
        self.shm = shared_memory.SharedMemory(name=shared_memory_name)
        self.point_cloud = np.ndarray(array_shape, dtype=dtype, buffer=self.shm.buf)

    def update(self):
        point_cloud = self.point_cloud.copy()
        point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1)]
        obstacle_radii = torch.tensor(0.04, dtype=torch.float32, device='cpu').expand(point_cloud.shape[0], 1).to(device = 'cpu',dtype=torch.float32).numpy()
        for i in range(1000):
            self.meshcat.Delete(f"/obstacles/{i}")
        for i, q_sample in enumerate(point_cloud):
            self.meshcat.SetObject(
                f"/obstacles/{i}", Sphere(obstacle_radii[i])
            )
            self.meshcat.SetProperty(
                f"/obstacles/{i}", "position", q_sample.tolist()
            )
point_cloud_viz = PointCloudViz(meshcat)
point_cloud_viz.update()


collision_options = {
    "print_time": True,
    "ipopt": {
        "print_level": 3,
        "hessian_approximation": "exact",
        "linear_solver": "mumps",
        "max_iter": 10000,
        "constr_viol_tol": 1e-4,
        "tol": 1e-4,
        "acceptable_tol": 1e-2,
        "acceptable_obj_change_tol": 1e-1,
        "max_wall_time": 10.,
        "warm_start_init_point": "yes",
        'warm_start_bound_push'  :     1e-9,
        'warm_start_bound_frac':       1e-9,
        'warm_start_slack_bound_frac': 1e-9,
        'warm_start_slack_bound_push': 1e-9,
        'warm_start_mult_bound_push' : 1e-9,
        'nlp_scaling_method':'none',
        'mu_strategy':'monotone',
        'mu_init':0.001,
    },
}





# collision_options = {
#     "print_time": True,
#     "ipopt": {
#         "print_level": 3,
#         "hessian_approximation": "exact",
#         "linear_solver": "ma57",
#         "max_iter": 10000,
#         # "print_timing_statistics":"yes",
#         "constr_viol_tol": 1e-4,
#         "tol": 1e-4,
#         "acceptable_tol": 1e-2,
#         "acceptable_obj_change_tol": 1e-1,
#         "max_wall_time": 10.,
#         "warm_start_init_point": "yes",
#         # 'warm_start_same_structure':'no',
#         'warm_start_bound_push'  :     1e-9,
#         'warm_start_bound_frac':       1e-9,
#         'warm_start_slack_bound_frac': 1e-9,
#         'warm_start_slack_bound_push': 1e-9,
#         'warm_start_mult_bound_push' : 1e-9,
#         'nlp_scaling_method':'none',
#         'mu_strategy':'monotone',
#         'mu_init':0.001,
#         # 'mu_init':0.1,
#         'ma57_pre_alloc':3.
#         # "fast_step_computation": "yes",
#     },
# }
collision_options_very_short = {
    "ipopt": {
        "print_level": 0,
        "hessian_approximation": "exact",
        "linear_solver": "mumps",
        "max_iter": 10000,
        "constr_viol_tol": 1e-4,
        "tol": 1e-4,
        "acceptable_tol": 1e-2,
        "acceptable_obj_change_tol": 1e-1,
        "max_wall_time": 0.25,
        "warm_start_init_point": "yes",
        'warm_start_bound_push'  :     1e-9,
        'warm_start_bound_frac':       1e-9,
        'warm_start_slack_bound_frac': 1e-9,
        'warm_start_slack_bound_push': 1e-9,
        'warm_start_mult_bound_push' : 1e-9,
        'nlp_scaling_method':'none',
        'mu_strategy':'monotone',
        'mu_init':0.001,
    },
}




# collision_options_very_short = {
#     "ipopt": {
#         "print_level": 0,
#         "hessian_approximation": "exact",
#         "linear_solver": "ma57",
#         "max_iter": 10000,
#         # "print_timing_statistics":"yes",
#         "constr_viol_tol": 1e-4,
#         "tol": 1e-4,
#         "acceptable_tol": 1e-2,
#         "acceptable_obj_change_tol": 1e-1,
#         "max_wall_time": 0.25,
#         "warm_start_init_point": "yes",
#         # 'warm_start_same_structure':'no',
#         'warm_start_bound_push'  :     1e-9,
#         'warm_start_bound_frac':       1e-9,
#         'warm_start_slack_bound_frac': 1e-9,
#         'warm_start_slack_bound_push': 1e-9,
#         'warm_start_mult_bound_push' : 1e-9,
#         'nlp_scaling_method':'none',
#         'mu_strategy':'monotone',
#         'mu_init':0.001,
#         # 'mu_init':0.1,
#         'ma57_pre_alloc':3.
#         # "fast_step_computation": "yes",
#     },
# }

num_samples = 10
num_control_points = 20
order = 3

import importlib
import mpc
import mpc.planner_implementation.make

importlib.reload(mpc.optimisation)
importlib.reload(mpc.planner_implementation.make)
from mpc.optimisation import *

# from diff_co_mpc.mpc import make_planner_with_collision, make_planner_with_no_collision
from mpc.planner_implementation.make import *

diff_co_options = DiffCoOptions.from_yaml(pathlib.Path(yaml_path))

gaze_options = VisionOptions.from_yaml(pathlib.Path(yaml_path))

opt_with_collision_compile_path = CODEGEN_FOLDER / "nlp_with_collision_vision"
opt_with_collision_compile_path.mkdir(exist_ok=True, parents=True)

# the flag I changed 
rebuild_col = False

if rebuild_col:
    opt_collision = make_planner_with_collision(
        robots,
        carried_object,
        num_samples,
        num_control_points,
        order,
        diff_co_options,
        gaze_options,
        collision_options,
        opt_with_collision_compile_path,
    )
else:
    opt_collision = load_opt_info(
        opt_with_collision_compile_path,
        robots,
        carried_object,
        num_control_points,
        order,
        diff_co_options,
        gaze_options,
        collision_options,
    )
opt_collision_very_short = load_opt_info(
    opt_with_collision_compile_path,
    robots,
    carried_object,
    num_control_points,
    order,
    diff_co_options, 
    gaze_options,
    collision_options_very_short,
)
from diff_co_lcm import lcmt_gaze_polytopes,lcmt_pose
import mpc.gaze_functions as gaze


def meshcat_arrow(meshcat, path,position, direction, size, head_size ):
    p1 = np.zeros((3,1))
    x_axis = np.array([1.,0,0]).reshape(-1,1)
    y_axis = np.array([0,1.,0]).reshape(-1,1)
    z_axis = np.array([0,0,1.]).reshape(-1,1)
    rotation_matrix = RotationMatrix.MakeFromOneVector(direction,0)
    transform = RigidTransform(p=position, R=rotation_matrix)
    p2 = p1 + x_axis*size
    line_head_1_a = p1 + y_axis*head_size + x_axis*(size-head_size)
    line_head_1_b = p1 + z_axis*head_size + x_axis*(size-head_size)
    line_head_2_a = p1 - y_axis*head_size + x_axis*(size-head_size)
    line_head_2_b = p1 - z_axis*head_size + x_axis*(size-head_size)
    start = p1
    end = p2
    meshcat.SetLineSegments(path,start,end,1)
    vertices = np.hstack([p2,line_head_1_b,line_head_2_a,line_head_2_b,line_head_1_a])
    faces = np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1],[1,2,3],[1,3,4]]).T
    meshcat.SetTriangleMesh(path+"/head",vertices,faces)
    meshcat.SetTransform(path,transform)
def my_handler(*args):
    global polytopes,line_point_,transform_masks,ik_mask,iks_robot_1,polyharmonic_weights,polyharmonic_sv,polynomial_weights
    # print(args[0].decode())
    t0  = time.perf_counter()
    msg = lcmt_gaze_polytopes.decode(args[0])
    
    vertices = np.asarray(msg.vertices)
    simplices = np.asarray(msg.simplices)
    indices = np.asarray(msg.indices)
    for i in range(30):
        meshcat.Delete(f'/polytope_{i}')
    polytopes  = []

    placing_spot_pose = RigidTransform(planner.lcm_subscription_handler.placing_spot_pose)

    pose_gaze_samples,iks_robot_1,transform_masks, ik_mask = gaze.get_samples_around_placing_spot(sampling_size_around_spot,num_samples_spot,num_samples_gaze,num_samples_gaze_turn,placing_spot_pose,q7_guesses,elevation,robot_base_pose_1)
    line_point_ = pose_gaze_samples[:,:3,3].astype(np.float64)
    line_direction_ = pose_gaze_samples[:,:3,2].astype(np.float64)
    for i,(a,b,c,d) in enumerate(indices):
        polytope_vertices= vertices[a:b]

        polytope_simplices = simplices[c:d]
        meshcat.SetTriangleMesh(f'/polytope_{i}',polytope_vertices.T,polytope_simplices.T)

        polytopes.append( { 'vertices': vertices[a:b], 'simplices': simplices[c:d] } )
        mean_vertices = np.mean(polytope_vertices,axis=0)
        if np.linalg.norm(mean_vertices-placing_spot_pose.translation()) > 0.5:
            continue
        polytope_vertices_ = polytope_vertices.astype(np.float64)
        faces = polytope_vertices[polytope_simplices]
        faces_ = np.array(faces).astype(np.float64)



        intersects = gaze.line_intersects_with_polytope_batch(polytope_vertices_,faces_, line_point_[transform_masks.squeeze()], line_direction_[transform_masks.squeeze()])
        intersects_ = np.ones(transform_masks.shape[0],dtype=bool)
        intersects_[transform_masks.squeeze()] = intersects
        transform_masks = np.logical_and(transform_masks,~intersects_.reshape(-1,1))

    where_not_nan = np.nonzero(ik_mask[transform_masks.squeeze()])
    unique_rows, indices = np.unique(where_not_nan[0], return_index=True)
    unique_gazes = iks_robot_1[transform_masks.squeeze()][where_not_nan][indices]
    support_vectors = torch.from_numpy(unique_gazes).to(torch.float32)
    Y_support_vectors  = torch.ones(unique_gazes.shape[0])
# robots[0].vision_model.forward_kinematics_groups_torch['group_1']
# 
    # fk_support_vectors = robots[0].vision_model.forward_kinematics_groups_torch['group_1'](support_vectors).squeeze()
    fk_sv, K_rg = kernel_rg(support_vectors,2.,0.1)
    fk_sv = fk_sv.view(support_vectors.shape[0],-1)
    # mock_weights = torch.ones((fk_sv.shape[0],1))
    # Y_support_vectors = torch.vmap(lambda sv,svv, w: kernel_gaze(sv,svv)@w, in_dims = (0,None,None))(fk_support_vectors.view(fk_support_vectors.shape[0],-1),fk_support_vectors.view(fk_support_vectors.shape[0],-1),mock_weights)
    Y_support_vectors = K_rg.sum(dim=1)
    Y_support_vectors /= (Y_support_vectors).max()
    # Y_support_vectors = torch.ones(Y_support_vectors.shape[0])
    _, K_ph = kernel_ph(support_vectors,1,None)
    # print(fk_sv[0],Y_support_vectors[0])
    # polyharmonic_weights = torch.linalg.solve(K_ph, Y_support_vectors.to(torch.float32))
    N, d = fk_sv.shape
    B = torch.cat([torch.ones((N, 1),device=support_vectors.device), fk_sv], dim=1)
    top = torch.cat([K_ph, B], dim=1)
    bottom = torch.cat([B.T, torch.zeros((d+1, d+1), device=support_vectors.device)], dim=1)
    M = torch.cat([top, bottom], dim=0)
    f = torch.cat([Y_support_vectors.view(-1,1), torch.zeros((d+1, 1), device=support_vectors.device)])
    try:
        polyharmonic_weights = torch.linalg.solve(M, f)
    except:
        print('solve failed, doing pseudoinverse')
        M_inv = torch.linalg.pinv(M)
        polyharmonic_weights = M_inv @ f
    support_vectors = support_vectors.to(torch.float64)
    # K_ph = torch.vmap(polyharmonic_kernel, in_dims = (0,None))(fk_support_vectors.view(fk_support_vectors.shape[0],-1),fk_support_vectors.view(fk_support_vectors.shape[0],-1))
    # polyharmonic_weights = torch.linalg.solve(K_ph, Y_support_vectors.to(torch.float32))
    # polyharmonic_weights = polyharmonic_weights - polyharmonic_weights.mean()
    polyharmonic_sv = fk_sv
    print('time',time.perf_counter()-t0)
    is_placing_spot_hidden = msg.is_placing_spot_hidden
    num_support_vectors = support_vectors.shape[0]
    polynomial_weights = polyharmonic_weights[num_support_vectors :].to(torch.float64)
    polyharmonic_weights = polyharmonic_weights[:num_support_vectors].to(torch.float64)
    # q1 = support_vectors[0]
    # (fk_S,fk_SV), K = kernel_ph([(q1).view(1,-1),support_vectors],1,None)
    # print(K,polyharmonic_weights,polynomial_weights,fk_S)
    # score = (K.to(torch.float64)@polyharmonic_weights.view(-1,1)).view(-1) + polynomial_weights[0].reshape(-1).to(torch.float64) + torch.dot(polynomial_weights[1:].reshape(-1).to(torch.float64), fk_S.reshape(-1).to(torch.float64))
    # print(fk_S,score)
    # asdfdfssd
    # for i in range(2000): meshcat.Delete(f"/gaze/gaze_direction_{i}")
    # for i in range(2000): meshcat.Delete(f"/gaze/gaze_direction_camera_{i}")
    # for i in range(2000): meshcat.Delete(f"/gaze/gaze_direction_gradient_{i}")
    # for i in range(2000): meshcat.Delete(f"/gaze/gaze_direction_gradient_{i}_2")
    # for i in range(2000): meshcat.Delete(f"/gaze/gaze_direction_gradient_{i}_line")
    # for i in range(2000): meshcat.Delete(f"/gaze/gaze_direction_gradient_{i}")
    # for i in range(2000): meshcat.Delete(f"/gaze/gaze_direction_gradient_{i}_b")
    # for j in range(0,100):
    #     print('j',j)
    #     print()
    #     print()
    #     print()
    #     print()
    #     for i,q1, in enumerate(unique_gazes):
    #         if i+j*unique_gazes.shape[0] > 200:
    #             print("max samples reached")
    #             break
    #         q1 = q1 + np.random.randn(*q1.shape)*(j/20+0.1)
    #         q1 = np.concatenate([q1,np.zeros(2)])
    #         # viz_helper.set_position('robot_0',q1)
    #         # fk = robots[0].forward_kinematic_gaze(q1).full()
    #         fk = robots[0].vision_model.forward_kinematics_groups_casadi['group_1'](q1).full()
    #         z = fk[0,:] - fk[1,:]
    #         rotation = RotationMatrix.MakeFromOneVector(z,2)
            
    # # def svm_callback(self, msg, robot_name, group_name):

    # #     msg = lcmt_support_vector.decode(msg)

    # #     support_vectors_1 = msg.support_vectors
    # #     weights_1 = msg.weights[: msg.num_support_vectors]
    # #     # constant_1 = msg.weights_1[msg.num_support_vectors_1]
    # #     # pol_weights_1 = msg.weights_1[msg.num_support_vectors_1 + 1 :]
    # #     pol_weights_1 = msg.weights[msg.num_support_vectors :]
    # #     # print(msg.weights)
    # #     sv_1 = np.array(support_vectors_1).T
    # #     w_1 = np.array(weights_1).T
    # #     # constant_1 = np.array(constant_1).T
    # #     pols_1 = np.array(pol_weights_1).T
    # #     self.last_svm[robot_name][group_name].append(
    # #         SVM(weights=w_1, sv=sv_1, pol_weights=pols_1)
    # #     )

    #         # v = torch.from_numpy(fk).to(torch.float32).view(-1)
    #         # v.requires_grad = True
    #         # score = polyharmonic_kernel(v,polyharmonic_sv.view(polyharmonic_sv.shape[0],-1))@polyharmonic_weights
    #         # score = kernel_ph([torch.from_numpy(q1),polyharmonic_sv],2,None)[1]@polyharmonic_weights
    #         # score = kernel_ph([torch.from_numpy(q1).view(1,-1),support_vectors],2,None)[1]@polyharmonic_weights
    #         (fk_S,fk_SV), K = kernel_ph([torch.from_numpy(q1).view(1,-1),support_vectors],2,None)
    #         fk_S.requires_grad = True
    #         score = (K.to(torch.float64)@polyharmonic_weights.view(-1,1)).view(-1) + polynomial_weights[0].reshape(-1).to(torch.float64) + torch.dot(polynomial_weights[1:].reshape(-1).to(torch.float64), fk_S.reshape(-1).to(torch.float64))
    #         print(score)
    #         sample_index = i+j*unique_gazes.shape[0]
            
    #         # if score > 0:
    #         #     meshcat.SetObject(f"/gaze/gaze_direction_gradient_{sample_index}",Sphere(score*0.01))
    #         #     meshcat.SetProperty(f"/gaze/gaze_direction_gradient_{sample_index}",'position',fk_S.view(-1,3)[0].detach().numpy())
    #         #     meshcat.SetObject(f"/gaze/gaze_direction_gradient_{sample_index}_2",Sphere(score*0.01))
    #         #     meshcat.SetProperty(f"/gaze/gaze_direction_gradient_{sample_index}_2",'position',fk_S.view(-1,3)[1].detach().numpy())
    #         #     meshcat.SetProperty(f"/gaze/gaze_direction_gradient_{sample_index}", 'color', [1,0,0,1])
    #         # continue
    #         score.backward()


    #         gradient = fk_S.grad.view(-1,3)
    #         direction_EE = gradient[1]
    #         arrow_size_EE = torch.linalg.norm(gradient[1])/40
    #         direction_8 = gradient[0]
    #         arrow_size_8 = torch.linalg.norm(gradient[0])/40
            

            
    #         meshcat_arrow(meshcat,f"/gaze/gaze_direction_gradient_{sample_index}",fk[1,:],direction_EE.numpy(),arrow_size_EE.numpy(),arrow_size_EE.numpy()/10)
    #         meshcat_arrow(meshcat,f"/gaze/gaze_direction_gradient_{sample_index}_2",fk[1,:] + 0.05*(fk[0,:] - fk[1,:]),direction_8.numpy(),arrow_size_8.numpy(),arrow_size_8.numpy()/10)
    #         meshcat.SetLine(f'/gaze/gaze_direction_gradient_{sample_index}_line',np.hstack([fk[1,:].reshape(3,1), (fk[1,:] + 0.05*(fk[0,:] - fk[1,:])).reshape(3,1)]))
    #         time.sleep(0.002)
    #         meshcat.SetProperty(f"/gaze/gaze_direction_gradient_{sample_index}", 'color', [1,0,0,1])
    #         meshcat.SetProperty(f"/gaze/gaze_direction_gradient_{sample_index}_2", 'color', [0,1,0,1])
    #         meshcat.Flush()
    #         # viz_helper.publish()
    #         time.sleep(0.01)
    
    # if is_placing_spot_hidden:
    #     meshcat.SetProperty(f'/placing_spot','color',[1,0,0,1])
    # else:
    #     meshcat.SetProperty(f'/placing_spot','color',[0,1,0,1])

    # for i,(pose, has_ik) in enumerate(zip(pose_gaze_samples, transform_masks)):

    #     rotation = RotationMatrix(pose[:3,:3])
    #     position = pose[:3,3]
    #     meshcat.SetProperty(f'/samples/sample_{i}','visibility',True)
    #     meshcat.SetTransform(f"/samples/sample_{i}", RigidTransform(p=position,R = rotation))
    #     if has_ik:
    #         meshcat.SetProperty(f"/samples/sample_{i}", "color",[0,1,0,0.5])
    #     else:
    #         meshcat.SetProperty(f"/samples/sample_{i}", "color",[1,1,1,0.5])
    # for i in range(600-i):
    #     meshcat.SetProperty(f'/samples/sample_{i}','visibility',False)
    # dagdsgds

# kernel_gaze = ForwardKinematicKernel(RationalQuadraticKernel(alpha = 2,length_scale = 0.05),robots[0].forward_kinematic_gaze)


from pydrake.all import DrakeLcm
lcm = DrakeLcm()
import pydrake.geometry as pydgeo
COMMAND_HZ = 20
lcm = DrakeLcm()
subscription = lcm.Subscribe("gaze_polytopes", my_handler)
polytopes = []
sampling_size_around_spot = 0.01
num_samples_spot = 50
num_samples_gaze = 2 #for each spot sample
num_samples_gaze_turn = 1 #for each gaze sample, rotate around z axis, set to one to not make K_ph singular
num_q7s = 7
q7_guesses = np.linspace(robots[0].plant.GetPositionLowerLimits()[6],robots[0].plant.GetPositionUpperLimits()[6],num_q7s)
robot_base_pose_1 = np.linalg.inv(robots[0].plant.GetFrameByName('panda_link0').CalcPose(robots[0].plant.CreateDefaultContext(),robots[0].plant.GetFrameByName('world')).GetAsMatrix4())
placing_spot_pose = RigidTransform(p=[0.1, -0.4, 0.0],rpy=RollPitchYaw([0, 0, np.pi/4]))
elevation = [0.5,0.55]
from diff_co.geometrical_model import Kernel
kernel_rg = Kernel(robots[0].vision_model.forward_kinematics_groups_torch['group_1'],'','rational_quadratic')
kernel_ph = Kernel(robots[0].vision_model.forward_kinematics_groups_torch['group_1'],'','polyharmonic')
# subscription
time.sleep(1)
# lcm.HandleSubscriptions(0)
# try:
#     while True:
#         # lcm.Publish(channel="camera_pose", buffer = state.encode())
#         lcm.HandleSubscriptions(0)
#         time.sleep(1)
# except KeyboardInterrupt:
#     pass
import mpc.planner_implementation.lcm_handler
import mpc.planner_implementation.ros_handler
import mpc.planner_implementation.helper
import mpc.planner_implementation.helper_mixin
import mpc.helper_functions
importlib.reload(mpc.planner_implementation.lcm_handler)
importlib.reload(mpc.planner_implementation.ros_handler)
importlib.reload(mpc.planner_implementation.helper_mixin)
importlib.reload(mpc.planner_implementation.helper)
importlib.reload(mpc.helper_functions)
from mpc.planner_implementation.helper import Planner
from mpc.planner_implementation.lcm_handler import VisionParameters
from mpc.planner_implementation.ros_handler import ROSHandler
ros_handler = ROSHandler(None)
lcm_subscription_handler_configuration = VisionParameters(
    sampling_size_around_spot = 1,
    num_samples_spot=1,
    num_samples_gaze=1,
    num_samples_gaze_turn=1,
    q7_guesses=1,
    elevation=1,
    robot_base_pose_1= 1,
    polyharmonic_kernel = None,
    kernel_gaze = None,
)
planner = Planner(opt_collision, 
                  meshcat,
                    lcm_subscription_handler_configuration = lcm_subscription_handler_configuration, 
                    ros_handler = ros_handler, 
                    viz_helper = viz_helper,
                    num_parallel_plans = 12
)
planner.set_plan_grasping_bounds(np.array([0.00,0.1]),np.array([-0.1,-0.00]))
planner.set_costs(vision_cost = 0, duration_cost = 0.001, acceleration_cost = 0.001, manipulability_cost = .0001, replan_connection_cost = 1.,slack_cost_weight = -0.1)
planner.set_time_bounds_plan(0.5,20.)
planner.set_velocity_scaling(0.1)
planner.parallel_solve_data.set_initial_guess('duration', np.array(2.))
planner.initial_plan_data.set_initial_guess('duration', np.array(2.))

planner.EE_transform_1 = RigidTransform(
    p=[0.1, 0, 0], rpy=RollPitchYaw([0, 0, 0])
).GetAsMatrix4()
planner.EE_transform_2 = RigidTransform(
    p=[-0.1, 0, 0], rpy=RollPitchYaw([0, 0, np.pi])
).GetAsMatrix4()
def set_parallel_initial_guesses(self,num_q7s,initial_pose,end_pose):
    num_initial_guesses = self.parallel_solve_data.map_size
    q1 = self.robot_1_inverse_kinematics.casadi_obj_IK_compiled(initial_pose,np.linspace(-np.pi,np.pi,num_q7s).reshape(1,-1),np.array([0.]*7),self.EE_transform_1)
    q1 = q1.full().reshape(-1,7)
    q1 = q1[~np.isnan(q1).any(axis=1)]

    q2 = self.robot_2_inverse_kinematics.casadi_obj_IK_compiled(initial_pose,np.linspace(-np.pi,np.pi,num_q7s).reshape(1,-1),np.array([0.]*7),self.EE_transform_2)
    q2 = q2.full().reshape(-1,7)
    q2 = q2[~np.isnan(q2).any(axis=1)]
    q1_combinations = q1[np.arange(q1.shape[0]).repeat(q2.shape[0])]
    q2_combinations = q2[np.tile(np.arange(q2.shape[0]),q1.shape[0])]
    indices = np.random.randint(0,q2_combinations.shape[0],num_initial_guesses)
    q_1_start = q1_combinations[indices]
    q_2_start = q2_combinations[indices]
    assert q_1_start.size > 0
    assert q_2_start.size > 0
    obj_start = np.concatenate((RotationMatrix(initial_pose[:3,:3]).ToQuaternion().wxyz(),initial_pose[:3,3]  + np.array([0,0,0.1])))
    obj_end = np.concatenate((RotationMatrix(end_pose[:3,:3]).ToQuaternion().wxyz(),end_pose[:3,3] + np.array([0,0,0.1])))
    self.parallel_solve_data.set_initial_guess('carried_object_control_points', np.linspace(obj_start,obj_end,self.num_control_points_mpc).T)
    for i,(q1,q2) in enumerate(zip(q_1_start,q_2_start)):
        self.parallel_solve_data.set_initial_guess('robot_1_control_points', configuration_to_control_points(q1, self.num_control_points_mpc),i)
        self.parallel_solve_data.set_initial_guess('robot_2_control_points', configuration_to_control_points(q2, self.num_control_points_mpc),i)
planner.set_parallel_initial_guesses = partial(set_parallel_initial_guesses,planner)
time.sleep(2)

planner.parallel_solve_data.x0[:] = 0.
planner.parallel_solve_data.lam_g0[:] = 0.
planner.parallel_solve_data.lam_x0[:] = 0.
num_q7s = 20
planner.parallel_solve_data.set_initial_guess('duration', np.array(2.))
planner.set_parallel_initial_guesses(num_q7s,planner.lcm_subscription_handler.initial_pose,planner.lcm_subscription_handler.placing_spot_pose)
planner.set_collision_bounds(-np.inf,np.inf)


with threadpoolctl.threadpool_limits(limits={'blas':1,'openmp':12}):
    planner.parallel_plans()