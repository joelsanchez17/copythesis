#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    meshcat
    assert False
except NameError:

    pass


# In[ ]:





# In[2]:


import os, importlib
import psutil
p = psutil.Process()
p.cpu_affinity([6,7,8,9,10,11])

os.environ['OMP_NUM_THREADS'] = '6'
os.environ['OMP_PLACES'] = '{6:11}'
os.environ['OMP_PROC_BIND'] = 'TRUE'
# os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libiomp5.so:'
# os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/local/lib/python3.10/dist-packages/torch/lib/libgomp-a34b3233.so.1'
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libiomp5.so:
# os.environ['MKL_THREADING_LAYER'] = 'gnu'


# In[ ]:


import copy
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import torch
torch._dynamo.config.capture_func_transforms=True
from scipy.spatial.transform import Rotation as scipy_R
import sys

sys.path += [".."]
sys.path += ["/opt/ros/noetic/lib/python3/dist-packages"]
sys.path += [str(pathlib.Path(os.getcwd()) / '..' / "diff_co_mpc")]
import rospy as ros
ros.init_node('camera_processing')
# import rospy as ros
import numpy as np

# from actionlib import SimpleActionClient
# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from control_msgs.msg import FollowJointTrajectoryAction, \
#                              FollowJointTrajectoryGoal, FollowJointTrajectoryResult
# ros.init_node('ros_point_cloud_svm')
import cProfile
import einops
import importlib
import open3d.core as o3c
import open3d as o3d

from projects.refactor_mpb.multibody_wrapper import MultiBodyPlantWrapper

import diff_co_mpc.diff_co as diff_co
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


    
from torch.export import export
torch._dynamo.config.capture_func_transforms=True




import casadi as ca

from pydrake.all import DiagramBuilder, LogVectorOutput,LcmSubscriberSystem,VideoWriter,PixelType,LcmInterfaceSystem,LeafSystem,Value,Image,ImageDepth32F, PySerializer,ImageRgba8U
import numpy as np
# sys.path.append('/workspaces/toys/projects/thesis/cooperative_diff_co_mpc/8/temp/codegen/robot_1/collision_geometry_positions/pytorch')
# sys.path.append('/workspaces/toys/projects/thesis/cooperative_diff_co_mpc/8/temp/codegen/robot_2/collision_geometry_positions/pytorch')

from pydrake.perception import DepthImageToPointCloud
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import (
    CameraInfo,
)

import pydrake.geometry as pydgeo
from pydrake.all import (
    LeafSystem,
    StartMeshcat,
    AbstractValue, BaseField,PixelType,Rgba, DrakeLcm, LcmImageArrayToImages,ImageDepth16U,LcmInterfaceSystem,ImageIo,Value,PointCloud,ImageDepth32F,ImageToLcmImageArrayT,LcmPublisherSystem,ImageRgba8U
)


from drake import lcmt_robot_state, lcmt_point_cloud, lcmt_image, lcmt_image_array
from mpc.optimisation import make_system
# from diff_co_mpc.global_planner import *
# from diff_co_mpc.plant import plant_from_yaml

from diff_co_mpc.diff_co_lcm import lcmt_pose,lcmt_support_vector,lcmt_gaze_polytopes,lcmt_global_solve
try:
    meshcat
    print("meshcat already defined",meshcat.web_url())
except NameError:
    meshcat = StartMeshcat()

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


# In[ ]:


import importlib
import diff_co_mpc.point_cloud.leafsystem
importlib.reload(diff_co_mpc.point_cloud.leafsystem)
import diff_co_mpc.misc.leafsystem
importlib.reload(diff_co_mpc.misc.leafsystem)

from diff_co_mpc.point_cloud.leafsystem import PreprocessPointCloud
from diff_co_mpc.point_cloud.point_cloud_processing import Message, MessageType
from diff_co_mpc.misc.leafsystem import CustomLCMToImages, TrajectoryPlotter,PoseLCMToTransform,RobotConfigurationToCameraPose,RealsenseReader,ROSJointState
realsense_intrinsics = CameraInfo(
    width=640,
    height=480,
    fov_y=1.0,
)
device = 'cpu'
voxel_size = 0.04
num_samples = 6000
max_iterations_initialization = 80000

max_iterations_update = 160000
lcm = DrakeLcm()
builder = DiagramBuilder()


simulation = True
dual_panda = True


lcm_sys = builder.AddSystem(LcmInterfaceSystem(lcm))

meshcat.SetProperty("/drake/carried_object",'visible',False)
for i in range(10):
    meshcat.SetProperty("/drake/obstacle_{i}",'visible',False)
robot_camera_depth_in_EE = np.array([[-0.00380074, -0.99995455, -0.00878683,  0.08132166],
       [ 0.99953527, -0.00406475,  0.03022601, -0.06282357],
       [-0.03026036, -0.00866787,  0.99950444, -0.073011  ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
covariance = 0.005
prob_threshold = 0.9
final_pose_subscriber = builder.AddSystem(LcmSubscriberSystem.Make("final_pose",lcm =lcm_sys,lcm_type=lcmt_pose,use_cpp_serializer=False))
lcm_final_pose_to_transform = builder.AddSystem(PoseLCMToTransform())


global_trajectory_subscriber = builder.AddSystem(LcmSubscriberSystem.Make("global_trajectory",lcm =lcm_sys,lcm_type=lcmt_global_solve,use_cpp_serializer=False))


if simulation:
    camera_intrinsics = ( realsense_intrinsics.focal_x(), realsense_intrinsics.focal_y(), realsense_intrinsics.center_x(), realsense_intrinsics.center_y() )
    point_cloud_filter = builder.AddSystem(PreprocessPointCloud(margin = 0.04,robots = robots,device = device, voxel_size=voxel_size,camera_1_intrinsics=camera_intrinsics,camera_2_intrinsics=camera_intrinsics, meshcat = meshcat, CODEGEN_FOLDER=CODEGEN_FOLDER))
    # depth_to_point_cloud_1 = builder.AddSystem(DepthImageToPointCloud(camera_info=realsense_intrinsics, fields= BaseField.kXYZs, pixel_type=PixelType.kDepth32F))
    # depth_to_point_cloud_2 = builder.AddSystem(DepthImageToPointCloud(camera_info=realsense_intrinsics, fields= BaseField.kXYZs, pixel_type=PixelType.kDepth32F))

    camera_pose_subscriber = builder.AddSystem(LcmSubscriberSystem.Make("camera_pose",lcm =lcm,lcm_type=lcmt_pose,use_cpp_serializer=False,wait_for_message_on_initialization_timeout = 0))
    camera_pose_subscriber_2 = builder.AddSystem(LcmSubscriberSystem.Make("camera_pose_2",lcm =lcm,lcm_type=lcmt_pose,use_cpp_serializer=False,wait_for_message_on_initialization_timeout = 0))
    lcm_camera_1_pose_to_transform = builder.AddSystem(PoseLCMToTransform())
    lcm_camera_2_pose_to_transform = builder.AddSystem(PoseLCMToTransform())
    lcm_image_subscriber = builder.AddSystem(LcmSubscriberSystem.Make("simulation_depth_image",lcm =lcm,lcm_type=lcmt_image_array,use_cpp_serializer=False,wait_for_message_on_initialization_timeout = 0))
    lcm_image_subscriber_2 = builder.AddSystem(LcmSubscriberSystem.Make("simulation_depth_image_2",lcm =lcm,lcm_type=lcmt_image_array,use_cpp_serializer=False,wait_for_message_on_initialization_timeout = 0))
    lcm_to_images_1 = builder.AddSystem(CustomLCMToImages())
    lcm_to_images_2 = builder.AddSystem(CustomLCMToImages())
    builder.Connect(lcm_image_subscriber.get_output_port(),lcm_to_images_1.get_input_port())
    builder.Connect(lcm_image_subscriber_2.get_output_port(),lcm_to_images_2.get_input_port())
    # builder.Connect(lcm_to_images_1.depth_image_output_port,depth_to_point_cloud_1.depth_image_input_port())
    # builder.Connect(lcm_to_images_2.depth_image_output_port,depth_to_point_cloud_2.depth_image_input_port())
    builder.Connect(camera_pose_subscriber.get_output_port(),lcm_camera_1_pose_to_transform.get_input_port())
    builder.Connect(camera_pose_subscriber_2.get_output_port(),lcm_camera_2_pose_to_transform.get_input_port())
    # builder.Connect(lcm_camera_1_pose_to_transform.get_output_port(),depth_to_point_cloud_1.camera_pose_input_port())
    # builder.Connect(lcm_camera_2_pose_to_transform.get_output_port(),depth_to_point_cloud_2.camera_pose_input_port())
    builder.Connect(lcm_camera_1_pose_to_transform.get_output_port(),point_cloud_filter.camera_pose_1_input_port)
    builder.Connect(lcm_camera_2_pose_to_transform.get_output_port(),point_cloud_filter.camera_pose_2_input_port)


    robot_1_pose_subscriber = builder.AddSystem(LcmSubscriberSystem.Make("robot_1_position",lcm =lcm,lcm_type=lcmt_robot_state,use_cpp_serializer=False,wait_for_message_on_initialization_timeout = 0))
    robot_2_pose_subscriber = builder.AddSystem(LcmSubscriberSystem.Make("robot_2_position",lcm =lcm,lcm_type=lcmt_robot_state,use_cpp_serializer=False,wait_for_message_on_initialization_timeout = 0))

    builder.Connect(robot_1_pose_subscriber.get_output_port(),point_cloud_filter.robot_1_position_port)
    builder.Connect(robot_2_pose_subscriber.get_output_port(),point_cloud_filter.robot_2_position_port)

    builder.Connect(lcm_to_images_1.depth_image_output_port,point_cloud_filter.depth_image_1_input_port)
    builder.Connect(lcm_to_images_2.depth_image_output_port,point_cloud_filter.depth_image_2_input_port)
    
else:
    # try:
    realsense_reader = builder.AddSystem(RealsenseReader())
    from functools import partial
    EE_function = partial(
            robots[0].wrapper.calc_frame_pose_in_frame, 
            frame = robots[0].plant.GetFrameByName("EE_frame"),
            frame_expressed =robots[0].plant.world_frame())
    robot_configuration_to_camera_pose = builder.AddSystem(RobotConfigurationToCameraPose(robot_camera_depth_in_EE,EE_function))
    # RobotConfigurationToCameraPose.port
    # point_cloud_filter = builder.AddSystem(PreprocessPointCloud(margin = 0.04,
    #                                                             robots = robots,
    #                                                             device = device, 
    #                                                             voxel_size=voxel_size,
    #                                                             camera_1_intrinsics=realsense_reader.camera_1_params_depth,
    #                                                             camera_2_intrinsics=realsense_reader.camera_2_params_depth,
    #                                                             ))
    margin = 0.06 #others
    # margin = 0.045 #scenario 3
    point_cloud_filter = builder.AddSystem(PreprocessPointCloud(margin = margin,
                                                                robots = robots,
                                                                device = device, 
                                                                voxel_size=voxel_size,
                                                                camera_1_intrinsics=realsense_reader.camera_1_params_depth,
                                                                camera_2_intrinsics=realsense_reader.camera_2_params_depth, 
                                                                meshcat = meshcat, 
                                                                CODEGEN_FOLDER=CODEGEN_FOLDER))
    # depth_to_point_cloud_2 = builder.AddSystem(DepthImageToPointCloud(camera_info=realsense_reader.camera_1_drake_info, fields= BaseField.kXYZs, pixel_type=PixelType.kDepth32F))
    # depth_to_point_cloud_1 = builder.AddSystem(DepthImageToPointCloud(camera_info=realsense_reader.camera_1_drake_info, fields= BaseField.kXYZs, pixel_type=PixelType.kDepth32F))
    # builder.Connect(realsense_reader.depth_image_1_output_port,depth_to_point_cloud_1.depth_image_input_port())
    builder.Connect(realsense_reader.depth_image_1_output_port,point_cloud_filter.depth_image_1_input_port)
    builder.Connect(realsense_reader.depth_image_2_output_port,point_cloud_filter.depth_image_2_input_port)
    # builder.Connect(realsense_reader.transform_output_port,depth_to_point_cloud_1.camera_pose_input_port())
    # builder.Connect(realsense_reader.depth_image_1_output_port,depth_to_point_cloud_2.depth_image_input_port())
    # builder.Connect(realsense_reader.transform_output_port,depth_to_point_cloud_2.camera_pose_input_port())
    builder.Connect(realsense_reader.transform_output_port,point_cloud_filter.camera_pose_1_input_port)
    builder.Connect(robot_configuration_to_camera_pose.camera_pose_output_port ,point_cloud_filter.camera_pose_2_input_port)
    ros_joint_subscriber_drake = builder.AddSystem(ROSJointState())
    # if dual_panda:
    #     ros_joint_subscriber_drake = builder.AddSystem(ROSJointState(ros_dual_joint_subscriber))
    # else:
    #     ros_joint_subscriber_drake = builder.AddSystem(ROSJointState(ros_joint_subscriber))

    builder.Connect(ros_joint_subscriber_drake.get_output_port(0),point_cloud_filter.robot_1_position_port)
    builder.Connect(ros_joint_subscriber_drake.get_output_port(1),point_cloud_filter.robot_2_position_port)
    builder.Connect(ros_joint_subscriber_drake.get_output_port(0) ,robot_configuration_to_camera_pose.robot_state_input_form)
# printer = builder.AddSystem(Printer())







builder.Connect(final_pose_subscriber.get_output_port(),lcm_final_pose_to_transform.get_input_port())

gaze_polytopes_publisher = builder.AddSystem(LcmPublisherSystem.Make("gaze_polytopes",lcm =lcm_sys,lcm_type=lcmt_gaze_polytopes,use_cpp_serializer=False))
builder.Connect(point_cloud_filter.lcmt_gaze_polytopes_output_port,gaze_polytopes_publisher.get_input_port())

# builder.Connect(depth_to_point_cloud_1.point_cloud_output_port(),point_cloud_filter.point_cloud_1_input_port)
# builder.Connect(depth_to_point_cloud_2.point_cloud_output_port(),point_cloud_filter.point_cloud_2_input_port)
builder.Connect(lcm_final_pose_to_transform.transform_output_port,point_cloud_filter.final_pose_input_port)
# builder.Connect(point_cloud_filter.point_cloud_output_port,svm_pipeline.point_cloud_input_port)
# builder.Connect(svm_pipeline.lcmt_support_vector_output_port,support_vector_publisher.get_input_port())
# builder.Connect(lcm_to_images.color_image_output_port(),printer.get_input_port(0))
# builder.Connect(lcm_pose_to_transform.get_output_port(),printer.get_input_port(1))
# log = LogVectorOutput(printer.get_output_port(),builder)
# builder

diag = builder.Build()
diagram_context = diag.CreateDefaultContext()
point_cloud_filter_context = point_cloud_filter.GetMyContextFromRoot(diagram_context)
# lcm_to_images_context = lcm_to_images.GetMyMutableContextFromRoot(diagram_context)
from pydrake.all import Simulator
# https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/systems/test/lcm_test.py
simulator = Simulator(diag,diagram_context)
camera = meshcat.OrthographicCamera(left=-1.5,
                                    right=2,
                                    top=1.5,
                                    bottom=-0.2,
                                    near=-1,
                                    far=2,
                                    zoom=1.0)
# meshcat.SetCamera(camera)
# BUGIO
dt = 0.01
iteration = 0
saved_fk_w = []
import gc
from IPython.display import clear_output

try:
    while True:
        # _process_event(diag)
        t0 = time.perf_counter()
        print("ITERATION",iteration, '-'*100)

        simulator.AdvanceTo(0.000001)
        diag.ForcedPublish(diagram_context)
        actual_dt = time.perf_counter() - t0
        # print("num support vectors",svm_pipeline.workers[0].num_support_vectors,svm_pipeline.workers[1].num_support_vectors)
        print(f"dt: {actual_dt}")
        # print()
        time.sleep(min(max(dt - actual_dt,0),2))
        
        iteration+=1
        if iteration % 1000 == 0:
            clear_output(wait=True)
            meshcat.Delete('/processed_point_cloud')
            meshcat.Delete('/processed_point_cloud_filtered')
            meshcat.Delete('/original_point_cloud_1')
            meshcat.Delete('/original_point_cloud_2')
            meshcat.Delete('/original_point_cloud_1_ds')
            gc.collect()

        try:
            q2 = point_cloud_filter.robot_2_position_port.Eval(point_cloud_filter_context)
            q1 = point_cloud_filter.robot_1_position_port.Eval(point_cloud_filter_context)
            point_cloud_filter.point_cloud_output_port.Eval(point_cloud_filter_context)
            viz_helper.set_position("robot_1",q2.joint_position)
            viz_helper.set_position("robot_0",q1.joint_position)
            viz_helper.publish()
        except:
            pass
except KeyboardInterrupt:
    pass
finally:
    # for worker in svm_pipeline.workers:
    #     worker.comm_in.put(diff_co.Message(diff_co.MessageType.STOP))
    for i in range(point_cloud_filter.num_workers):
        point_cloud_filter.comms_to_worker[i].put(Message(MessageType.STOP))
    # realsense_reader.shm_at_2.close()
    # realsense_reader.shm_depth_2.close()
    # realsense_reader.shm_color_2.close()
    # realsense_reader.shm_at_1.close()
    # realsense_reader.shm_depth_1.close()
    # realsense_reader.shm_color_1.close()
    # while True:
    #     try:
    #         realsense_reader.pipeline_1.stop()
    #     except KeyboardInterrupt:
    #         pass
    #     except:
    #         break
    # while True:
    #     try:
    #         realsense_reader.pipeline_2.stop()
    #     except KeyboardInterrupt:
    #         pass
    #     except:
    #         break
# lcm.unsubscribe(subscription)
        # self.end_position_slack = self.variable(
        #     "end_position_slack", 3, 1
        # )end_position_slack_cost_weight
        # x_s += [ca.MX(3,1),ca.MX(1,1)]
        # x_s += [self.end_position_slack,self.end_angle_bounds_slack]
        # self.end_angle_bounds_slack = self.variable("end_angle_bounds_slack", 1, 1)


        # end_position_slack = ca.MX.sym("end_position_slack", 3,1)
        # end_angle_bounds_slack = ca.MX.sym("end_angle_bounds_slack", 1,1)
        #         g_translation, lbg_translation, ubg_translation = (
        #     KinematicOptimization.fwd_kin_translation_constraint(
        #         EE_frame_pose,
        #         [0, 0, 0],
        #         ca.DM.eye(4),
        #         translation_MX + end_position_slack,
        #         [0, 0, 0],
        #         [0, 0, 0],
        #     )
        # )
        # g_orientation, lbg_orientation, ubg_orientation = (
        #     KinematicOptimization.fwd_kin_orientation_constraint(
        #         rotation_MX, ca.DM.eye(3), EE_frame_pose, end_angle_bounds_slack
        #     )
        # )        x = veccat(
        #     q_MX,end_position_slack,end_angle_bounds_slack
        # )


# ### Run everything up to this point

# In[ ]:


ros_joint_subscriber_drake.joint_subscriber.joints_2


# In[ ]:


import torch
robot_centers_in_camera = torch.randn(100,3)
robot_geometry_radii = torch.rand(100,1)
last_point_cloud_in_camera_torch = torch.randn(200,3)
d_norm = torch.linalg.norm(robot_centers_in_camera,dim=1,keepdim=True)
theta = torch.asin(robot_geometry_radii/d_norm)
cone_direction = robot_centers_in_camera/d_norm.view(-1,1)

cos_theta = torch.cos(theta)
v = last_point_cloud_in_camera_torch
v_norm = v / torch.norm(v, dim=1, keepdim=True)
dot_product = torch.vmap(torch.matmul,in_dims=(None,0))(v_norm, cone_direction)
# indices_in_poly.append(dot_product >= cos_theta)

dot_product >= cos_theta


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'robot_centers_in_camera = torch.randn(100, 3)\n# robot_centers_in_camera = torch.randn(100, 3)\n# robot_centers_in_camera = torch.tensor([\n#     [0.0, 0.0, 0.6],\n#     [0.1, 0.1, 0.6],\n#     [1.5, -0.4, 0.6],\n#     ])\nrobot_geometry_radii = torch.rand(100, 1)*.1\n# robot_geometry_radii = torch.tensor([[0.2],[0.2],[0.2]])\nlast_point_cloud_in_camera_torch = torch.randn(1000, 3)\nd_norm = torch.linalg.norm(robot_centers_in_camera, dim=1, keepdim=True)\ntheta = torch.asin(robot_geometry_radii / d_norm)\ncone_direction = robot_centers_in_camera / d_norm.view(-1, 1)\n\ncos_theta = torch.cos(theta)\nv = last_point_cloud_in_camera_torch\nn = torch.norm(v, dim=1, keepdim=True)\nv_norm = v / n\ndot_product = torch.vmap(torch.matmul, in_dims=(None, 0))(v_norm, cone_direction)\n\n# Checking which points are inside the cones\ninside_cone = (dot_product >= cos_theta) & (n.T >= d_norm)\n\n# Convert the boolean tensor to a mask\ninside_cone_mask = inside_cone.any(dim=0)\n')


# In[ ]:


inside_cone.shape


# In[ ]:


import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'ipympl')
# Provided code to set up the scenario
robot_centers_in_camera = torch.randn(100, 3)
# robot_centers_in_camera = torch.randn(100, 3)
# robot_centers_in_camera = torch.tensor([
#     [0.0, 0.0, 0.6],
#     [0.1, 0.1, 0.6],
#     [1.5, -0.4, 0.6],
#     ])
robot_geometry_radii = torch.rand(100, 1)*.1
# robot_geometry_radii = torch.tensor([[0.2],[0.2],[0.2]])
last_point_cloud_in_camera_torch = torch.randn(1000, 3)
d_norm = torch.linalg.norm(robot_centers_in_camera, dim=1, keepdim=True)
theta = torch.asin(robot_geometry_radii / d_norm)
cone_direction = robot_centers_in_camera / d_norm.view(-1, 1)

cos_theta = torch.cos(theta)
v = last_point_cloud_in_camera_torch
n = torch.norm(v, dim=1, keepdim=True)
v_norm = v / n
dot_product = torch.vmap(torch.matmul, in_dims=(None, 0))(v_norm, cone_direction)

# Checking which points are inside the cones
inside_cone = (dot_product >= cos_theta) & (n.T >= d_norm)

# Convert the boolean tensor to a mask
inside_cone_mask = inside_cone.any(dim=0)

# Points inside the cone
points_inside_cone = last_point_cloud_in_camera_torch[inside_cone_mask]
points_outside_cone = last_point_cloud_in_camera_torch[~inside_cone_mask]

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot robot centers
ax.scatter(robot_centers_in_camera[:, 0].numpy(), robot_centers_in_camera[:, 1].numpy(), robot_centers_in_camera[:, 2].numpy(), c='blue', label='Robot Centers')

# Plot last point cloud (inside cone in green, outside cone in red)
ax.scatter(points_inside_cone[:, 0].numpy(), points_inside_cone[:, 1].numpy(), points_inside_cone[:, 2].numpy(), c='green', label='Points Inside Cone')
ax.scatter(points_outside_cone[:, 0].numpy(), points_outside_cone[:, 1].numpy(), points_outside_cone[:, 2].numpy(), c='red', label='Points Outside Cone')

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Visualization of Points Inside and Outside of Cones')
ax.legend()

plt.show()


# In[ ]:


import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def plot_cone_simple(ax, apex, direction, angle, height, color='blue', alpha=0.2):
    """Plot a simple cone given the apex, direction, angle, and height."""
    # Create a cone
    cone_base_radius = height * np.tan(angle)
    cone_base_center = apex + direction * height

    # Parametric circle (base of the cone)
    theta = np.linspace(0, 2 * np.pi, 30)
    x_base = cone_base_radius * np.cos(theta)
    y_base = cone_base_radius * np.sin(theta)
    z_base = np.zeros_like(x_base)

    # Rotate the base circle to align with the cone direction
    cone_direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    if not np.allclose(cone_direction, z_axis):
        v = np.cross(z_axis, cone_direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, cone_direction)
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + k + k.dot(k) * ((1 - c) / (s ** 2))
        base_circle = np.dot(np.c_[x_base, y_base, z_base], R.T)
    else:
        base_circle = np.c_[x_base, y_base, z_base]

    # Translate to the cone base center
    base_circle += cone_base_center

    # Plot the cone surface
    for i in range(len(theta)):
        vertices = [apex, base_circle[i - 1], base_circle[i]]
        tri = Poly3DCollection([vertices], color=color, alpha=alpha)
        ax.add_collection3d(tri)

# Set up the plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot robot centers
ax.scatter(robot_centers_in_camera[:, 0].numpy(), robot_centers_in_camera[:, 1].numpy(), robot_centers_in_camera[:, 2].numpy(), c='blue', label='Robot Centers')

# Plot cones
for i in range(len(robot_centers_in_camera)):
    plot_cone_simple(ax, robot_centers_in_camera[i].numpy(), cone_direction[i].numpy(), theta[i].item(), height=3.0, color='cyan', alpha=0.3)

# Plot last point cloud (inside cone in green, outside cone in red)
ax.scatter(points_inside_cone[:, 0].numpy(), points_inside_cone[:, 1].numpy(), points_inside_cone[:, 2].numpy(), c='green', label='Points Inside Cone')
ax.scatter(points_outside_cone[:, 0].numpy(), points_outside_cone[:, 1].numpy(), points_outside_cone[:, 2].numpy(), c='red', label='Points Outside Cone')

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Visualization of Cones and Points in 3D Space')
ax.legend()

plt.show()


# In[ ]:


# dot_product
# robot_geometry_radii
cos_theta


# In[ ]:


robot_geometry_radii/d_norm


# In[ ]:


point_cloud_filter.camera_pose_1_input_port.Eval(point_cloud_filter_context)
# point_cloud_filter.camera_pose_2_input_port.Eval(point_cloud_filter_context)
# point_cloud_filter.camera_pose_1_input_port


# In[ ]:


lcm = DrakeLcm()
def get_msg(msgs):
    global msg
    # print( 'fasd')
    # print(msg)
    msg = lcmt_image_array.decode(msgs)

lcm.Subscribe('simulation_depth_image_2',get_msg)
lcm.HandleSubscriptions(300)


# In[ ]:


array = np.frombuffer(msg.images[0].data,dtype=np.float32).reshape(480,640,1).copy()
plt.imshow(array)


# 

# In[ ]:


for i in range(point_cloud_filter.num_workers):
    point_cloud_filter.comms_to_worker[i].put(diff_co.Message(diff_co.MessageType.STOP))
try:
    
    realsense_reader.pipeline_1.stop()
except:
    pass
try:
    realsense_reader.pipeline_2.stop()
except:
    pass


# In[ ]:


from IPython.display import clear_output
meshcat.SetProperty("/drake","opacity",0.25)
meshcat.SetProperty("/drake","modulated_opacity",0.25)
meshcat.SetProperty("/original_point_cloud_1","opacity",0.25)


# In[ ]:


svm_pipeline.point_cloud_torch_gpu.shape


# In[ ]:


msg = point_cloud_filter.message
vertices = np.asarray(msg.vertices)
simplices = np.asarray(msg.simplices)
indices = np.asarray(msg.indices)
# print(indices[-1],vertices.shape)
# TODO: calculate this once and just shift the points
# pose_gaze_samples,iks_robot_1,transform_masks, ik_mask = gaze.get_samples_around_placing_spot(sampling_size_around_spot,num_samples_spot,num_samples_gaze,num_samples_gaze_turn,placing_spot_pose,q7_guesses,elevation,robot_base_pose_1)
# line_point_ = pose_gaze_samples[:,:3,3].astype(np.float64)
# line_direction_ = pose_gaze_samples[:,:3,2].astype(np.float64)
polytopes = []
for i,(a,b,c,d) in enumerate(indices):
    polytope_vertices= vertices[a:b]

    polytope_simplices = simplices[c:d]
    meshcat.SetTriangleMesh(f'/polytope_{i}',polytope_vertices.T,polytope_simplices.T)

    polytopes.append( { 'vertices': vertices[a:b], 'simplices': simplices[c:d] } )
    mean_vertices = np.mean(polytope_vertices,axis=0)
    # if np.linalg.norm(mean_vertices-placing_spot_pose.translation()) > 0.5:
        # continue
    polytope_vertices_ = polytope_vertices.astype(np.float64)
    faces = polytope_vertices[polytope_simplices]
    faces_ = np.array(faces).astype(np.float64)


# In[ ]:


msg.vertices.shape,b
msg.simplices.shape,d


# In[ ]:


self = point_cloud_filter
message = lcmt_gaze_polytopes()
gaze_indices = self.work_tensors[0]['gaze_indices_buffer'][:self.num_polytopes].cpu().numpy()
a,b,c,d = gaze_indices[-1]
gaze_polytopes = self.work_tensors[0]['gaze_polytopes_buffer'][:b].cpu().numpy()
gaze_simplices = self.work_tensors[0]['gaze_simplices_buffer'][:d].cpu().numpy()
message.num_polytopes = self.num_polytopes
message.num_vertices = b
message.num_simplices = d
# print(gaze_indices[-1])
message.indices = gaze_indices
message.vertices = gaze_polytopes
message.simplices = gaze_simplices
message.is_placing_spot_hidden = self.placing_spot_hidden
self.message = message


# In[ ]:


gaze_simplices.shape,message.simplices.shape,b,d,gaze_indices


# In[ ]:


message.indices


# In[ ]:


point_cloud_filter.work_tensors[0]["gaze_simplices_buffer"][:d]


# In[ ]:


try:
    realsense_reader.pipeline_1.stop()
except:
    pass
try:
    realsense_reader.pipeline_2.stop()
except:
    pass


# In[ ]:


worker_idx = 1
self = svm_workers[worker_idx]
from diff_co_mpc.diff_co import *

kernel_matrix_function = (torch.export.load(self.kernel_matrix_function_name).module().to(self.device))
in_collision_function = (torch.export.load(self.in_collision_function_name).module().to(self.device))
# if self.compile:
#     kernel_matrix_function = torch.compile(kernel_matrix_function, dynamic  = True)
#     in_collision_function = torch.compile(in_collision_function, dynamic  = True)
#     for i in range(10):
#         kernel_matrix_function(torch.randn(1000+i*100,self.X_buffer.shape[1],device=self.device))
#         in_collision_function(torch.randn(1000+i*100,self.X_buffer.shape[1],device=self.device),torch.randn(200+i*10,3,device=self.device),torch.randn(200+i*10,1,device=self.device))


# In[ ]:


worker = self
num_exploration_samples = 3000
last_num_support_vectors = worker.num_support_vectors
last_support_vectors = worker.X_buffer[:last_num_support_vectors]#$#.clone()
last_weights = worker.W_buffer[:last_num_support_vectors]#.clone()
last_H_s = worker.H_s_buffer[:last_num_support_vectors]#.clone()
lower_limits = torch.tensor(robots[0].plant.GetPositionLowerLimits(),dtype = torch.float32,device = 'cpu')
upper_limits = torch.tensor(robots[0].plant.GetPositionUpperLimits(),dtype = torch.float32,device = 'cpu')

# get exploration samples based on saved trajectories

# x_initial_guess = svm_pipeline.initial_guess_control_points[worker_idx]
# random_indices = torch.randperm(x_initial_guess.shape[0])[:num_exploration_samples]
# exploration_samples = x_initial_guess[random_indices]
# exploration_samples += torch.randn_like(exploration_samples)*0.4
# exploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)
exploration_samples = torch.randn(num_exploration_samples,9,device='cpu')
exploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)
exploitation_size = 3000
random_indices = torch.randint(0, last_support_vectors.shape[0], (exploitation_size,))
exploitation_samples = last_support_vectors[random_indices]
exploitation_samples += torch.randn_like(exploitation_samples)*1
exploitation_samples = torch.clip(exploitation_samples,lower_limits,upper_limits)
new_X = torch.vstack((last_support_vectors,exploitation_samples,exploration_samples))

worker.X_buffer[:new_X.shape[0]] = new_X
import tempfile
temp_folder = pathlib.Path(tempfile.mkdtemp())


# In[ ]:


last_num_support_vectors


# In[ ]:


# lib = ctypes.CDLL('./libsupport_vectors99.so')

# # Define the argument and return types
# lib.calculate_support_vectors_indexes.argtypes = [
#     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
#     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
#     ctypes.c_int, ctypes.c_int, ctypes.c_int,
#     ctypes.c_int,
#     ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_bool),
#     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
# ]

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
so_file = temp_folder / (randomword(10) + '.so')
print(os.system(f"g++ -shared -o {str(so_file)} -O3 -march=native -fPIC calculate_support_vector_indexes.cpp -I /usr/include/eigen3 -ffast-math"))
lib = ctypes.CDLL(str(so_file))

# Define the argument and return types
lib.calculate_support_vectors_indexes.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_bool),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
]
def calculate_support_vectors_indexes_py(Y, H, W, K, MAX_ITERATION=10000000):
    rows,cols = Y.shape

    k_rows, k_cols = K.shape
    cols = 1
    # Convert numpy arrays to C arrays
    # t0 = time.perf_counter()
    Y_c = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    H_c = H.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    W_c = W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    K_c = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # print("time ctypes data as",time.perf_counter() - t0)
    non_zero_W_indices = -np.ones(rows, dtype=np.int32)
    completed = np.zeros(1, dtype=np.bool_)
    min_M = ctypes.c_float()
    zero_M_count = ctypes.c_int()
    # t0 = time.perf_counter()
    lib.calculate_support_vectors_indexes(
        Y_c, H_c, W_c, K_c,
        rows, k_rows, k_cols,
        MAX_ITERATION,
        np.ctypeslib.as_ctypes(non_zero_W_indices),
        # non_zero_W_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        completed.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.byref(min_M), ctypes.byref(zero_M_count)
    )
    # print("time in cpp",time.perf_counter() - t0)
    # Return results
    # t0 = time.perf_counter()
    non_zero_W_indices = non_zero_W_indices[non_zero_W_indices != -1]
    # print("time this shit right here",time.perf_counter() - t0)
    return non_zero_W_indices, completed, min_M.value, zero_M_count.value


# In[ ]:


so_file


# In[ ]:


# lib = ctypes.CDLL('./libsupport_vectorsxxxx.so')
# lib.
point_cloud = torch.randn(200,3)*0.1


# In[ ]:


point_cloud_size = point_cloud.shape[0]

from diff_co_mpc.diff_co import *
t_start = time.perf_counter()
support_vector_calculation_time = 0
kernel_calculation_time = 0
in_collision_calculation_time = 0

# worker.num_support_vectors = 1
last_num_support_vectors = worker.num_support_vectors
# last_num_support_vectors = 1
last_support_vectors = worker.X_buffer[:last_num_support_vectors]#$#.clone()

num_exploration_samples = 200
# x_initial_guess = svm_pipeline.initial_guess_control_points[worker_idx]
# random_indices = torch.randperm(x_initial_guess.shape[0])[:num_exploration_samples]
# exploration_samples = x_initial_guess[random_indices]
# exploration_samples += torch.randn_like(exploration_samples)*0.4
# exploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)
exploration_samples = torch.randn(num_exploration_samples,9,device='cpu')
exploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)

exploitation_size = 200
random_indices = torch.randint(0, last_support_vectors.shape[0], (exploitation_size,))
exploitation_samples = last_support_vectors[random_indices].clone()
exploitation_samples += torch.randn_like(exploitation_samples)*0.1
exploitation_samples = torch.clip(exploitation_samples,lower_limits,upper_limits)
new_X = torch.vstack((last_support_vectors,exploitation_samples,exploration_samples))

worker.X_buffer[:new_X.shape[0]] = new_X


max_iter = 80000000
sample_size = new_X.shape[0]
# sample_size = 1000
# point_cloud = self.point_cloud_buffer[:point_cloud_size].clone()
X = self.X_buffer[:sample_size].clone()

num_exploitation_vectors = self.num_support_vectors + exploitation_size
num_exploration_samples = sample_size - num_exploitation_vectors


# new_samples = X[self.num_support_vectors:].clone()

t0 = time.perf_counter()
fk_X,K = kernel_matrix_function(X)
fk_X = fk_X.reshape(-1,fk_X.shape[1]*fk_X.shape[2])
kernel_calculation_time += time.perf_counter()-t0



t0 = time.perf_counter()
Y = (in_collision_function(X,point_cloud,self.obstacle_radii.repeat(point_cloud_size,1))*2-1  ).reshape(-1,self.Y_buffer.shape[1])
in_collision_calculation_time += time.perf_counter()-t0


max_iter = 1000000
K_np = K.cpu().numpy().copy()
# print(K_np)
# X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy().astype(np.float32).copy()
# safsaf
# plt.plot(self.W_buffer[:self.num_support_vectors].cpu().numpy())
if self.num_support_vectors > 0:
    # pass
    
    last_new_H_np = np.zeros((num_exploitation_vectors,Y_np.shape[1]),dtype=np.float32)
    last_new_W = np.zeros((num_exploitation_vectors,Y_np.shape[1]),dtype=np.float32)
    
    last_new_W[:self.num_support_vectors] = self.W_buffer[:self.num_support_vectors].cpu().numpy().copy()

    last_new_Y = Y_np[:num_exploitation_vectors].copy()
    last_support_vector_K = K_np[:num_exploitation_vectors,:num_exploitation_vectors].copy()
    last_new_H_np[:] = last_support_vector_K@last_new_W
    
    # plt.plot()
    plt.plot(last_new_Y*last_new_H_np)
    # afsdsfdf
    t0 = time.perf_counter()
    # support_vector_indices, completed, max_margin, num_mislabeled = calculate_support_vectors_indexes_py(last_new_Y, last_new_H_np, last_new_W, last_support_vector_K,max_iter) 
    support_vector_indices, completed, max_margin, num_mislabeled =calculate_support_vectors_indexes_numba(None, last_new_Y, H=last_new_H_np, W=last_new_W, K=last_support_vector_K, MAX_ITERATION=max_iter)
    print(f'Time 1st pass {self.index}: {time.perf_counter()-t0}s')
    # plt.plot(last_new_W)
    # plt.plot(K_np[:,:219][:219]@last_new_W)
    # plt.plot(last_new_H_np)
    print(completed, max_margin, num_mislabeled)
    print(support_vector_indices.shape,self.num_support_vectors)
    support_vector_calculation_time += time.perf_counter()-t0
    support_vector_indices = support_vector_indices.reshape(-1)

    # W_np = np.zeros((sample_size,Y_np.shape[1]),dtype=np.float32)
    # H_np[:last_new_H_np.shape[0]] = last_new_H_np
    # W_np[:last_new_W.shape[0]] = last_new_W
    # H_np = K_np@W_np
    new_X = torch.empty((support_vector_indices.shape[0] + num_exploration_samples,X.shape[1]),dtype=torch.float32)
    new_X[:support_vector_indices.shape[0]] = X[support_vector_indices]
    new_X[support_vector_indices.shape[0]:] = X[num_exploitation_vectors:]
    new_fK_X = torch.empty((support_vector_indices.shape[0] + num_exploration_samples,fk_X.shape[1]),dtype=torch.float32)
    new_fK_X[:support_vector_indices.shape[0]] = fk_X[support_vector_indices]
    new_fK_X[support_vector_indices.shape[0]:] = fk_X[num_exploitation_vectors:]
    
    new_Y_np = np.empty((support_vector_indices.shape[0] + num_exploration_samples,Y_np.shape[1]),dtype=np.float32)
    new_Y_np[:support_vector_indices.shape[0]] = last_new_Y[support_vector_indices]
    new_Y_np[support_vector_indices.shape[0]:] = Y_np[num_exploitation_vectors:]
    Y_np = new_Y_np
    H_np = np.zeros((support_vector_indices.shape[0]+num_exploration_samples,Y_np.shape[1]),dtype=np.float32)
    W_np = np.zeros((support_vector_indices.shape[0]+num_exploration_samples,Y_np.shape[1]),dtype=np.float32)
    H_np[:support_vector_indices.shape[0]] = last_new_H_np[support_vector_indices]
    W_np[:support_vector_indices.shape[0]] = last_new_W[support_vector_indices]
    K_indices = np.concatenate([support_vector_indices,np.array(range(num_exploitation_vectors,K_np.shape[0]),dtype  = np.int32)])
    # plt.plot(W_np)
    # asfasffs
    # fsafss
    K_np = (K_np[K_indices][:,K_indices]).copy()
    # K_np = ((K_np.ravel()[(K_indices + (K_indices * K_np.shape[1]).reshape((-1,1))).ravel()]).reshape(K_indices.size, K_indices.size) ).copy()
    # plt.figure()
    # plt.plot(H_np,)
    H_np = K_np@W_np
    # plt.plot(H_np,'--')
    # plt.plot(H_np,'--')
    # adfsdf
else:
    H_np = np.zeros((sample_size,Y_np.shape[1]),dtype=np.float32)
    W_np = np.zeros((sample_size,Y_np.shape[1]),dtype=np.float32)
plt.figure()
# plt.plot(W_np)
t0 = time.perf_counter()
# dasfdfasfd
# support_vector_indices, completed, max_margin, num_mislabeled = calculate_support_vectors_indexes_py(Y_np, H_np, W_np, K_np,max_iter) 
support_vector_indices, completed, max_margin, num_mislabeled = calculate_support_vectors_indexes_numba(None, Y_np, H=H_np, W=W_np, K=K_np, MAX_ITERATION=80000)
print(f'Time 2nd pass {self.index}: {time.perf_counter()-t0}s')
print(support_vector_indices.shape[0],completed, max_margin, num_mislabeled)

num_support_vectors = support_vector_indices.size
support_vector_indices = support_vector_indices.squeeze()

self.X_buffer[:num_support_vectors] = new_X[support_vector_indices]
self.fk_X_buffer[:num_support_vectors] = new_fK_X[support_vector_indices]

self.H_s_buffer[:num_support_vectors] = torch.as_tensor(H_np,dtype=torch.float32,device = self.device)[support_vector_indices,:]
self.W_buffer[:num_support_vectors] = torch.as_tensor(W_np,dtype=torch.float32,device = self.device)[support_vector_indices,:]

self.num_support_vectors = num_support_vectors
# self.num_support_vectors = 1
# plt.plot(H_np*Y_np)
plt.plot(W_np)
# print(new_X)


# In[ ]:


self.num_support_vectors


# In[ ]:


W_result = W_np[support_vector_indices,:]
Y_result = Y_np[support_vector_indices,:]

K_result = K_np[:,support_vector_indices][support_vector_indices]
opti = ca.Opti('conic')
W_opt = opti.variable(*W_result.shape)
Y_opt = opti.parameter(*Y_result.shape)
K_opt = opti.parameter(*K_result.shape)
slack_variables = opti.variable(*Y_result.shape)
weight_fit = opti.parameter()
weight_reg = opti.parameter()
minimum_margin = opti.parameter()

hypothesis = (K_opt@W_opt)
margin = Y_opt*hypothesis
hinge_loss = weight_fit*ca.sum1(slack_variables)
margin_constraint = minimum_margin<=margin

opti.subject_to(margin_constraint)
opti.subject_to(margin>=1-slack_variables)
opti.subject_to(slack_variables>=0)
# cost_reg = weight_reg*ca.sum1(ca.fabs(W_opt))
cost_reg = weight_reg*ca.sumsqr(W_opt)
opti.minimize(cost_reg + hinge_loss)
# opti.solver('ipopt',{'ipopt':{'linear_solver':'ma57','print_level':3,    'max_iter': 2, 'max_wall_time':5,}})
opti.solver('qpoases')
opti.set_value(Y_opt,Y_result)
opti.set_value(K_opt,K_result)
opti.set_value(minimum_margin,-0.05)
opti.set_value(weight_fit,100)
opti.set_value(weight_reg,10)
opti.solve()
W_new  = opti.value(W_opt)


# In[ ]:


ca.buil


# In[ ]:


plt.plot(Y_result*K_result@W_new)
# Y_result*K_result@W_new
plt.plot(Y_result*K_result@W_result)


# In[ ]:


plt.plot(W_new)
# plt.plot(W_result)


# In[ ]:


opti.set_value(Y_opt,Y_result)
opti.set_value(K_opt,K_result)
opti.set_value(minimum_margin,-0.1)
opti.set_value(weight_fit,100)
opti.set_value(weight_reg,0.1)
try:
    opti.solve()
except:
    pass
W_new  = opti.value(W_opt)
plt.plot(Y_result*K_result@W_new)
plt.plot(Y_result*K_result@W_result)


# In[ ]:


W_new!=0


# In[ ]:


plt.plot(W_new)
plt.plot(W_result)
plt.scatter(range(0,W_new.shape[0]),W_new!=0,color='red',s=1)

plt.ylim(-1,1)


# In[ ]:


plt.plot(Y_result*K_result@W_new)
plt.plot(Y_result*K_result@W_result)
plt.ylim(-0.25,1)


# In[ ]:


W_new


# In[ ]:


K_np = (K_np[K_indices][:,K_indices]).copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', '(K_np).T[:,K_indices][K_indices]\n3\n# K.cpu().numpy().copy()\n# K_np = K.cpu().numpy().copy()\n')


# In[ ]:


a = K_np.copy()
cols = rows = K_indices
get_ipython().run_line_magic('timeit', '((a.ravel()[(cols + (rows * a.shape[1]).reshape((-1,1))).ravel()]).reshape(rows.size, cols.size) ).copy()')
get_ipython().run_line_magic('timeit', '((K_np).T[:,K_indices][K_indices]).copy()')


# In[ ]:


np.all(((K_np).T[:,K_indices][K_indices]).copy()==((K_np)[:,K_indices][K_indices]).copy())


# In[ ]:


num_exploitation_vectors
self.num_support_vectors


# In[ ]:


num_exploration_samples


# In[ ]:


X2


# In[ ]:


K_indices


# In[ ]:


plt.imshow(K_np[support_vector_indices][:,support_vector_indices])


# In[ ]:


# last_new_W.shape


# In[ ]:


K_indices = np.concatenate([support_vector_indices,np.array(range(num_exploitation_vectors,K_np.shape[0]))])
# asfasffs
K_support = K_np[K_indices][:,K_indices].copy()
K_support
plt.imshow(K_np)
plt.imshow(K_support)


# In[ ]:


kernel_matrix_functionc(torch.tensor([[0.]*9,[0.1]*9,[2.]*9,[0.1]*9],device='cpu'))


# In[ ]:


plt.plot(W1)
plt.plot(W2)


# In[ ]:


X1-X2


# In[ ]:


plt.plot(H2)
plt.plot((K2@W_np[support_vector_indices,:]))


# In[ ]:


plt.imshow(K1-K2)


# In[ ]:


K1.shape,K2.shape


# In[ ]:


K1[:num_exploitation_vectors,:num_exploitation_vectors] - K2[:num_exploitation_vectors,:num_exploitation_vectors].cpu().numpy()


# In[ ]:


K1.shape
K2.shape


# In[ ]:


num_exploitation_vectors


# In[ ]:


num_exploration_samples


# In[ ]:


num_exploitation_vectors


# In[ ]:


torch.cuda.empty_cache()
gc.collect()


# In[ ]:


plt.plot(K_np.reshape(-1))


# In[ ]:


np.empty((support_vector_indices.shape[0] + num_exploration_samples,Y_np.shape[1]),dtype=np.float32).shape


# In[ ]:


num_exploration_samples


# In[ ]:


last_new_W.shape


# In[ ]:


K_np@W_np


# In[ ]:


support_vector_indices, completed, max_margin, num_mislabeled = calculate_support_vectors_indexes_py(last_new_Y, last_new_H_np, last_new_W, last_support_vector_K) #calculate_support_vectors_indexes_numba(None, last_new_Y, H=last_new_H_np, W=last_new_W, K=last_support_vector_K, MAX_ITERATION=max_iter)


# In[ ]:


np.savez('variables.npz', Y=last_new_Y, H=last_new_H_np, W=last_new_W, K=last_support_vector_K)
# calculate_support_vectors_indexes_py(, , , ) #calculate_support_vectors_indexes_numba(None, last_new_Y, H=last_new_H_np, W=last_new_W, K=last_support_vector_K, MAX_ITERATION=max_iter)


# In[ ]:


K_np.shape


# In[ ]:


Y_np[num_exploitation_vectors:].shape


# In[ ]:


# W_np_cpp = W_np.copy()
plt.plot(W_np_cpp,alpha=0.5)
plt.plot(W_np,alpha=0.5)


# In[ ]:


import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libsupport_vectors11.so')

# Define the argument and return types
lib.calculate_support_vectors_indexes.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_bool),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
]

def calculate_support_vectors_indexes_py(Y, H, W, K, MAX_ITERATION=1000000):
    rows,cols = Y.shape

    k_rows, k_cols = K.shape
    cols = 1
    # Convert numpy arrays to C arrays
    Y_c = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    H_c = H.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    W_c = W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    K_c = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    non_zero_W_indices = np.zeros(rows, dtype=np.int32)
    completed = np.zeros(1, dtype=np.bool_)
    min_M = ctypes.c_float()
    zero_M_count = ctypes.c_int()
    lib.calculate_support_vectors_indexes(
        Y_c, H_c, W_c, K_c,
        rows, k_rows, k_cols,
        MAX_ITERATION,
        non_zero_W_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        completed.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        ctypes.byref(min_M), ctypes.byref(zero_M_count)
    )
    # print(Y)
    # Return results
    non_zero_W_indices = non_zero_W_indices[non_zero_W_indices != 0]
    return non_zero_W_indices, completed, min_M.value, zero_M_count.value

import time
t0 = time.perf_counter()
indices, completed, min_m, zero_m_count = calculate_support_vectors_indexes_py(Y_np.astype(np.float32).copy(), H_np.astype(np.float32).copy(), W_np.astype(np.float32).copy(), K_np.astype(np.float32).copy())
print(time.perf_counter()-t0)


# In[ ]:


X = data['X'].astype(np.float32)
Y = data['Y'].astype(np.float32)
H = data['H'].astype(np.float32)
W = data['W'].astype(np.float32)
K = data['K'].astype(np.float32)
X = data['X'].astype(np.float32)

# print(K)\
# print(Y.dtype)
# dasfds
import time
t0 = time.perf_counter()
Y = Y_np.astype(np.float32).copy()
H = H_np.astype(np.float32).copy()
W = W_np.astype(np.float32).copy()
K = K_np.astype(np.float32).copy()
indices, completed, min_m, zero_m_count = calculate_support_vectors_indexes_py(Y, H, W, K)
print(indices,min_m)
print(time.perf_counter()-t0)


# In[ ]:


indices


# In[ ]:


ros_joint_subscriber.joints_1


# ### Robot Camera EE to Camera transform

# In[ ]:





# In[ ]:


q = ros_dual_joint_subscriber.joints_1
extrinsic_t = np.array([-0.0594441, -0.000257048, 0.000345637]).reshape(3,1)
extrinsic_R = np.array([0.999989, 0.00467125, -0.0010296, -0.00467758, 0.99997, -0.00623559, 0.00100044, 0.00624033, 0.99998]).reshape(3,3)

T_depth_in_color = np.eye(4)
T_depth_in_color[:3,3:] = extrinsic_t*1
T_depth_in_color[:3,:3] = extrinsic_R.T

T_AT_in_color = np.array([[-0.02223654, -0.99964156,  0.01490906, -0.00598663],
       [ 0.99687251, -0.02103885,  0.07617461, -0.1026754 ],
       [-0.07583364,  0.01655629,  0.99698302,  0.62791229],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])


T_color_in_AT = np.linalg.inv(T_AT_in_color)
T_w_in_AT = RigidTransform(p = [0,-0.15,0], R = RotationMatrix.MakeXRotation(np.pi)).GetAsMatrix4()
T_AT_in_w = np.linalg.inv(T_w_in_AT)
T_depth_in_world = T_AT_in_w@T_color_in_AT@T_depth_in_color

m1 = viz_helper.plant.GetModelInstanceByName("robot_1")
m0 = viz_helper.plant.GetModelInstanceByName("robot_0")
viz_helper.set_position("robot_0",q)
viz_helper.publish()
T_EE_in_w = viz_helper.plant.GetFrameByName("EE_frame", m0).CalcPoseInWorld(viz_helper.plant_context).GetAsMatrix4()

# T_depth_in_world = T_EE_in_w@T_depth_in_EE
T_depth_in_EE = np.linalg.inv(T_EE_in_w)@T_depth_in_world


# In[ ]:


T_depth_in_EE


# In[ ]:


T_color_in_AT


# In[ ]:


T_depth_in_EE


# In[ ]:


T_depth_in_world


# In[ ]:


T_w_c = np.linalg.inv(T_c_w)
T_EE_w = np.linalg.inv(T_w_EE)
T_EE_c = T_w_c@T_EE_w
RigidTransform(T_EE_c)


# In[ ]:


T_w_c


# In[ ]:


T_EE_w


# In[ ]:





# In[ ]:


np.linalg.inv(T_c_EE)


# In[ ]:


meshcat.SetObject("/aaaaa",torch_to_drake_point_cloud(point_cloud_filter.output_point_cloud),0.02,Rgba(0,0,1,1))


# In[ ]:


from pydrake.all import Sphere


# In[ ]:


b = torch.cdist(point_cloud_filter.output_point_cloud, torch.tensor([0.25,-0.5,0.35]).reshape(-1,3)).argmin()
# point_cloud_filter.output_point_cloud[67]

args = torch.cdist(torch.tensor([0.25,-0.5,0.35]).reshape(-1,3).reshape(-1,3), point_cloud_filter.output_point_cloud).argsort().squeeze()
point_cloud_filter.voxel_size
meshcat.SetObject("/bbbb",Sphere(point_cloud_filter.voxel_size))
meshcat.SetTransform("/bbbb",RigidTransform(p =point_cloud_filter.output_point_cloud[args[1]] ))


# In[ ]:


torch.cdist(point_cloud_filter.output_point_cloud,point_cloud_filter.output_point_cloud)


# In[ ]:


torch.cdist(point_cloud_filter.output_point_cloud[args[1]].reshape(1,3),point_cloud_filter.output_point_cloud).argsort()


# In[ ]:





# In[ ]:


# d = torch.




(point_cloud_filter.output_point_cloud,point_cloud_filter.output_point_cloud).squeeze()
# d[86]
d.fill_diagonal_(point_cloud_filter.voxel_size+0.1)
pc2 = point_cloud_filter.output_point_cloud[~torch.any(torch.triu(d < 0.01),dim = 1)]
d = torch.cdist(pc2,pc2).squeeze()
d.fill_diagonal_(point_cloud_filter.voxel_size+0.1)
meshcat.SetObject("/aaaaa",torch_to_drake_point_cloud(pc2[~torch.all(d > point_cloud_filter.voxel_size,dim = 1)]),0.03,Rgba(0,0,0,1))


# In[ ]:


viz_helper.publish_diagram()


# In[ ]:


point_cloud_filter.output_point_cloud
meshcat.SetObject("/cccc",Sphere(0.01))
meshcat.SetTransform("/cccc",RigidTransform(p =point_cloud_filter.output_point_cloud[78] ))


# In[ ]:


torch.cdist(point_cloud_filter.output_point_cloud[67].reshape(-1,3), point_cloud_filter.output_point_cloud).argsort()[7]


# In[ ]:


viz_helper.publish()
m1 = viz_helper.plant.GetModelInstanceByName("robot_1")
m0 = viz_helper.plant.GetModelInstanceByName("robot_0")
viz_helper.plant.GetFrameByName("EE_frame", m1).CalcPose(viz_helper.plant_context,viz_helper.plant.GetFrameByName("panda_link0",m1),)
viz_helper.plant.GetFrameByName("EE_frame", m0).CalcPose(viz_helper.plant_context,viz_helper.plant.world_frame(),)
viz_helper.plant.GetFrameByName("EE_frame", m0).CalcPose(viz_helper.plant_context,viz_helper.plant.GetFrameByName("panda_link0",m0),)


# In[ ]:


p = np.array([-0.0508205,-0.998695,-0.00252562,0,-0.998108,0.0508773,-0.0342425,0,0.034327,0.000780632,-0.99941,0,0.535903,-0.0721604,0.205727,1]).reshape(4,4).T
p


# In[ ]:


import torch
import torch


rs_context = realsense_reader.GetMyContextFromRoot(diagram_context)
fx,fy,cx,cy = realsense_reader.camera_1_params_at
pc = depth_image_to_point_cloud(torch.as_tensor(realsense_reader.depth_image_1_output_port.Eval(rs_context).data).squeeze(),fx,fy,cx,cy)


# In[ ]:


realsense_reader.camera_1_params_at


# In[ ]:


i = torch.eye(3,device="cpu")
o = torch.ones((3,1),device="cpu")
l = torch.tensor([-1,-1,0.01],device="cpu")
u = torch.tensor([1,1,1.2],device="cpu")


# In[ ]:


t


# In[ ]:


viz_helper.set_position("robot_0",[0.365639,0.568507,-0.54315,-2.19033,0.568215,2.62768,1.76298,0,0])
viz_helper.publish()


# In[ ]:


# %%timeit
pc = depth_image_to_point_cloud(torch.as_tensor(realsense_reader.depth_image_1_output_port.Eval(rs_context).data).squeeze().cpu(),fx,fy,cx,cy)
# pc = torch.as_tensor(realsense_reader.points),
R = torch.as_tensor(realsense_reader.transform_output_port.Eval(rs_context).rotation().matrix(),dtype=torch.float32)
t = torch.as_tensor(realsense_reader.transform_output_port.Eval(rs_context).translation(),dtype=torch.float32).reshape(3,1)
pc_cam = R@pc.unsqueeze(-1) + t
# .Crop([-1,-1,0.01],[1,1,1.2])
cropped = pc_cam[torch.all((pc_cam > l.reshape(3,1)) & (pc_cam < u.reshape(3,1)),dim=1).squeeze()]


# In[ ]:


R,t


# In[ ]:


torch.all((pc_cam > l.reshape(3,1)) & (pc_cam < u.reshape(3,1)),dim=-1)


# In[ ]:


meshcat.ResetRenderMode()


# In[ ]:


meshcat.SetObject("/torch_pc",torch_to_drake_point_cloud(cropped),0.001,Rgba(0,0,1,1))


# In[ ]:


c = depth_to_point_cloud_1.GetMyContextFromRoot(diagram_context)

c.DisableCaching()

# R


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndepth_to_point_cloud_1.get_output_port().Eval(c).Crop([0,0,0],[1,1,1])\n')


# In[ ]:


torch.as_tensor(realsense_reader.depth_image_1_output_port.Eval(rs_context).data).shape


# In[ ]:


import scipy
scipy.__version__


# In[ ]:


# realsense_intrinsics.focal_y(),
# --https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20#intrinsic-camera-parameters
CameraInfo(
    width=1280,
    height=720,
    fov_y=87/180*np.pi,
).focal_y()


# In[ ]:


worker = svm_workers[0]
prob_in_collision_function  = (torch.export.load(worker.prob_in_collision_function_name).module().cuda())
kernel_matrix_function = (torch.export.load(worker.kernel_matrix_function_name).module().cuda())


# In[ ]:


X = torch.randn(5000,9,device = "cuda")
lower_limits = torch.tensor(robots[0].plant.GetPositionLowerLimits(),dtype = torch.float32,device = 'cuda')
upper_limits = torch.tensor(robots[0].plant.GetPositionUpperLimits(),dtype = torch.float32,device = 'cuda')
x_initial_guess = svm_pipeline.initial_guess_control_points[0]
random_indices = torch.randperm(x_initial_guess.shape[0])[:10000]
X = x_initial_guess[random_indices]
X += torch.randn_like(X)*0.3
X = torch.clip(X,lower_limits,upper_limits)
X = X#.to('cuda')
# Y = ((prob_in_collision_function(X,point_cloud,self.obstacle_radii.repeat(point_cloud_size,1),covariance.repeat(point_cloud_size,1,1)) >= probability_threshold) *2-1  ).reshape(-1,self.Y_buffer.shape[1])
# covariance: float = 1e-2
# probability_threshold: float = 0.8
covariance = (torch.eye(3,device="cuda")*1e-2)
for i in range(len(svm_pipeline.recording["r1"])):
    point_cloud, fk_S, W = svm_pipeline.recording["r1"][i]
    print(fk_S.shape)
    W = torch.as_tensor(W,device = "cuda")
    fk_S = torch.as_tensor(fk_S,device = "cuda")
    W, constant, pol_W = W[:fk_S.shape[0]],W[fk_S.shape[0]], W[fk_S.shape[0]+1:]
    point_cloud_size = point_cloud.shape[0]
    threshold = 0.8
    g = []
    for j in range(10):
        Xl = X[1000*j:1000*(j+1)]
        ground_truth = (prob_in_collision_function(Xl,point_cloud,worker.obstacle_radii.repeat(point_cloud_size,1),covariance.repeat(point_cloud_size,1,1)) > threshold)*2-1
        g.append(ground_truth)
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    # K = kernel_matrix_function(X)
    ground_truth = torch.vstack(g).reshape(-1,1)
    fk_vector = robots[0].forward_kinematic_diff_co(X).reshape(-1,15).to(torch.float32)

    predicted = torch.sign(torch.vmap(polyharmonic_kernel,in_dims = (0,None))(fk_vector,fk_S)@W + constant + fk_vector@pol_W)
    print(torch.nonzero(predicted == ground_truth).shape)
    time.sleep(0.2)
# score = (polyharmonic_kernel(fk_vector,fk_support_vectors.view(fk_support_vectors.shape[0],-1))@polyharmonic_weights + constant + torch.dot(pol_weights.squeeze(),fk_vector.squeeze()))
                # x_initial_guess = self.initial_guess_control_points[i]
                # 


# In[ ]:


torch.cuda.empty_cache()
import gc
gc.collect()


# In[ ]:





# In[ ]:


fk_vector.shape


# In[ ]:


print("num support vectors",svm_pipeline.workers[0].num_support_vectors,svm_pipeline.workers[1].num_support_vectors)


# In[ ]:


worker.W_buffer.max()


# In[ ]:


worker = svm_pipeline.workers[0]
fk = worker.fk_X_buffer[:worker.num_support_vectors]
in_col = worker.Y_buffer[:worker.num_support_vectors] == 1
meshcat.SetObject('/pc',torch_to_drake_point_cloud(worker.fk_X_buffer[:worker.num_support_vectors][0].reshape(-1,3)),0.01,Rgba(1,0,0,0.5))


# In[ ]:


svm_pipeline.point_cloud_torch_gpu.shape
# meshcat.SetObject('/aaaaa',torch_to_drake_point_cloud(svm_pipeline.point_cloud_torch_gpu[:300]),0.01,Rgba(1,1,1,0.5))


# In[ ]:


self = worker = svm_pipeline.workers[0]
kernel_matrix_function = (torch.export.load(self.kernel_matrix_function_name).module().to(self.device))
in_collision_function = (torch.export.load(self.in_collision_function_name).module().to(self.device))


# In[ ]:


get_ipython().run_cell_magic('time', '', "point_cloud_size = svm_pipeline.point_cloud_torch_gpu.shape[0]\n\n\nlast_num_support_vectors = worker.num_support_vectors\nlast_support_vectors = worker.X_buffer[:last_num_support_vectors].clone()\nlast_weights = worker.W_buffer[:last_num_support_vectors].clone()\nlast_H_s = worker.H_s_buffer[:last_num_support_vectors].clone()\nlower_limits = torch.tensor(robots[0].plant.GetPositionLowerLimits(),dtype = torch.float32,device = 'cuda')\nupper_limits = torch.tensor(robots[0].plant.GetPositionUpperLimits(),dtype = torch.float32,device = 'cuda')\n# print(last_H_s)\n\n# get exploration samples based on saved trajectories\nnum_exploration_samples = 1000\nnum_old_samples = 1000\nx_initial_guess = svm_pipeline.initial_guess_control_points[i]\nrandom_indices = torch.randperm(x_initial_guess.shape[0])[:num_exploration_samples]\nexploration_samples = x_initial_guess[random_indices]\nexploration_samples += torch.randn_like(exploration_samples)*0.5\nexploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)\nexploration_samples = exploration_samples#.to('cuda')\nactual_num_samples = min(num_old_samples,8*(last_support_vectors.shape[0])//8)\nold_samples_random_indices = torch.randperm(last_support_vectors.shape[0])[:actual_num_samples]\nold_samples = last_support_vectors[old_samples_random_indices]\nnew_X = torch.vstack((old_samples,exploration_samples))\nnew_H_s = torch.zeros((new_X.shape[0],last_H_s.shape[1]),device='cuda')\nnew_W = torch.zeros((new_X.shape[0],last_H_s.shape[1]),device='cuda')\n# new_H_s[:actual_num_samples] = last_H_s[old_samples_random_indices].clone()\n# new_W[:actual_num_samples] = last_weights[old_samples_random_indices].clone()\n# old_samples = last_support_vectors\n# new_X = torch.vstack((old_samples,exploration_samples))\n# new_H_s = torch.zeros((new_X.shape[0],last_H_s.shape[1]),device='cuda')\n# new_W = torch.zeros((new_X.shape[0],last_H_s.shape[1]),device='cuda')\n# new_H_s[:last_support_vectors.shape[0]] = last_H_s[:].clone()/10\n# new_W[:last_support_vectors.shape[0]] = last_weights[:].clone()/10\n\nsample_size = new_X.shape[0]\n\npoint_cloud = self.point_cloud_buffer[:point_cloud_size]\nX = new_X\nH_s_np = new_H_s.cpu().numpy()\nW_np = new_W.cpu().numpy()\nt0 = time.perf_counter()\nfk_X,K = kernel_matrix_function(X)\nfk_X = fk_X.reshape(-1,fk_X.shape[1]*fk_X.shape[2])\nkernel_calculation_time = time.perf_counter()-t0\nt0 = time.perf_counter()\n  \nY = ((in_collision_function(X,point_cloud,self.obstacle_radii.repeat(point_cloud_size,1))*2-1  ).reshape(-1,self.Y_buffer.shape[1])).to(torch.float32)\nin_collision_calculation_time = time.perf_counter()-t0\n\n\n\nK_np = K.cpu().numpy()\n# del K\n# fk_X_np = fk_X.cpu().numpy()\nY_np = Y.cpu().numpy()\n\nt0 = time.perf_counter()\nsupport_vector_indices, completed, max_margin, num_mislabeled = diff_co.calculate_support_vectors_indexes_numba(None, Y_np, H=H_s_np, W=W_np, K=K_np, MAX_ITERATION=78000)\nprint(support_vector_indices.shape,max_margin,completed\n      )\nprint(time.perf_counter()-t0)\n# plt.plot(W_np)\n# plt.plot(last_weights[:].cpu())\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# old_samples = last_support_vectors[old_samples_random_indices]\npoint_cloud_size = svm_pipeline.point_cloud_torch_gpu.shape[0]\n\n\nlast_num_support_vectors = worker.num_support_vectors\nlast_support_vectors = worker.X_buffer[:last_num_support_vectors].clone()\nlast_weights = worker.W_buffer[:last_num_support_vectors].clone()\nlast_H_s = worker.H_s_buffer[:last_num_support_vectors].clone()\npoint_cloud = self.point_cloud_buffer[:point_cloud_size]\n\nlower_limits = torch.tensor(robots[0].plant.GetPositionLowerLimits(),dtype = torch.float32,device = 'cuda')\nupper_limits = torch.tensor(robots[0].plant.GetPositionUpperLimits(),dtype = torch.float32,device = 'cuda')\n# print(last_H_s)\n\n# get exploration samples based on saved trajectories\nnum_exploration_samples = 0\nnum_old_samples = 1000\nx_initial_guess = svm_pipeline.initial_guess_control_points[i]\n\nsample_size = last_support_vectors.shape[0]\n\n\nH_s_np = np.zeros((last_support_vectors.shape[0],last_H_s.shape[1]))\nW_np = np.zeros((last_support_vectors.shape[0],last_H_s.shape[1]))\nt0 = time.perf_counter()\nfk_X,K = kernel_matrix_function(last_support_vectors)\nfk_X = fk_X.reshape(-1,fk_X.shape[1]*fk_X.shape[2])\nY = ((in_collision_function(last_support_vectors,point_cloud,self.obstacle_radii.repeat(point_cloud_size,1))*2-1  ).reshape(-1,self.Y_buffer.shape[1])).to(torch.float32)\n\nK_np = K.cpu().numpy()\nY_np = Y.cpu().numpy()\nt0 = time.perf_counter()\nsupport_vector_indices, completed, max_margin, num_mislabeled = diff_co.calculate_support_vectors_indexes_numba(None, Y_np, H=H_s_np, W=W_np, K=K_np, MAX_ITERATION=78000)\nprint(time.perf_counter()-t0,last_support_vectors.shape,max_margin,support_vector_indices.shape,\n      )\nindices_torch = torch.as_tensor(support_vector_indices,device = last_support_vectors.device).squeeze()\nold_new_support_vectors = last_support_vectors[indices_torch]\nold_new_Y = Y[indices_torch]\n\nold_new_weights = W_np[support_vector_indices.squeeze()]\nold_new_H = H_s_np[support_vector_indices.squeeze()]\n\nnum_exploration_samples = 1000\nx_initial_guess = svm_pipeline.initial_guess_control_points[i]\nrandom_indices = torch.randperm(x_initial_guess.shape[0])[:num_exploration_samples]\nexploration_samples = x_initial_guess[random_indices]\nexploration_samples += torch.randn_like(exploration_samples)*0.5\nexploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)\nexploration_samples = exploration_samples#.to('cuda')\n\nnew_X = torch.vstack((old_new_support_vectors,exploration_samples))\n\nfk_X,K = kernel_matrix_function(new_X)\nfk_X = fk_X.reshape(-1,fk_X.shape[1]*fk_X.shape[2])\n  \nY = ((in_collision_function(exploration_samples,point_cloud,self.obstacle_radii.repeat(point_cloud_size,1))*2-1  ).reshape(-1,self.Y_buffer.shape[1])).to(torch.float32)\n\nY = torch.vstack([old_new_Y,Y])\n\nH_s_np = np.zeros((new_X.shape[0],last_H_s.shape[1]))\nW_np = np.zeros((new_X.shape[0],last_H_s.shape[1]))\n\nH_s_np[:old_new_support_vectors.shape[0]] = old_new_H\nW_np[:old_new_support_vectors.shape[0]] = old_new_weights\n\n\nK_np = K.cpu().numpy()\n# del K\nY_np = Y.cpu().numpy()\n\nt0 = time.perf_counter()\nsupport_vector_indices, completed, max_margin, num_mislabeled = diff_co.calculate_support_vectors_indexes_numba(None, Y_np, H=H_s_np, W=W_np, K=K_np, MAX_ITERATION=78000)\nprint(support_vector_indices.shape,max_margin,completed\n      )\nprint(time.perf_counter()-t0)\nplt.plot(W_np)\n")


# In[ ]:


old_new_weights.shape


# In[ ]:


diff_co.calculate_support_vectors_indexes_numba(None, Y_np, H=H_s_np, W=W_np, K=K_np, MAX_ITERATION=78000)


# In[ ]:


H_s_np.shape


# In[ ]:


# in_col = in_collision_function(X,point_cloud,worker.obstacle_radii.repeat(point_cloud.shape[0],1))
in_col = worker.Y_buffer[:worker.num_support_vectors] == 1
x_in_col = X[in_col.squeeze()].cpu().numpy()
for x in x_in_col:
    viz_helper.set_position('robot_1', x )
    viz_helper.publish()
    time.sleep(1)


# In[ ]:


# viz_helper.diagram.ForcedPublish(viz_helper.diagram_context)


# In[ ]:


worker.X_buffer[5].cpu()


# In[ ]:


c = torch.compile(robots[0].collision_sampler.robot_point_positions_func_pytorch)
c(torch.zeros((9)))


# In[ ]:


viz_helper.set_free_body_position('obstacle_6',RigidTransform(p=[0.0,-0.1,0.05],rpy=RollPitchYaw([0.1,0.0,0.0])))
viz_helper.publish()



# In[ ]:


for i in range(2):
    Y_support_vectors,fk_support_vectors = svm_pipeline.Y_support_vectors[i],svm_pipeline.fk_support_vectors[i]
    N, d = fk_support_vectors.shape
    K_ph = torch.vmap(polyharmonic_kernel, in_dims = (0,None))(fk_support_vectors.view(fk_support_vectors.shape[0],-1),fk_support_vectors.view(fk_support_vectors.shape[0],-1))
    B = torch.cat([torch.ones((N, 1)), fk_support_vectors], dim=1)
    top = torch.cat([K_ph, B], dim=1)
    bottom = torch.cat([B.T, torch.zeros((d+1, d+1))], dim=1)
    M = torch.cat([top, bottom], dim=0)
    f = torch.cat([Y_support_vectors, torch.zeros((d+1, 1))])
    polyharmonic_weights = torch.linalg.solve(M, f)
    polyharmonic_weights, constant, pol_weights = polyharmonic_weights[:N], polyharmonic_weights[N], polyharmonic_weights[N+1:]
    torch.save({'polyharmonic_weights':polyharmonic_weights,'constant':constant,'pol_weights':pol_weights, 'fk_support_vectors':fk_support_vectors},str(TEMP_FOLDER / f'polyharmonic_weights_{i}.pt'),)

# polyharmonic_weights, constant, pol_weights = polyharmonic_weights[:N], polyharmonic_weights[-1],polyharmonic_weights[N:-1]
# 
# polyharmonic_weights = torch.linalg.solve(K_ph, Y_support_vectors.to(torch.float32))
# now read the files:
# torch.load(str(TEMP_FOLDER / f'polyharmonic_weights_{i}.pt'))


# In[ ]:





# In[ ]:


i = 1
# Y_support_vectors,fk_support_vectors = svm_pipeline. [i],svm_pipeline.fk_support_vectors[i]
polyharmonic_weights = svm_pipeline.last_weights[i]
fk_support_vectors = svm_pipeline.last_fk_support_vectors[i]
N = fk_support_vectors.shape[0]
polyharmonic_weights, constant, pol_weights = polyharmonic_weights[:N], polyharmonic_weights[N], polyharmonic_weights[N+1:]
print(polyharmonic_weights)


# In[ ]:


polyharmonic_weights.shape


# In[ ]:


fk_vector = fk_support_vectors[0]
(polyharmonic_kernel(fk_vector,fk_support_vectors.view(fk_support_vectors.shape[0],-1))@polyharmonic_weights + constant + torch.dot(pol_weights.squeeze(),fk_vector.squeeze()))
# (polyharmonic_kernel(fk_support_vectors[1],fk_support_vectors.view(fk_support_vectors.shape[0],-1))@polyharmonic_weights)


# In[ ]:


for i in range(10000):
    meshcat.Delete(f'/teste/fk_point_cloud_{i}')
for i,fk_vector in enumerate(fk_support_vectors):
    meshcat.SetObject(f'/teste/fk_point_cloud_{i}',torch_to_drake_point_cloud(fk_vector.reshape(-1,3)),0.05,Rgba(1,0,0,0.5))


# In[ ]:


for i in range(10000):
    meshcat.Delete(f'/arrow_{i}')


# In[ ]:


for q7 in np.linspace(-3,3,20):
    ik = robots[1].EE_IK(RigidTransform(p=[0.0,-0.1,0.3],rpy=RollPitchYaw([np.pi/2,0.0,0.0])).GetAsMatrix4(),q7,np.zeros((7,1)),).full()
    ik = ik[np.all(~np.isnan(ik),axis=1)]
    if ik.shape[0] > 0:
        print(ik)
        viz_helper.set_position('robot_1',np.concatenate([ik[0],np.zeros((2,))]))
        viz_helper.publish()
        break


# In[ ]:


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

kk = 0
num_x,num_y,num_z = 1,30,25
value = np.zeros((num_x,num_y,num_z))
gradient = np.zeros((num_x,num_y,num_z,2))
configurations = torch.zeros((num_x,num_y,num_z,7))
for i,x in enumerate(np.linspace(-0.1,0.1,num_x)):
    for j,y in enumerate(np.linspace(-0.4,0.2,num_y)):
        for k,z in enumerate(np.linspace(0.0,0.5,num_z)):
            # fk_vector = torch.tensor([0.0,y,z,] + [0]*12,dtype=torch.float32,requires_grad=True)
            for q7 in np.linspace(-1,1,20):
                ik = robots[1].EE_IK(RigidTransform(p=[x,y,z],rpy=RollPitchYaw([-np.pi,0.0,0.0])).GetAsMatrix4(),q7,np.zeros((7,1)),).full()
                ik = ik[np.all(~np.isnan(ik),axis=1)]
                if ik.shape[0] > 0:
                    
                    break
            if ik.shape[0] == 0:
                continue
            configurations[i,j,k] = torch.from_numpy(ik[0])
            # viz_helper.set_position('robot_1',np.concatenate([ik[0],np.zeros((2,))]))
            # viz_helper.publish()
            fk_vector = robots[1].forward_kinematic_diff_co(torch.from_numpy(ik[0])).reshape(-1,15).to(torch.float32)
            fk_vector.requires_grad = True
            score = (polyharmonic_kernel(fk_vector,fk_support_vectors.view(fk_support_vectors.shape[0],-1))@polyharmonic_weights + constant + torch.dot(pol_weights.squeeze(),fk_vector.squeeze()))
            # score = -polyharmonic_kernel(fk_vector,fk_support_vectors.view(fk_support_vectors.shape[0],-1))@polyharmonic_weights
            value[i,j,k] = score
            score.backward()
            gradient[i,j,k] = fk_vector.grad.squeeze()[1:3]
            print(score)


            # direction_EE = fk_vector.grad.view(-1,3)[0]
            # arrow_size_EE = torch.linalg.norm(direction_EE)/6000
            # meshcat_arrow(meshcat,f'/arrow_{kk}',[x,fk_vector[0,1].detach().numpy(),fk_vector[0,2].detach().numpy()],direction_EE.cpu().numpy(),arrow_size_EE.cpu().numpy(),arrow_size_EE.cpu().numpy()/10)
            kk+=1
        # asdfd
        # meshcat.Delete(f'/arrow_{y}_{z}')
# v = torch.from_numpy(fk).to(torch.float32).view(-1)
# v.requires_grad = True
# score = polyharmonic_kernel(v,polyharmonic_sv.view(polyharmonic_sv.shape[0],-1))@polyharmonic_weights


# In[ ]:


batch_over_i_configs = torch.vmap(robots[1].collision_sampler.configuration_is_in_collision, in_dims = (0,None,None))

Ys = []
for y in range(num_y):
    new_X = configurations[0][y].reshape(-1,7)
    new_X = torch.concatenate([new_X,torch.zeros((new_X.shape[0],2))],axis=1).to('cuda')
    collision_robot = batch_over_i_configs(new_X,svm_pipeline.point_cloud_torch_gpu,svm_pipeline.obstacle_radii)
    new_Y = ((einops.rearrange([collision_robot['obstacle_collision'],], "b n -> n b",).any(dim=-1)*2-1).to(torch.float32))
    Ys.append(new_Y)
Ys = torch.vstack(Ys)
# for j,Yy in enumerate(Ys):
#     for k,Yz in enumerate(Yy):
#         if Yz == 1:
#             q = configurations[0,j,k].detach().numpy()
#             viz_helper.set_position('robot_1',np.concatenate([q,np.zeros((2,))]))
#             viz_helper.publish()
#             input()


# In[ ]:


viz_helper.set_position('robot_1',np.concatenate([q,np.zeros((2,))]))
viz_helper.publish()


# In[ ]:


# meshcat.Set2dRenderMode()
# meshcat.Set2dRenderMode(False)
meshcat.ResetRenderMode()
meshcat.SetCamera(meshcat.OrthographicCamera())
# meshcat.Set
meshcat.SetCameraPose(np.array([2.0,0,0],),np.array([-1,0,0.]))
# meshcat.SetCameraTarget(np.array([-,0,0.]))


# In[ ]:


v = value[0].copy()
# v[0,:] = range(num_y)
# v[:,0] = range(num_y)
# extent = [x_min , x_max, y_min , y_max]
extent = [-0.4,0.2,0.0,0.5]
plt.imshow(v.T[::-1,:]>0,extent=extent)
plt.colorbar()
plt.figure()
plt.imshow(Ys.T.cpu().numpy()[::-1,:])
# plt.imshow(v.T[::-1,:])


y, x = np.mgrid[0:v.shape[0], 0:v.shape[1]]

# plt.quiver(x, y, gradient[0][::-1,:,0], -gradient[0][:,:,1], color='black')


# In[ ]:


v.shape


# In[ ]:


fk_vector = torch.tensor([0.0,y,z,] + [0]*12,dtype=torch.float32,requires_grad=True)
# fk_vector = fk_vector1
# fk_vector.requires_grad = True
score = polyharmonic_kernel(fk_vector,fk_support_vectors.view(fk_support_vectors.shape[0],-1))@polyharmonic_weights
score.backward()
fk_vector.grad


# In[ ]:





# In[ ]:


for i in range(12):
    indices = point_cloud_filter.work_tensors[0]['gaze_indices_buffer']
    a,b,c,d = indices[i]
    vertices = point_cloud_filter.work_tensors[0]['gaze_polytopes_buffer'][a:b]
    simplices = point_cloud_filter.work_tensors[0]['gaze_simplices_buffer'][c:d]
    faces = vertices[simplices]
    meshcat.SetTriangleMesh(f'polytope_{i}',vertices.T,simplices.T)


# In[ ]:


point_cloud_filter.work_tensors[0]['gaze_indices_buffer'][0:20]


# In[ ]:


context = lcm_camera_1_pose_to_transform.GetMyContextFromRoot(diagram_context)
lcm_camera_1_pose_to_transform.get_output_port().Eval(context)
context = lcm_camera_2_pose_to_transform.GetMyContextFromRoot(diagram_context)
lcm_camera_2_pose_to_transform.get_output_port().Eval(context)


# In[ ]:


point_cloud_filter.num_polytopes


# In[ ]:


self = point_cloud_filter
context = point_cloud_filter.GetMyMutableContextFromRoot(diagram_context)
voxel_size = self.voxel_size
print("calc_point_cloud filter")
original_point_cloud:PointCloud = self.point_cloud_input_port.Eval(context).Crop([-10,-10,0.2],[10,10,10])
robot_1_position_lcm = self.robot_1_position_port.Eval(context)
robot_2_position_lcm = self.robot_2_position_port.Eval(context)


original_point_cloud = original_point_cloud.VoxelizedDownSample(voxel_size=voxel_size)

meshcat.SetObject('/point_cloud_incoming',original_point_cloud,0.01,Rgba(1,1,1,0.5))

# camera_pose = torch.tensor(self.camera_pose_input_port.Eval(context).GetAsMatrix4()).to(self.device,dtype=torch.float32)
world_to_camera = torch.from_numpy(np.linalg.inv(self.camera_pose_input_port.Eval(context).GetAsMatrix4())).to(self.device)
robot_1_positions = torch.tensor(robot_1_position_lcm.joint_position,dtype=torch.float32,device=self.device)
robot_2_positions = torch.tensor(robot_2_position_lcm.joint_position,dtype=torch.float32,device=self.device)
point_cloud_1_torch = torch.from_numpy(original_point_cloud.xyzs()).to(self.device).T


# In[ ]:


configurations = {
                'robot_geometry_centers_function_filename_1':'robot_collision_geometry_positions_1',
                'robot_geometry_centers_function_filename_2':'robot_collision_geometry_positions_2',
                    'robot_geometry_centers_function_name':self.robots[0].collision_sampler.robot_point_positions_func_casadi.name(),
                    'codegen_path':CODEGEN_FOLDER / 'misc_functions'}
robot_geometry_centers_func_1 = ca_utils.Compile(file_name = configurations['robot_geometry_centers_function_filename_1'],path = configurations['codegen_path'],function_name = configurations['robot_geometry_centers_function_name'],get_cached_or_throw = True)
robot_geometry_centers_func_2 = ca_utils.Compile(file_name = configurations['robot_geometry_centers_function_filename_2'],path = configurations['codegen_path'],function_name = configurations['robot_geometry_centers_function_name'],get_cached_or_throw = True)


# In[ ]:


point_cloud_size = point_cloud_1_torch.shape[0]
point_cloud_torch_buffer = self.work_tensors[0]['point_cloud_buffer']
last_point_cloud_buffer = self.work_tensors[0]['last_point_cloud_buffer']
robot_1_positions_buffer = self.work_tensors[0]['robot_1_positions_buffer']
robot_2_positions_buffer = self.work_tensors[0]['robot_2_positions_buffer']
world_to_camera_buffer = self.work_tensors[0]['world_to_camera_buffer']
robot_geometry_radii_buffer = self.work_tensors[0]['robot_geometry_radii_buffer']

