from pydrake.all import Value, ImageDepth32F,CameraInfo,RotationMatrix,ConvertDepth16UTo32F, ImageDepth16U, LeafSystem, RigidTransform, PixelType
from drake import lcmt_image_array
import numpy as np
import time
from diff_co_mpc.diff_co_lcm import lcmt_pose, lcmt_global_solve
from drake import lcmt_robot_state
from utils.math.BSpline import BSpline

try:
    import pyrealsense2 as rs
except:
    print("pyrealsense2 not found")

try:
    from dt_apriltags import Detector
except:
    print("dt_apriltags not found")
try:
    import rospy as ros
    from sensor_msgs.msg import JointState
except:
    print("ros not found")
class CustomLCMToImages(LeafSystem):
    def __init__(self, ):
        super().__init__()
        # because the normal one doesn't work
        self.lcm_image_array_input_port = self.DeclareAbstractInputPort(name="in_lcm_array",
                                      model_value=Value(lcmt_image_array()))
        self.depth_image_output_port = self.DeclareAbstractOutputPort(
            name="out",
            alloc=lambda: Value(ImageDepth32F()),
            # alloc=lambda: Value(ImageDepth16U()),
            calc=self.calc_depth)
        
        
    def calc_depth(self, context, output):
        t0 = time.perf_counter()
        lcmt_array = self.lcm_image_array_input_port.Eval(context)
        for image in lcmt_array.images:
            # for some reason it always receives it as 16U
            if PixelType(image.pixel_format) == PixelType.kDepth32F or PixelType(image.pixel_format) == PixelType.kDepth16U:
                width = image.width
                height = image.height

                # np.frombuffer(image.data,dtype=np.float32).reshape(height,width,1)
                image_= ImageDepth32F(width,height)
                image_.mutable_data[:] = np.frombuffer(image.data,dtype=np.float32).reshape(height,width,1)
                output.set_value(image_)
                return
            raise ValueError("No depth image found")
        
class PoseLCMToTransform(LeafSystem):
    def __init__(self, ):
        super().__init__()
        self.pose_input_port = self.DeclareAbstractInputPort(name="pose_lcm",
                                      model_value=Value(lcmt_pose()))
        self.transform_output_port = self.DeclareAbstractOutputPort(
            name="out",
            alloc=lambda: Value(RigidTransform()),
            calc=self.calc_transform)
    def calc_transform(self, context, output):
        pose = np.asarray(self.pose_input_port.Eval(context).pose).reshape(4,4)
        output.set_value(RigidTransform(pose)) 

from multiprocessing import shared_memory
from multiprocessing import Process, resource_tracker
def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]
import cv2
from multiprocessing.resource_tracker import unregister

class RealsenseReader(LeafSystem):
    def __init__(self, ):
        super().__init__()
        remove_shm_from_resource_tracker()
        width, height = 640,480
        self.width = width
        self.height = height
        # camera 1:
        # depth
        focal_x,focal_y,center_x,center_y = 386.07855224609375, 386.07855224609375, 319.1185607910156, 238.21234130859375
        self.depth_camera_1_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)
        self.camera_1_params_depth = ( self.depth_camera_1_drake_info.focal_x(), self.depth_camera_1_drake_info.focal_y(), self.depth_camera_1_drake_info.center_x(), self.depth_camera_1_drake_info.center_y() )
        # color
        focal_x,focal_y,center_x,center_y = 384.0457458496094, 383.09625244140625, 313.0089111328125, 250.05235290527344
        self.color_camera_1_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)
        # camera 2:
        # depth
        focal_x,focal_y,center_x,center_y = 389.1177673339844, 389.1177673339844, 317.7033386230469, 238.56051635742188
        self.depth_camera_2_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)
        self.camera_2_params_depth = ( self.depth_camera_2_drake_info.focal_x(), self.depth_camera_2_drake_info.focal_y(), self.depth_camera_2_drake_info.center_x(), self.depth_camera_2_drake_info.center_y() )
        # color
        focal_x,focal_y,center_x,center_y = 386.09088134765625, 385.0380554199219, 309.70367431640625, 241.99278259277344
        self.color_camera_2_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)

        self.camera_1_extrinsic_t = np.array([-0.0593118, 0.000105446, 0.000400316]).reshape(3,1)
        self.camera_1_extrinsic_R = np.array([0.999997, -0.00226942, -0.000883205, 0.0022657, 0.999989, -0.00418478, 0.000892692, 0.00418276, 0.999991]).reshape(3,3)
        self.camera_2_extrinsic_t = np.array([-0.0594441, -0.000257048, 0.000345637]).reshape(3,1)
        self.camera_2_extrinsic_R = np.array([0.999989, 0.00467125, -0.0010296, -0.00467758, 0.99997, -0.00623559, 0.00100044, 0.00624033, 0.99998]).reshape(3,3)

        dtype = np.float32
        array_shape = (480, 640, 1)
        shared_memory_depth_name = "depth_image_2"
        # try:
        self.shm_depth_2 = shared_memory.SharedMemory(name=shared_memory_depth_name, create=False, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
        self.depth_image_2 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_depth_2.buf)

        dtype = np.float32
        num_tags_max = 10
        array_shape = (4, 4, num_tags_max)
        shared_memory_at_name = "apriltags_2"
        try:
            self.shm_at_2 = shared_memory.SharedMemory(name=shared_memory_at_name, create=True, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
            self.apriltags_2 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_at_2.buf)
            self.apriltags_2[:] = np.nan
            print("Created new shared memory.")
        except FileExistsError:
            self.shm_at_2 = shared_memory.SharedMemory(name=shared_memory_at_name)
            self.apriltags_2 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_at_2.buf)
            print("Linked to existing shared memory.")

        dtype = np.uint8
        array_shape = (480, 640, 3)
        shared_memory_color_name = "color_image_2"
        # try:
        self.shm_color_2 = shared_memory.SharedMemory(name=shared_memory_color_name, create=False, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
        self.color_image_2 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_color_2.buf)

        unregister(self.shm_color_2.name, 'shared_memory')
        unregister(self.shm_depth_2.name, 'shared_memory')
        unregister(self.shm_at_2.name, 'shared_memory')

        dtype = np.float32
        array_shape = (480, 640, 1)
        shared_memory_depth_name = "depth_image_1"
        # try:
        self.shm_depth_1 = shared_memory.SharedMemory(name=shared_memory_depth_name, create=False, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
        self.depth_image_1 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_depth_1.buf)

        dtype = np.float32
        num_tags_max = 10
        array_shape = (4, 4, num_tags_max)
        shared_memory_at_name = "apriltags_1"
        try:
            self.shm_at_1 = shared_memory.SharedMemory(name=shared_memory_at_name, create=True, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
            self.apriltags_1 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_at_1.buf)
            self.apriltags_1[:] = np.nan
            print("Created new shared memory.")
        except FileExistsError:
            self.shm_at_1 = shared_memory.SharedMemory(name=shared_memory_at_name)
            self.apriltags_1 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_at_1.buf)
            print("Linked to existing shared memory.")

        dtype = np.uint8
        array_shape = (480, 640, 3)
        shared_memory_color_name = "color_image_1"
        # try:
        self.shm_color_1 = shared_memory.SharedMemory(name=shared_memory_color_name, create=False, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
        self.color_image_1 = np.ndarray(array_shape, dtype=dtype, buffer=self.shm_color_1.buf)

        unregister(self.shm_color_1.name, 'shared_memory')
        unregister(self.shm_depth_1.name, 'shared_memory')
        unregister(self.shm_at_1.name, 'shared_memory')



        self.depth_image_1_output_port = self.DeclareAbstractOutputPort(
            name="out_1",
            alloc=lambda: Value(ImageDepth32F()),
            
            calc=self.get_depth_1)
        self.depth_image_2_output_port = self.DeclareAbstractOutputPort(
            name="out_2",
            alloc=lambda: Value(ImageDepth32F()),
            
            calc=self.get_depth_2)
        # self.init_realsense()
        self.points = np.zeros((0,3))
        self.april_tag_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        self.transform_output_port = self.DeclareAbstractOutputPort(
            name="camera_transform_out",
            alloc=lambda: Value(RigidTransform()),
            calc=self.calc_transform)
        self.tag = None
        self.positions_calibration = []
        self.quaternions_calibration = []
        self.final_apriltag_pose = None
        self.last_depth_1 = ImageDepth32F(20,20)
        self.last_depth_2 = ImageDepth32F(20,20)
    def calc_transform(self,context,output):
        # if self.final_apriltag_pose is not None:
        #     output.set_value(self.final_apriltag_pose) 
        #     return
        # if self.tag is None:
        #     return
        extrinsic_t = self.camera_2_extrinsic_t
        extrinsic_R = self.camera_2_extrinsic_R

        T_depth_in_color = np.eye(4)
        T_depth_in_color[:3,3:] = extrinsic_t*1
        T_depth_in_color[:3,:3] = extrinsic_R.T

        T_AT_in_color = np.array([[ 0.99516671,  0.07074268,  0.06810796,  0.07231502],
       [-0.08373974,  0.24908092,  0.96485561,  0.25031985],
       [ 0.05129208, -0.96589553,  0.25380101,  0.81090267],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

        T_color_in_AT = np.linalg.inv(T_AT_in_color)
        T_w_in_AT = RigidTransform(p = [0,-0.13,0], R = RotationMatrix.MakeXRotation(np.pi)).GetAsMatrix4()
        T_AT_in_w = np.linalg.inv(T_w_in_AT)
        
        T_depth_in_world = T_AT_in_w@T_color_in_AT@T_depth_in_color
        pose = RigidTransform((T_depth_in_world))
        output.set_value(pose) 
        return
    # def init_realsense(self):
    #     camera_1_serial = "046122251304"
    #     camera_2_serial = "046122251075"
    #     pipeline_1 = rs.pipeline()
    #     config_1 = rs.config()
    #     config_1.enable_device(camera_1_serial)
    #     config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    #     # config_1.enable_stream(rs.stream.color, 640, 480, rs.format.y16, 30) #grayscale
    #     config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    #     pipeline_2 = rs.pipeline()
    #     config_2 = rs.config()
    #     config_2.enable_device(camera_2_serial)
    #     config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    #     # config_2.enable_stream(rs.stream.color, 640, 480, rs.format.y16, 30) #grayscale
    #     config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    #     pipeline_1.start(config_1)
    #     time.sleep(1)
    #     try:
    #         pipeline_2.start(config_2)
    #     except:
    #         pipeline_1.stop()
    #         assert False
    #     profile = pipeline_1.get_active_profile()
    #     depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    #     color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    #     focal_x = depth_intrinsics.fx
    #     focal_y = depth_intrinsics.fy
    #     width,  height = depth_intrinsics.width,depth_intrinsics.height
    #     center_x, center_y = depth_intrinsics.ppx, depth_intrinsics.ppy
    #     self.depth_camera_1_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)
    #     focal_x = color_intrinsics.fx
    #     focal_y = color_intrinsics.fy
    #     width,  height = color_intrinsics.width,color_intrinsics.height
    #     center_x, center_y = color_intrinsics.ppx, color_intrinsics.ppy
    #     self.color_camera_1_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)
    #     self.camera_1_params_at = ( self.color_camera_1_drake_info.focal_x(), self.color_camera_1_drake_info.focal_y(), self.color_camera_1_drake_info.center_x(), self.color_camera_1_drake_info.center_y() )
    #     self.camera_1_params_depth = ( self.depth_camera_1_drake_info.focal_x(), self.depth_camera_1_drake_info.focal_y(), self.depth_camera_1_drake_info.center_x(), self.depth_camera_1_drake_info.center_y() )
    #     self.tag_size = 0.061
    #     self.pipeline_1 = pipeline_1
    #     self.config_1 = config_1
    #     self.camera_1_extrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color).as_video_stream_profile())



        
    #     profile = pipeline_2.get_active_profile()
    #     depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    #     color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    #     focal_x = depth_intrinsics.fx
    #     focal_y = depth_intrinsics.fy
    #     width,  height = depth_intrinsics.width,depth_intrinsics.height
    #     center_x, center_y = depth_intrinsics.ppx, depth_intrinsics.ppy
    #     self.depth_camera_2_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)
    #     focal_x = color_intrinsics.fx
    #     focal_y = color_intrinsics.fy
    #     width,  height = color_intrinsics.width,color_intrinsics.height
    #     center_x, center_y = color_intrinsics.ppx, color_intrinsics.ppy
    #     self.color_camera_2_drake_info = CameraInfo(width,  height,focal_x,focal_y,center_x, center_y)
    #     self.camera_2_params_at = ( self.color_camera_2_drake_info.focal_x(), self.color_camera_2_drake_info.focal_y(), self.color_camera_2_drake_info.center_x(), self.color_camera_2_drake_info.center_y() )
    #     self.camera_2_params_depth = ( self.depth_camera_2_drake_info.focal_x(), self.depth_camera_2_drake_info.focal_y(), self.depth_camera_2_drake_info.center_x(), self.depth_camera_2_drake_info.center_y() )
    #     self.tag_size = 0.061
    #     self.pipeline_2 = pipeline_2
    #     self.config_2 = config_2
    #     self.camera_2_extrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color).as_video_stream_profile())

    # def stop(self):
    #     self.pipeline_1.stop()
    #     self.pipeline_2.stop()
    def get_depth_1(self, context, output):
        depth_image = self.depth_image_1.copy()
        if np.any(np.isnan(depth_image)):
            return
        image_32F = ImageDepth32F(self.width,self.height)
        image_32F.mutable_data[:] = depth_image.reshape(self.height,self.width,1)
        output.set_value(image_32F )
        # frame_received, frame = self.pipeline_1.try_wait_for_frames(5)
        # if frame_received:
        #     t0 = time.perf_counter()
        #     # depth_image = np.asanyarray(depth_frame.get_data()).copy()
        #     depth_frame = frame.get_depth_frame()
        #     width = depth_frame.width
        #     height = depth_frame.height
        #     image_= ImageDepth16U(width,height)
        #     image_.mutable_data[:] = np.frombuffer(depth_frame.get_data(),dtype=np.uint16).reshape(height,width,1)
        #     image_32F = ImageDepth32F(width,height)
        #     image_ = ConvertDepth16UTo32F(image_,image_32F)
        #     # output.set_value(image_32F)
        #     self.last_depth_1 = image_32F
        #     color_frame = frame.get_color_frame()
        #     self.color_image = np.asanyarray(color_frame.get_data())
        #     if self.final_apriltag_pose is None:
        #         try:
        #             tags = self.april_tag_detector.detect(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=self.camera_1_params_at, tag_size=self.tag_size)
        #             self.tag = tags[0]
                    
        #         except:
        #             pass
        #     # print("get depth image from rs",time.perf_counter()-t0)
        # output.set_value(self.last_depth_1)
    def get_depth_2(self, context, output):
        depth_image = self.depth_image_2.copy()
        if np.any(np.isnan(depth_image)):
            return
        image_32F = ImageDepth32F(self.width,self.height)
        image_32F.mutable_data[:] = depth_image.reshape(self.height,self.width,1)
        output.set_value(image_32F )
        # frame_received, frame = self.pipeline_2.try_wait_for_frames(5)
        # if frame_received:
        #     t0 = time.perf_counter()
        #     depth_frame = frame.get_depth_frame()
        #     width = depth_frame.width
        #     height = depth_frame.height
        #     image_= ImageDepth16U(width,height)
        #     image_.mutable_data[:] = np.frombuffer(depth_frame.get_data(),dtype=np.uint16).reshape(height,width,1)
        #     image_32F = ImageDepth32F(width,height)
        #     image_ = ConvertDepth16UTo32F(image_,image_32F)
        #     # output.set_value(image_32F)
        #     self.last_depth_2 = image_32F
        #     color_frame = frame.get_color_frame()
        #     self.color_image = np.asanyarray(color_frame.get_data())
        # output.set_value(self.last_depth_2)
            # print("get depth image from rs",time.perf_counter()-t0)
    def average_quaternions(self,quaternions):
        """
        Calculate average quaternion

        :params quaternions: is a Nx4 numpy matrix and contains the quaternions
            to average in the rows.
            The quaternions are arranged as (w,x,y,z), with w being the scalar

        :returns: the average quaternion of the input. Note that the signs
            of the output quaternion can be reversed, since q and -q
            describe the same orientation
        """

        # Number of quaternions to average
        samples = quaternions.shape[0]
        mat_a = np.zeros(shape=(4, 4), dtype=np.float64)

        for i in range(0, samples):
            quat = quaternions[i, :]
            # multiply quat with its transposed version quat' and add mat_a
            mat_a = np.outer(quat, quat) + mat_a

        # scale
        mat_a = (1.0/ samples)*mat_a
        # compute eigenvalues and -vectors
        eigen_values, eigen_vectors = np.linalg.eig(mat_a)
        # Sort by largest eigenvalue
        eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
        # return the real part of the largest eigenvector (has only real part)
        return np.real(np.ravel(eigen_vectors[:, 0]))  
class ROSJointState(LeafSystem):
    
    class ROSJointSubscriber:
        def __init__(self, namespace = ""):
            super().__init__()
            ros.Subscriber(namespace+"/franka_state_controller/joint_states", JointState, self.ros_callback_joints)
            ros.Subscriber(namespace+"/franka_gripper/joint_states", JointState, self.ros_callback_gripper )

            self.joints_1 = np.array([0.]*9)
            self.joints_2 = np.array([0.]*9)
        def ros_callback_gripper(self,msg):
            self.joints_1[-2:]= msg.position
        def ros_callback_joints(self,msg):
            self.joints_1[:7]= msg.position
    class ROSDualJointSubscriber:
        def __init__(self, ):
            super().__init__()
            ros.Subscriber("/panda_1/franka_state_controller/joint_states", JointState, self.ros_callback_joints_1)
            ros.Subscriber("/panda_1/franka_gripper/joint_states", JointState, self.ros_callback_gripper_1 )
            ros.Subscriber("/panda_2/franka_state_controller/joint_states", JointState, self.ros_callback_joints_2)
            ros.Subscriber("/panda_2/franka_gripper/joint_states", JointState, self.ros_callback_gripper_2 )
            self.joints_1 = np.array([0.]*9)
            self.joints_2 = np.array([0.]*9)
            self.i = 0
        def ros_callback_gripper_1(self,msg):
            self.joints_1[-2:]= msg.position
        def ros_callback_joints_1(self,msg):
            self.joints_1[:7]= msg.position
        def ros_callback_gripper_2(self,msg):
            self.joints_2[-2:]= msg.position
        def ros_callback_joints_2(self,msg):
            self.joints_2[:7]= msg.position

    def __init__(self, ):
        super().__init__()
        self.joint_subscriber = ROSJointState.ROSDualJointSubscriber()
        self.robot_1_joint_state_port = self.DeclareAbstractOutputPort(
            name="joints_1",
            alloc=lambda: Value(lcmt_robot_state()),
            calc=self.calc_1)
        self.robot_2_joint_state_port = self.DeclareAbstractOutputPort(
            name="joints_2",
            alloc=lambda: Value(lcmt_robot_state()),
            calc=self.calc_2)
    def calc_1(self, context, output):
        msg = lcmt_robot_state()
        msg.joint_position = self.joint_subscriber.joints_1
        msg.joint_name = ['joint_{}'.format(i) for i in range(7)] + [ "panda_finger_joint1","panda_finger_joint2"]
        msg.num_joints = 9
        output.set_value(msg)
        pass
    def calc_2(self, context, output):
        msg = lcmt_robot_state()
        msg.joint_position = self.joint_subscriber.joints_2
        msg.joint_name = ['joint_{}'.format(i) for i in range(7)] + [ "panda_finger_joint1","panda_finger_joint2"]
        msg.num_joints = 9
        output.set_value(msg)
        pass
from functools import partial
class RobotConfigurationToCameraPose(LeafSystem):
    def __init__(self, robot_camera_depth_in_EE, EE_function):
        super().__init__()
        # self.DeclareForcedPublishEvent(self.publish)
        self.robot_state_input_form = self.DeclareAbstractInputPort(name="in",
                                      model_value=Value(lcmt_robot_state()))
        # self.in_sth = self.DeclareAbstractInputPort(name="i2n",
        #                               model_value=Value(RigidTransform()))
        self.camera_pose_output_port = self.DeclareAbstractOutputPort(name="out",
                                      alloc=lambda: Value(RigidTransform()),
                                      calc = self.calc
                                      )
        self.robot_camera_depth_in_EE = robot_camera_depth_in_EE
        # self.EE_function = partial(
        #     robots[0].wrapper.calc_frame_pose_in_frame, 
        #     frame = robots[0].plant.GetFrameByName("EE_frame"),
        #     frame_expressed =robots[0].plant.world_frame())
        self.EE_function = EE_function
    def calc(self, context, output):
        msg:lcmt_robot_state = self.robot_state_input_form.Eval(context)
        transform = RigidTransform(self.EE_function(msg.joint_position).full()@self.robot_camera_depth_in_EE)
        output.set_value(transform)
        pass
class TrajectoryPlotter(LeafSystem):
    def __init__(self, robots, meshcat):
        super().__init__()
        # self.DeclareForcedPublishEvent(self.publish)
        self.trajectory_input_port = self.DeclareAbstractInputPort(name="in",
                                      model_value=Value(lcmt_global_solve()))
        self.bspline_1 = None
        self.bspline_2 = None
        self.t0 = -1
        self.DeclareForcedPublishEvent(self.publish)

        self.plant_1_context = robots[0].plant.CreateDefaultContext()
        self.plant_2_context = robots[1].plant.CreateDefaultContext()
        self.robots = robots
        self.meshcat = meshcat
    def publish(self, context):
        
        lcmt_trajetory = self.trajectory_input_port.Eval(context)
        try:
            self.bspline_1 = BSpline(np.asarray(lcmt_trajetory.bspline_robot_1.control_points), 3)
            self.bspline_2 = BSpline(np.asarray(lcmt_trajetory.bspline_robot_2.control_points), 3)
            self.bspline_obj = BSpline(np.asarray(lcmt_trajetory.bspline_object.control_points), 3)

            pos_obj = []
            pos_EE_1 = []
            pos_EE_2 = []
            for s in np.linspace(0,1,30):

                q_1 = self.bspline_1.evaluate(s)
                q_2 = self.bspline_2.evaluate(s)
                q_obj = self.bspline_obj.evaluate(s)
                q_1 = np.concatenate([q_1,np.zeros(2)])
                q_2 = np.concatenate([q_2,np.zeros(2)])
                self.robots[0].plant.SetPositions(self.plant_1_context,q_1)
                self.robots[1].plant.SetPositions(self.plant_2_context,q_2)
                pos_EE_1.append(self.robots[0].plant.GetFrameByName("EE_frame").CalcPoseInWorld(self.plant_1_context).translation().reshape(1,-1))
                pos_EE_2.append(self.robots[1].plant.GetFrameByName("EE_frame").CalcPoseInWorld(self.plant_2_context).translation().reshape(1,-1))
                pos_obj.append(q_obj[4:7])
            pos_obj = np.vstack(pos_obj)
            pos_EE_1 = np.vstack(pos_EE_1)
            pos_EE_2 = np.vstack(pos_EE_2)
            self.meshcat.SetLineSegments(f"/plan_obj/",pos_obj[:-1].T,pos_obj[1:].T)
            self.meshcat.SetProperty(f"/plan_obj/","color",[1,0,0,1])
            self.meshcat.SetLineSegments(f"/plan_EE_1/",pos_EE_1[:-1].T,pos_EE_1[1:].T)
            self.meshcat.SetProperty(f"/plan_EE_1/","color",[0,0,0,1])
            self.meshcat.SetLineSegments(f"/plan_EE_2/",pos_EE_2[:-1].T,pos_EE_2[1:].T)
            self.meshcat.SetProperty(f"/plan_EE_2/","color",[0,0,0,1])
                
            # self.meshcat.
        except Exception as e:
            print(e)
