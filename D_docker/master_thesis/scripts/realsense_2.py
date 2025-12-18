import pyrealsense2 as rs
from multiprocessing import shared_memory
from pydrake.all import ImageDepth16U,ImageDepth32F,ConvertDepth16UTo32F
from multiprocessing.resource_tracker import unregister
import numpy as np
import time
import psutil
from dt_apriltags import Detector
import cv2
p = psutil.Process()
p.cpu_affinity([6,7,8,9,10,11])
at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=2,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
tag_size = 0.061

dtype = np.float32
array_shape = (480, 640, 1)
shared_memory_depth_name = "depth_image_2"
try:
    shm_depth = shared_memory.SharedMemory(name=shared_memory_depth_name, create=True, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
    shared_depth = np.ndarray(array_shape, dtype=dtype, buffer=shm_depth.buf)
    shared_depth[:] = np.nan
    print("Created new shared memory.")
except FileExistsError:
    shm_depth = shared_memory.SharedMemory(name=shared_memory_depth_name)
    shared_depth = np.ndarray(array_shape, dtype=dtype, buffer=shm_depth.buf)
    print("Linked to existing shared memory.")
dtype = np.uint8
array_shape = (480, 640, 3)
shared_memory_color_name = "color_image_2"
try:
    shm_color = shared_memory.SharedMemory(name=shared_memory_color_name, create=True, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
    shared_color = np.ndarray(array_shape, dtype=dtype, buffer=shm_color.buf)
    # shared_color[:] = np.nan
    print("Created new shared memory.")
except FileExistsError:
    shm_color = shared_memory.SharedMemory(name=shared_memory_color_name)
    shared_color = np.ndarray(array_shape, dtype=dtype, buffer=shm_color.buf)
    print("Linked to existing shared memory.")


dtype = np.float32
num_tags_max = 10
array_shape = (4, 4, num_tags_max)
shared_memory_at_name = "apriltags_2"
try:
    shm_at = shared_memory.SharedMemory(name=shared_memory_at_name, create=True, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
    shared_at = np.ndarray(array_shape, dtype=dtype, buffer=shm_at.buf)
    shared_at[:] = np.nan
    print("Created new shared memory.")
except FileExistsError:
    shm_at = shared_memory.SharedMemory(name=shared_memory_at_name)
    shared_at = np.ndarray(array_shape, dtype=dtype, buffer=shm_at.buf)
    print("Linked to existing shared memory.")

camera_1_serial = "046122251075"
# camera_1_serial = "046122251304"
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(camera_1_serial)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
import time
pipeline_1.start(config_1)
profile = pipeline_1.get_active_profile()

depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
focal_x = depth_intrinsics.fx
focal_y = depth_intrinsics.fy
width,  height = depth_intrinsics.width,depth_intrinsics.height
center_x, center_y = depth_intrinsics.ppx, depth_intrinsics.ppy

print(focal_x,focal_y,center_x,center_y)

color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
focal_x = color_intrinsics.fx
focal_y = color_intrinsics.fy
width,  height = color_intrinsics.width,color_intrinsics.height
center_x, center_y = color_intrinsics.ppx, color_intrinsics.ppy
camera_1_params_at = ( focal_x, focal_y, center_x, center_y )
print(focal_x,focal_y,center_x,center_y)
camera_1_extrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color).as_video_stream_profile())
print('extrinsics',camera_1_extrinsics)
t0 = time.perf_counter()
try:
    while True:
        frame_received, frame = pipeline_1.try_wait_for_frames(10)
        if frame_received:
            # print('tloop',t0-time.perf_counter())
            t0 = time.perf_counter()
            depth_frame = frame.get_depth_frame()
            width = depth_frame.width
            height = depth_frame.height

            image_= ImageDepth16U(width,height)
            image_.mutable_data[:] = np.frombuffer(depth_frame.get_data(),dtype=np.uint16).reshape(height,width,1)
            image_32F = ImageDepth32F(width,height)
            image_ = ConvertDepth16UTo32F(image_,image_32F)
            shared_depth[:] = image_32F.data
            color_frame = frame.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            shared_color[:] = color_image
            # print('t1',time.perf_counter() - t0)
            tags = at_detector.detect(cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=camera_1_params_at, tag_size=tag_size)
            # print('t2',time.perf_counter() - t0)
            if len(tags) > 0:
                shared_at[:] = np.nan
                # print(len(tags))
                for i,tag in enumerate(tags):
                    shared_at[:3,:3,i] = tag.pose_R
                    shared_at[:3,3,i] = tag.pose_t.squeeze()
                    shared_at[3,0,i] = time.perf_counter()
                    shared_at[3,1,i] = tag.tag_id

            # if len (tags) > 0:
            #     print(tags[0])
except:
    pass
finally:
    pipeline_1.stop()
    shm_color.close()
    shm_depth.close()
    shm_color.unlink()
    shm_depth.unlink()