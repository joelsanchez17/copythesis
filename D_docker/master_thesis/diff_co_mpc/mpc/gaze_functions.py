import pydrake.geometry as pydgeo
import math,numba
import numpy as np
from mpc.helper_functions import *
from mpc.inverse_kinematics import rotation_matrix,batch_franka_IK_EE_


@numba.njit(parallel=True)
def calculate_samples(position_samples,distance_samples,phi_samples,theta_samples,gaze_z_angles):
    position_gaze_samples = []
    orientation_gaze_samples = []
    num_samples_spot = position_samples.shape[0]
    num_samples_gaze = distance_samples.shape[0]//num_samples_spot
    
    # position_gaze_samples = np.empty((num_samples_spot*num_samples_gaze,3))
    for i in range(0,num_samples_spot,1):
        for j,angle in enumerate(gaze_z_angles):
            position = np.empty((num_samples_gaze,3))
            orientation = np.empty((num_samples_gaze,3,3))
            position[:,0:1] = np.cos(theta_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])*np.sin(phi_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])*distance_samples[i*num_samples_gaze:(i+1)*num_samples_gaze]
            position[:,1:2] =np.sin(theta_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])*np.sin(phi_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])*distance_samples[i*num_samples_gaze:(i+1)*num_samples_gaze]
            position[:,2:3] = np.cos(phi_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])*distance_samples[i*num_samples_gaze:(i+1)*num_samples_gaze]
            position += position_samples[i]
            position_gaze_samples.append(position)

            z_axis = np.empty((num_samples_gaze,3))
            z_axis[:,0:1] = -np.cos(theta_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])*np.sin(phi_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])
            z_axis[:,1:2] = -np.sin(theta_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])*np.sin(phi_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])
            z_axis[:,2:3] = -np.cos(phi_samples[i*num_samples_gaze:(i+1)*num_samples_gaze])

            x_axis = np.empty((num_samples_gaze,3))
            x_axis[:,0:1] = z_axis[:,1:2]
            x_axis[:,1:2] = -z_axis[:,0:1]
            x_axis[:,2:3] = 0

            rotations = rotation_matrix(z_axis,angle)
            for l in range(num_samples_gaze):
                x_axis[l] = x_axis[l] / np.linalg.norm(x_axis[l])
                x_axis[l] = rotations[l]@x_axis[l]
                
            y_axis = np.cross(z_axis,x_axis)
            orientation = np.empty((num_samples_gaze,3,3))
            for l in range(num_samples_gaze):
                orientation[l][:,0] = x_axis[l]
                orientation[l][:,1] = y_axis[l]
                orientation[l][:,2] = z_axis[l]
            orientation_gaze_samples.append(orientation)
    # print(len(position_gaze_samples),len(orientation_gaze_samples))
    return position_gaze_samples,orientation_gaze_samples
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def calculate_samples(position_samples, distance_samples, phi_samples, theta_samples, gaze_z_angles):
    num_samples_spot = position_samples.shape[0]
    num_samples_gaze = distance_samples.shape[0] // num_samples_spot
    
    position_gaze_samples = np.empty((num_samples_spot, len(gaze_z_angles), num_samples_gaze, 3))
    orientation_gaze_samples = np.empty((num_samples_spot, len(gaze_z_angles), num_samples_gaze, 3, 3))
    pose_gaze_samples = np.empty((num_samples_spot, len(gaze_z_angles), num_samples_gaze, 4, 4))
    for i in prange(num_samples_spot):
        for j, angle in enumerate(gaze_z_angles):
            position = np.empty((num_samples_gaze, 3))
            orientation = np.empty((num_samples_gaze, 3, 3))
            pose = np.zeros((num_samples_gaze, 4, 4))
            idx_start = i * num_samples_gaze
            idx_end = (i + 1) * num_samples_gaze
            theta_slice = theta_samples[idx_start:idx_end]
            phi_slice = phi_samples[idx_start:idx_end]
            distance_slice = distance_samples[idx_start:idx_end]
            
            position[:, 0:1] = np.cos(theta_slice) * np.sin(phi_slice) * distance_slice
            position[:, 1:2] = np.sin(theta_slice) * np.sin(phi_slice) * distance_slice
            position[:, 2:3] = np.cos(phi_slice) * distance_slice
            position += position_samples[i]
            position_gaze_samples[i, j] = position

            z_axis = np.empty((num_samples_gaze, 3))
            z_axis[:, 0:1] = -np.cos(theta_slice) * np.sin(phi_slice)
            z_axis[:, 1:2] = -np.sin(theta_slice) * np.sin(phi_slice)
            z_axis[:, 2:3] = -np.cos(phi_slice)

            x_axis = np.empty((num_samples_gaze, 3))
            x_axis[:, 0] = z_axis[:, 1]
            x_axis[:, 1] = -z_axis[:, 0]
            x_axis[:, 2] = 0

            rotations = rotation_matrix(z_axis, angle)
            for l in range(num_samples_gaze):
                x_axis[l] = x_axis[l] / np.linalg.norm(x_axis[l])
                x_axis[l] = rotations[l] @ x_axis[l]
                
            y_axis = np.cross(z_axis, x_axis)
            for l in range(num_samples_gaze):
                orientation[l, :, 0] = x_axis[l]
                orientation[l, :, 1] = y_axis[l]
                orientation[l, :, 2] = z_axis[l]
                
            orientation_gaze_samples[i, j] = orientation
            pose[:,:3,:3] = orientation
            pose[:,:3,3] = position
            pose[:,3,3] = 1
            pose_gaze_samples[i, j] = pose

    return pose_gaze_samples
@numba.njit('Tuple((float64[:,:], float64[:,:]))(float64[:,:,:])')
def get_normals_polytopes(faces):
    # Calculate the normal vector of the face plane
    normals = np.empty((faces.shape[0],3),dtype=np.float64)
    b_s = np.empty((faces.shape[0],1),dtype=np.float64)
    for i,face_vertices in enumerate(faces):
        v1 = face_vertices[1,:] - face_vertices[0,:]
        v2 = face_vertices[2,:] - face_vertices[0,:]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        normals[i] = normal
        b_s[i,0] = np.dot(normal, face_vertices[0,:])
    return normals, b_s,

@numba.njit('float64[:,:](float64[:,:], float64[:,:,:], float64[:],float64[:])')
def get_intersection_points_planes(normals,faces, line_point, line_direction):
    
    points = np.empty((normals.shape[0],3),dtype=np.float64)
    for i,(face_vertices,normal) in enumerate(zip(faces,normals)):
        
        dot_normal_direction = np.dot(line_direction, normal)
        if dot_normal_direction == 0:
            points[i] = np.inf*np.ones((3,))
            continue
        t = np.dot(face_vertices[0,:] - line_point, normal) / dot_normal_direction
        intersection_point = line_point + (t) * line_direction
        points[i] = intersection_point
    return points
@numba.njit('boolean[:](float64[:,:],float64[:,:,:],float64[:,:],float64[:,:])')
def line_intersects_with_polytope_batch(polytope_vertices,faces, line_points, line_directions):
    
    
    normals, b = get_normals_polytopes(faces)
    middle_point = np.zeros((1,3),dtype=np.float64)
    for i in range(polytope_vertices.shape[0]):
        middle_point += polytope_vertices[i]
    middle_point /= polytope_vertices.shape[0]
    signs = normals@middle_point.T >= b
    intersects = np.array([False]*line_points.shape[0],)
    for i in range(line_points.shape[0]):
        points = get_intersection_points_planes(normals,faces, line_points[i], line_directions[i])
        
        temp = np.logical_or(np.abs(normals@points.T-b) <= 1e-6,(~np.logical_xor(normals@points.T >= b,signs)))
        all_ = np.array([True]*temp.shape[1],)
        for j in range(temp.shape[1]):
            for k in range(temp.shape[0]):
                # print(temp[k,j])
                all_[j] = all_[j] & temp[k,j]
                
        any_ = False
        for j in range(temp.shape[1]):
            any_ = any_ | all_[j]
        intersects[i] = any_
    return intersects

def get_samples_around_placing_spot(sampling_size_around_spot,num_samples_spot,num_samples_gaze,num_samples_gaze_turn,placing_spot_pose,q7_guesses,elevation,robot_base_pose_1):
    rng = np.random.default_rng(0)
    gaze_z_angles = np.linspace(0,2*np.pi,num_samples_gaze_turn,endpoint=False)
    position_samples = rng.normal(placing_spot_pose.translation()[:3],[sampling_size_around_spot,sampling_size_around_spot,sampling_size_around_spot],(num_samples_spot, 3))
    position_samples[:,2] = 0.0
    # elevation = [0.35,0.45]
    max_cone_angle = np.pi/4
    distance_samples = rng.uniform(elevation[0],elevation[1],(num_samples_spot*num_samples_gaze, 1))
    phi_samples = rng.uniform(-max_cone_angle,max_cone_angle,(num_samples_spot*num_samples_gaze, 1))
    theta_samples = rng.uniform(0,2*np.pi-0.1,(num_samples_spot*num_samples_gaze, 1))
    random_config = np.zeros((7,1))
    # q7_guesses = np.linspace(robots[0].plant.GetPositionLowerLimits()[6],robots[0].plant.GetPositionUpperLimits()[6],num_q7s)
    num_q7s = q7_guesses.size
    pose_gaze_samples = calculate_samples(position_samples,distance_samples,phi_samples,theta_samples,gaze_z_angles)
    pose_gaze_samples = pose_gaze_samples.reshape(-1,4,4)
    iks_robot_1 = batch_franka_IK_EE_(robot_base_pose_1,np.repeat(pose_gaze_samples,num_q7s,axis=0),np.broadcast_to(q7_guesses,(pose_gaze_samples.shape[0],q7_guesses.size)).reshape(-1),np.broadcast_to(random_config,(pose_gaze_samples.shape[0],7,1)))
    iks_robot_1 = iks_robot_1.reshape(-1, num_q7s, 4, 7)
    ik_mask = ~np.isnan(iks_robot_1).any(axis=-1,)
    transform_masks = np.any(np.any((ik_mask).reshape(pose_gaze_samples.shape[0],num_q7s,-1,1),axis = 1),axis=1)
    # which transforms have IK solutions
    return pose_gaze_samples,iks_robot_1,transform_masks, ik_mask