import torch
import queue
import multiprocessing
import numpy
import open3d as o3d
import open3d.core as o3c
import numpy as np
import einops
import torch.multiprocessing as mp
from queue import Empty
from enum import Enum
from concave_hull import convex_hull, convex_hull_indexes
import utils.my_casadi.misc as ca_utils
class MessageType:
    STOP = 0
    WORK = 1
    RESULT = 2
    ERROR = 3
    TEST = 4
from dataclasses import dataclass, field
import typing as T
@dataclass
class WorkData:
    point_cloud_1_size: int
    point_cloud_2_size: int
    voxel_size: float
    margin: float
@dataclass
class ResultData:
    processed_point_cloud_size: int
    point_cloud_without_robots_size: int
    last_point_cloud_size: int
    placing_spot_hidden: bool
    gaze_indices_size: int = 0
                # final point cloud = filtered_point_cloud_torch
                # point_cloud_without_robots = point_cloud_without_robots_torch
@dataclass
class ErrorData:
    error_message: str
@dataclass
class TestData:
    message: str
@dataclass
class Message:
    type: MessageType
    data: T.Optional[T.Union[WorkData, ResultData, ErrorData,TestData]] = field(default=None)
@torch.compile(dynamic = True)
def project_points_onto_plane_torch(points, d):
    # Convert points to a PyTorch tensor if they are not already
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    theta = torch.atan2(points[...,0],points[...,2])
    xl = torch.tan(theta)*d
    yl = points[...,1]*d/(points[...,2] * torch.cos(theta)+1e-6)
    new_points = torch.stack([xl, yl, torch.full_like(yl, d)], dim=-1)


    return new_points
def depth_image_to_point_cloud(depth_image, fx, fy, cx, cy):
    """
    Transforms a depth image into a point cloud.
    
    Args:
        depth_image (torch.Tensor): A 2D tensor with shape (H, W) containing depth values.
        fx (float): Focal length along the x-axis.
        fy (float): Focal length along the y-axis.
        cx (float): Principal point along the x-axis.
        cy (float): Principal point along the y-axis.
        
    Returns:
        torch.Tensor: A tensor of shape (H*W, 3) containing the point cloud coordinates.
    """
    assert depth_image.dim() == 2, "Depth image must be a 2D tensor"

    # Get image dimensions
    height, width = depth_image.shape

    # Generate pixel coordinates
    i, j = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    i = i.to(depth_image.device).float()
    j = j.to(depth_image.device).float()

    # Unproject depth image to 3D points
    z = depth_image
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy

    # Stack to create a (H*W, 3) point cloud
    point_cloud = torch.stack((x, y, z), dim=2).reshape(-1, 3)

    return point_cloud
# what to consider when considering occluding objects
selection = """panda_link_0
/ 14
panda_link_1
0,2,4 / 6
panda_link_2
2,4,5 / 6 
panda_link_3
1 / 5
panda_link_4
0,3,4,5 / 7
panda_link_5
0,2,4,6,10 / 12
panda_link_6
0,3 / 5 
panda_link_7
0 / 2
panda_link_8
0 / 1
panda_hand
1,2,3 / 4
panda_link_0
/ 14
panda_link_1
0,2,4 / 6
panda_link_2
2,4,5 / 6 
panda_link_3
1 / 5
panda_link_4
0,3,4,5 / 7
panda_link_5
0,2,4,6,10 / 12
panda_link_6
0,3 / 5 
panda_link_7
0 / 2
panda_link_8
0 / 1
panda_hand
1,2,3 / 4
"""
def parse_selection(selection):
    lines = selection.split('\n')[1::2]
    count = 0
    indices_= []
    for line in lines:
        parts = line.split('/')
        indices = [int(part.strip())+count for part in parts[0].split(',') if part.strip() != '']
        indices_+=(indices)
        count += int(parts[1].strip())
    return indices_
INDICES_ = parse_selection(selection)


import traceback
def calculate_occlusion_polytopes(point_cloud_torch,point_cloud_without_robots_o3d,where_close_to_points_in_robots,camera_pose_inv):
    clusters = []
    # indices = INDICES_
    # for index in [indices[i:i+4] for i in range(0,len(indices),4)]:
    #     points = point_cloud_torch[torch.any(where_close_to_points_in_robots[index],dim=0)]
    #     if points.shape[0] < 4:
    #         continue
    #     clusters.append(o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(points))))
    if not point_cloud_without_robots_o3d.is_empty():
        labels = point_cloud_without_robots_o3d.cluster_dbscan(eps=0.1, min_points=5, print_progress=False)
        max_label = labels.max().item()
        for i in range(max_label+1):
            clusters.append(point_cloud_without_robots_o3d.select_by_mask((labels==i)))
    world_to_camera = camera_pose_inv
    hulls = []
    polytopes = []        
    for cluster in clusters:
        try:
            hull = cluster.compute_convex_hull()
            hulls.append(hull)
            # .triangles
            # mesh.triangle.indices
            # polytope_vertices_np = hull.vertex.positions.numpy()
            simplices = torch.utils.dlpack.from_dlpack(hull.triangle.indices.to_dlpack()).to(dtype=torch.float32,device=point_cloud_torch.device)
            
            # faces_np = polytope_vertices_np[simplices]
            vertices = torch.utils.dlpack.from_dlpack(hull.vertex.positions.to_dlpack()).to(dtype=torch.float32,device=point_cloud_torch.device)
            vertices_in_camera = (world_to_camera[:3,:3]@vertices.unsqueeze(dim=-1)).squeeze() + world_to_camera[:3,3]
            projected_vertices = project_points_onto_plane_torch(vertices_in_camera,3)
            
            vertices_indices = convex_hull_indexes(projected_vertices.cpu().numpy()[:,:2])
            vectors = [] 
            mean = torch.mean(vertices_in_camera[vertices_indices],dim=0)
            vertices_in_camera[:,:2] += (vertices_in_camera[:,:2] - mean[:2]) * 0.1

            for i,(v, p) in enumerate(zip(vertices_in_camera[vertices_indices],projected_vertices[vertices_indices])):
                
                vectors.append(p-v)
            vectors.append(vectors[0])
            vectors = torch.stack(vectors,)
            normals = torch.cross(vectors[:-1],vectors[1:])
            closest_point_to_camera = vertices_in_camera[torch.argmin(vertices_in_camera[:,2]),:]
            polytope_points = einops.einsum(normals,projected_vertices[vertices_indices],'i j,i j->i').view(-1,1)
            polytope_points = torch.concat((polytope_points,torch.dot(closest_point_to_camera,closest_point_to_camera).reshape(1,-1)))

            normals = torch.concat((normals,closest_point_to_camera.reshape(1,3)))
            
            polytopes.append({'A':normals,'b':polytope_points, 'vertices_gaze': vertices,'simplices_gaze': simplices})
        except:
            print('Error in calculate_occlusion_polytopes')
            traceback.print_exc()
    return polytopes

def are_points_in_cone(cone_position, cone_direction, points, cone_fov):
    cos_theta = torch.cos(cone_fov)
    v = points - cone_position
    v_norm = v / torch.norm(v, dim=1, keepdim=True)
    dot_product = torch.matmul(v_norm, cone_direction)
    return dot_product >= cos_theta
import time
def point_cloud_worker(work_tensors, comm_in,comm_out,configurations,watch_dog_time = 20, device = 'cpu'):
    device_o3d = o3c.Device('cpu:0') if device == 'cpu' else o3c.Device(device)
    # margin = work_tensors['margin']
    robot_geometry_radii_buffer = work_tensors['robot_geometry_radii_buffer']
    # voxel_size = work_tensors['voxel_size']
    world_to_camera_1_buffer = work_tensors['world_to_camera_1_buffer']
    world_to_camera_2_buffer = work_tensors['world_to_camera_2_buffer']
    point_cloud_1_buffer = work_tensors['point_cloud_1_buffer']
    point_cloud_2_buffer = work_tensors['point_cloud_2_buffer']
    point_cloud_result_buffer = work_tensors['point_cloud_result_buffer']
    last_point_cloud_buffer = work_tensors['last_point_cloud_buffer']
    last_point_cloud_buffer_size = 0
    robot_1_positions_buffer = work_tensors['robot_1_positions_buffer']
    robot_2_positions_buffer = work_tensors['robot_2_positions_buffer']
    placing_spot_position_buffer = work_tensors['placing_spot_position_buffer']

    gaze_polytopes_buffer = work_tensors['gaze_polytopes_buffer']
    gaze_simplices_buffer = work_tensors['gaze_simplices_buffer']
    gaze_indices_buffer = work_tensors['gaze_indices_buffer']


    camera_fovs = [configurations['camera_fov_1'],configurations['camera_fov_2']]
    robot_geometry_centers_func_1 = ca_utils.Compile(file_name = configurations['robot_geometry_centers_function_filename_1'],path = configurations['codegen_path'],function_name = configurations['robot_geometry_centers_function_name'],get_cached_or_throw = True)
    robot_geometry_centers_func_2 = ca_utils.Compile(file_name = configurations['robot_geometry_centers_function_filename_2'],path = configurations['codegen_path'],function_name = configurations['robot_geometry_centers_function_name'],get_cached_or_throw = True)

    while True:
        try:
            message = comm_in.get(block=True, timeout=watch_dog_time)
            # watchdog = time.perf_counter()

            if message.type == MessageType.STOP:
                print('point_cloud_worker received stop message')
                break
            elif message.type == MessageType.TEST:
                print('Test message:')
                print(message.data.message)
            elif message.type == MessageType.WORK:
                # print('Worker received work message')
                # print('Received at', time.perf_counter())
                t00 = time.perf_counter()
                work_data = message.data
                point_cloud_1_size = work_data.point_cloud_1_size
                point_cloud_2_size = work_data.point_cloud_2_size
                voxel_size = work_data.voxel_size
                margin = work_data.margin
                point_cloud_1_torch = point_cloud_1_buffer[:point_cloud_1_size]
                point_cloud_2_torch = point_cloud_2_buffer[:point_cloud_2_size]

                last_point_cloud = last_point_cloud_buffer[:last_point_cloud_buffer_size].clone()
                world_to_camera_1 = world_to_camera_1_buffer
                world_to_camera_2 = world_to_camera_2_buffer
                robot_1_positions = robot_1_positions_buffer
                robot_2_positions = robot_2_positions_buffer
                robot_geometry_radii = robot_geometry_radii_buffer
                placing_spot_position = placing_spot_position_buffer
                
                # point_cloud_total_torch = torch.cat([point_cloud_1_torch,point_cloud_2_torch],dim=0)
                robot_geometry_centers_1 = torch.from_numpy(robot_geometry_centers_func_1(robot_1_positions.cpu().numpy()).full()).reshape(-1,3).to(dtype=torch.float32,device=device)
                robot_geometry_centers_2 = torch.from_numpy(robot_geometry_centers_func_2(robot_2_positions.cpu().numpy()).full()).reshape(-1,3).to(dtype=torch.float32,device=device)
                robot_geometry_centers = torch.cat([robot_geometry_centers_1,robot_geometry_centers_2],dim=0)    

                polytopes = []
                point_cloud_without_robots_torch_list = [ ]

                count_vertices = 0
                count_simplices = 0
                count_indices = 0
                camera_can_see_placing_spot = []
                for fov,point_cloud_torch, world_to_camera in zip(camera_fovs,[point_cloud_1_torch,point_cloud_2_torch],[world_to_camera_1,world_to_camera_2]):
                    # maybe save distances from previous iteration to filter here as well
                    distances = torch.cdist(robot_geometry_centers,point_cloud_torch)
                    where_close_to_points_in_robots = (distances < (robot_geometry_radii + margin))
                    point_cloud_without_robots_torch = point_cloud_torch[~torch.any(where_close_to_points_in_robots,dim=0)]
                    point_cloud_without_robots_torch_list.append(point_cloud_without_robots_torch)
                    point_cloud_without_robots_o3d = o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(point_cloud_without_robots_torch)))

                    polytopes_camera = calculate_occlusion_polytopes(point_cloud_torch, point_cloud_without_robots_o3d, where_close_to_points_in_robots,world_to_camera)
                    placing_spot_position_in_camera = world_to_camera[:3,:3]@placing_spot_position.view(3,1) + world_to_camera[:3,3].view(3,1)
                    can_I_even_see_the_placing_spot = are_points_in_cone(torch.tensor([0,0,0],dtype=torch.float32,device=device),torch.tensor([0,0,1],dtype=torch.float32,device=device),placing_spot_position_in_camera.T,torch.tensor(fov,dtype=torch.float32,device=device))
                    camera_can_see_placing_spot.append(can_I_even_see_the_placing_spot.view(1,1))
                    print('camera_can_see_placing_spot',camera_can_see_placing_spot[-1])
                    for polytope in polytopes_camera:
                        # placing spot in polytopes?
                        A = polytope['A']
                        b = polytope['b']
                        camera_can_see_placing_spot[-1] &= ~torch.all(A@placing_spot_position_in_camera > b.view(-1,1), axis=0)# & can_I_even_see_the_placing_spot.view(1,1))

                        vertices = polytope['vertices_gaze']
                        simplices = polytope['simplices_gaze']
                        gaze_polytopes_buffer[count_vertices:count_vertices+vertices.shape[0],:] = vertices
                        gaze_simplices_buffer[count_simplices:count_simplices+simplices.shape[0],:] = simplices
                        gaze_indices_buffer[count_indices,0] = count_vertices
                        gaze_indices_buffer[count_indices,1] = count_vertices + vertices.shape[0]
                        gaze_indices_buffer[count_indices,2] = count_simplices
                        gaze_indices_buffer[count_indices,3] = count_simplices + simplices.shape[0]
                        count_vertices += vertices.shape[0]
                        count_simplices += simplices.shape[0]
                        count_indices += 1
                    polytopes.append(polytopes_camera)

                can_see_placing_spot = any(camera_can_see_placing_spot)
                placing_spot_hidden = not can_see_placing_spot
                # Joint point clouds and downsample
                point_cloud_without_robots_total_torch = torch.cat(point_cloud_without_robots_torch_list,dim=0)
                point_cloud_without_robots_total_o3d = o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(point_cloud_without_robots_total_torch)))
                point_cloud_without_robots_total_o3d = point_cloud_without_robots_total_o3d.voxel_down_sample(voxel_size=voxel_size)
                point_cloud_without_robots_total_torch = torch.utils.dlpack.from_dlpack(point_cloud_without_robots_total_o3d.point.positions.to_dlpack())

                if last_point_cloud_buffer_size == 0:

                    last_point_cloud_buffer_size = point_cloud_without_robots_total_torch.shape[0]
                    last_point_cloud_buffer[:last_point_cloud_buffer_size] = point_cloud_without_robots_total_torch.clone()
                    point_cloud_1_buffer[:point_cloud_without_robots_total_torch.shape[0]] = point_cloud_without_robots_total_torch.clone()
                    comm_out.put(Message(MessageType.RESULT, ResultData(processed_point_cloud_size = point_cloud_without_robots_total_torch.shape[0],
                                                                        point_cloud_without_robots_size = point_cloud_without_robots_total_torch.shape[0],
                                                                        last_point_cloud_size= last_point_cloud_buffer_size,
                                                                        placing_spot_hidden = placing_spot_hidden,
                                                                        gaze_indices_size = count_indices
                                                                        )
                                                                        ))
                    continue
                
                indices = []
                # for fov,polytopes_camera, world_to_camera in zip(camera_fovs,polytopes,[world_to_camera_1,world_to_camera_2]):
                    
                #     last_point_cloud_in_camera_torch = ((world_to_camera[:3,:3]@last_point_cloud.unsqueeze(dim=-1)).squeeze() + world_to_camera[:3,3]).view(-1,3)

                    

                #     indices_in_poly = []
                #     indices_in_poly.append(torch.full((last_point_cloud_in_camera_torch.shape[0],),False,dtype=torch.bool,device=device))
                #     for polytope in polytopes_camera:
                #         # re-add occluded points from the last point cloud
                #         A = polytope['A']
                #         b = polytope['b']
                #         indices_in_poly.append(torch.all(A@last_point_cloud_in_camera_torch.T > torch.broadcast_to(b,(A.shape[0],last_point_cloud_in_camera_torch.shape[0])), axis=0))

                #     robot_centers_in_camera = ((world_to_camera[:3,:3]@robot_geometry_centers.unsqueeze(dim=-1)).squeeze() + world_to_camera[:3,3]).view(-1,3)
                #     d_norm = torch.linalg.norm(robot_centers_in_camera, dim=1, keepdim=True)
                #     theta = torch.asin(robot_geometry_radii / d_norm)
                #     cone_direction = robot_centers_in_camera / d_norm.view(-1, 1)

                #     cos_theta = torch.cos(theta)
                #     v = last_point_cloud_in_camera_torch
                #     n = torch.norm(v, dim=1, keepdim=True)
                #     v_norm = v / n
                #     dot_product = torch.vmap(torch.matmul, in_dims=(None, 0))(v_norm, cone_direction)

                #     # Checking which points are inside the cones
                #     inside_cone = (dot_product >= cos_theta) & (n.T >= d_norm)  
                #     indices_in_poly.append(inside_cone.any(dim=0))
                    
                #     can_I_even_see_the_point = are_points_in_cone(torch.tensor([0,0,0],dtype=torch.float32,device=device),torch.tensor([0,0,1],dtype=torch.float32,device=device),last_point_cloud_in_camera_torch,torch.tensor(fov,dtype=torch.float32,device=device))
                #     indices_in_poly.append(~can_I_even_see_the_point) # in the "outside" "polytope"
                #     indices_in_poly = torch.vstack(indices_in_poly).any(dim=0)
                    
                #     indices_in_poly = ~torch.logical_and(~indices_in_poly,can_I_even_see_the_point.view(1,-1))
                #     # each row is a polytope. If any polytope is True, the point is inside the polytope, aka, occluded
                #     indices.append(indices_in_poly)
                indices_num = torch.arange(0,last_point_cloud.shape[0],device=device)
                for fov,polytopes_camera, world_to_camera in zip(camera_fovs,polytopes,[world_to_camera_1,world_to_camera_2]):
                    
                    last_point_cloud_in_camera_torch = ((world_to_camera[:3,:3]@last_point_cloud.unsqueeze(dim=-1)).squeeze() + world_to_camera[:3,3]).view(-1,3)
                    can_I_even_see_the_point = are_points_in_cone(torch.tensor([0,0,0],dtype=torch.float32,device=device),torch.tensor([0,0,1],dtype=torch.float32,device=device),last_point_cloud_in_camera_torch,torch.tensor(fov,dtype=torch.float32,device=device))
                    points_that_I_can_see = last_point_cloud_in_camera_torch[can_I_even_see_the_point]
                    indices_num_that_I_can_see = indices_num[can_I_even_see_the_point]
                    indices_in_poly = []
                    indices_in_poly.append(torch.full((points_that_I_can_see.shape[0],),False,dtype=torch.bool,device=device))
                    for polytope in polytopes_camera:
                        # re-add occluded points from the last point cloud
                        A = polytope['A']
                        b = polytope['b']
                        indices_in_poly.append(torch.all(A@points_that_I_can_see.T > torch.broadcast_to(b,(A.shape[0],points_that_I_can_see.shape[0])), axis=0))

                    robot_centers_in_camera = ((world_to_camera[:3,:3]@robot_geometry_centers.unsqueeze(dim=-1)).squeeze() + world_to_camera[:3,3]).view(-1,3)
                    d_norm = torch.linalg.norm(robot_centers_in_camera, dim=1, keepdim=True)
                    theta = torch.asin(robot_geometry_radii*0.9 / d_norm)
                    cone_direction = robot_centers_in_camera / d_norm.view(-1, 1)

                    cos_theta = torch.cos(theta)
                    v = points_that_I_can_see
                    n = torch.norm(v, dim=1, keepdim=True)
                    v_norm = v / n
                    dot_product = torch.vmap(torch.matmul, in_dims=(None, 0))(v_norm, cone_direction)

                    # Checking which points are inside the cones
                    inside_cone = (dot_product >= cos_theta) & (n.T >= d_norm)  
                    indices_in_poly.append(inside_cone.any(dim=0))
                    
                    
                    # indices_in_poly.append(~can_I_even_see_the_point) # in the "outside" "polytope"
                    indices_in_poly = torch.vstack(indices_in_poly).any(dim=0)
                    
                    # indices_in_poly = ~torch.logical_and(~indices_in_poly,can_I_even_see_the_point.view(1,-1))
                    # each row is a polytope. If any polytope is True, the point is inside the polytope, aka, occluded
                    
                    indices.append(indices_num_that_I_can_see[~indices_in_poly])
                # indices = torch.
                indices_mask = torch.ones(last_point_cloud_in_camera_torch.shape[0],dtype=torch.bool,device=device)
                for idx in indices:
                    indices_mask[idx] = False
                # point must be occluded in both cameras
                # indices = torch.vstack(indices).any(dim=0)
                # indices = torch.vstack(indices).all(dim=0)


                point_cloud_in_poly_torch = last_point_cloud[indices_mask]
                # meshcat.SetObject('/last_point_cloud',torch_to_drake_point_cloud(last_point_cloud_buffer_without_robots),0.01,Rgba(1,0,0,0.5))
                if point_cloud_in_poly_torch.numel() > 0 and point_cloud_in_poly_torch.shape[0]> 1:
                    d = torch.cdist(point_cloud_in_poly_torch,point_cloud_in_poly_torch).squeeze()
                    d.fill_diagonal_(torch.inf)
                    
                    point_cloud_in_poly_torch = point_cloud_in_poly_torch[~torch.any(torch.triu(d < 0.02),dim = 1)]
                    # point_cloud_in_poly_torch = point_cloud_in_poly_torch[:-3]
                    random_indices = torch.randint(0, point_cloud_in_poly_torch.shape[0], (3,))
                    mask = torch.ones(point_cloud_in_poly_torch.shape[0],dtype = torch.bool,device = device)
                    mask[random_indices] = False
                    point_cloud_in_poly_torch = point_cloud_in_poly_torch[mask]
                    if point_cloud_in_poly_torch.numel() > 0:
                        point_cloud_without_robots_total_o3d = point_cloud_without_robots_total_o3d.append(o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(point_cloud_in_poly_torch))))
                    # point_cloud_without_robots_total_o3d = point_cloud_without_robots_total_o3d.voxel_down_sample(voxel_size=voxel_size)


                filtered_point_cloud_torch = torch.utils.dlpack.from_dlpack(point_cloud_without_robots_total_o3d.point.positions.to_dlpack())
                
                point_cloud_result_buffer[:filtered_point_cloud_torch.shape[0]] = filtered_point_cloud_torch.clone()
                last_point_cloud_buffer[:filtered_point_cloud_torch.shape[0]] = filtered_point_cloud_torch.clone()
                last_point_cloud_buffer_size = filtered_point_cloud_torch.shape[0]
                comm_out.put(Message(MessageType.RESULT, ResultData(processed_point_cloud_size = filtered_point_cloud_torch.shape[0],
                                                                    point_cloud_without_robots_size = point_cloud_without_robots_torch.shape[0],
                                                                    last_point_cloud_size= last_point_cloud_buffer_size,
                                                                    placing_spot_hidden = placing_spot_hidden,
                                                                    gaze_indices_size = count_indices
                                                                    )
                                                                    ))
                print("time pc proc",time.perf_counter() - t00)
        except Empty as exc:
            print('point_cloud_worker queue timeout, quitting')
            break


