from pydrake.all import LeafSystem, Value,RigidTransform,ImageDepth32F, RotationMatrix, Rgba, Box
from drake import lcmt_robot_state, lcmt_point_cloud, lcmt_image, lcmt_image_array
from diff_co_mpc.diff_co_lcm import lcmt_gaze_polytopes
import numpy as np
import torch
from utils.my_casadi.misc import Compile
import open3d as o3d
import open3d.core as o3c
from .point_cloud_processing import point_cloud_worker, Message,WorkData,ResultData,TestData,MessageType, depth_image_to_point_cloud
from torch import multiprocessing as mp
from diff_co_mpc.misc.helper_functions import torch_to_drake_point_cloud
import time
import numpy as np
from multiprocessing import shared_memory
def are_points_in_cone(cone_position, cone_direction, points, cone_fov):
    cos_theta = torch.cos(cone_fov)
    v = points - cone_position
    v_norm = v / torch.norm(v, dim=1, keepdim=True)
    dot_product = torch.matmul(v_norm, cone_direction)
    return dot_product >= cos_theta
class PreprocessPointCloud(LeafSystem):
    def __init__(self, margin, robots,device,voxel_size, camera_1_intrinsics,camera_2_intrinsics, meshcat, CODEGEN_FOLDER):
        super().__init__()
        self.meshcat = meshcat
        self.CODEGEN_FOLDER = CODEGEN_FOLDER
        self.voxel_size = voxel_size
        self.margin = margin
        self.robots = robots
        self.camera_1_intrinsics = camera_1_intrinsics
        self.camera_2_intrinsics = camera_2_intrinsics

        # self.DeclareForcedPublishEvent(self.publish)
        self.robot_1_position_port = self.DeclareAbstractInputPort(name="robot_1_position",
                                      model_value=Value(lcmt_robot_state()))
        self.robot_2_position_port = self.DeclareAbstractInputPort(name="robot_2_position",
                                      model_value=Value(lcmt_robot_state()))
        self.camera_pose_1_input_port = self.DeclareAbstractInputPort(name="camera_pose_1",
                                      model_value=Value(RigidTransform()))
        self.camera_pose_2_input_port = self.DeclareAbstractInputPort(name="camera_pose_2",
                                      model_value=Value(RigidTransform()))
        self.final_pose_input_port = self.DeclareAbstractInputPort(name="final_pose",
                                      model_value=Value(RigidTransform()))
        
        # self.point_cloud_1_input_port = self.DeclareAbstractInputPort(name="in_point_cloud_1",
        #                               model_value=Value(PointCloud()))
        # self.point_cloud_2_input_port = self.DeclareAbstractInputPort(name="in_point_cloud_2",
        #                               model_value=Value(PointCloud()))
        self.depth_image_1_input_port = self.DeclareAbstractInputPort(name="in_depth_image_1",
                                      model_value=Value(ImageDepth32F()))
        self.depth_image_2_input_port = self.DeclareAbstractInputPort(name="in_depth_image_2",
                                      model_value=Value(ImageDepth32F()))
        
        self.lcmt_gaze_polytopes_output_port = self.DeclareAbstractOutputPort(
            name="support_vectors",
            alloc=lambda: Value(lcmt_gaze_polytopes()),
            calc=self.calc_gaze_polytopes)
        self.point_cloud_output_port = self.DeclareAbstractOutputPort(
            name="out",
            alloc=lambda: Value(torch.empty(1,)),
            calc=self.calc_point_cloud)
        self.device = device
        # robots_namedtuple = namedtuple
        # for robot in robots:
        #     _col_sampler = type('robot', (_col_sampler,), {'robot_point_radii_values_pytorch':robot.collision_sampler.robot_point_radii_values_pytorch,
        #                                                    'robot_point_positions_func_pytorch':robot.collision_sampler.robot_point_positions_func_pytorch})()
        # obj = type('robot', (object,), {'collision_sampler':{}})()
        self.robot_1_point_radii_values = self.robots[0].collision_model.all_geometry_radii_torch.to(self.device,dtype=torch.float32).reshape(-1,1)
        self.robot_2_point_radii_values = self.robots[1].collision_model.all_geometry_radii_torch.to(self.device,dtype=torch.float32).reshape(-1,1)
        self.robot_geometry_radii = torch.cat([self.robot_1_point_radii_values,self.robot_2_point_radii_values],dim=0)
        self.robot_1_geometry_centers_func = (self.robots[0].collision_model.all_geometries_torch)
        self.robot_2_geometry_centers_func = (self.robots[1].collision_model.all_geometries_torch)
        # self.distance_from_point_cloud_to_robot_points = torch.compile(self.distance_from_point_cloud_to_robot_points)
        (self.CODEGEN_FOLDER / 'misc_functions').mkdir(exist_ok=True)

        self.robot_geometry_centers_function_filename_1 = 'robot_collision_geometry_positions_1'
        self.robot_geometry_centers_function_filename_2 = 'robot_collision_geometry_positions_2'
        Compile(file_name = self.robot_geometry_centers_function_filename_1,path = self.CODEGEN_FOLDER / 'misc_functions',function = self.robots[0].collision_model.all_geometry_casadi)
        Compile(file_name = self.robot_geometry_centers_function_filename_2,path = self.CODEGEN_FOLDER / 'misc_functions',function = self.robots[1].collision_model.all_geometry_casadi)
        # robot_geometry_centers_func_1 = ca_utils.Compile(file_name = configurations['robot_geometry_centers_function_filename'],path = configurations['codegen_path'],function_name = configurations['robot_geometry_centers_function_name'],get_cached_or_throw = True)
        self.last_point_cloud = o3d.t.geometry.PointCloud(device=self.device_o3d)

        self.num_workers = 1
        self.comms_to_worker = [mp.Queue() for i in range(self.num_workers)]
        self.comms_from_worker = [mp.Queue() for i in range(self.num_workers)]
        point_cloud_max_size = 10000
        self.work_tensors = [{
                            'point_cloud_1_buffer': torch.zeros((point_cloud_max_size,3), dtype=torch.float32,device=device).share_memory_(),
                            'point_cloud_2_buffer': torch.zeros((point_cloud_max_size,3), dtype=torch.float32,device=device).share_memory_(),
                            'point_cloud_result_buffer': torch.zeros((point_cloud_max_size,3), dtype=torch.float32,device=device).share_memory_(),
                             'last_point_cloud_buffer': torch.zeros((point_cloud_max_size,3), dtype=torch.float32,device=device).share_memory_(),
                             'robot_1_positions_buffer': torch.zeros((9), dtype=torch.float32,device=device).share_memory_(),
                             'robot_2_positions_buffer': torch.zeros((9), dtype=torch.float32,device=device).share_memory_(),
                             'world_to_camera_1_buffer': torch.zeros((4,4), dtype=torch.float32,device=device).share_memory_(),
                             'world_to_camera_2_buffer': torch.zeros((4,4), dtype=torch.float32,device=device).share_memory_(),
                             'robot_geometry_radii_buffer': self.robot_geometry_radii.clone().share_memory_(),
                             'placing_spot_position_buffer': torch.zeros((1,3), dtype=torch.float32,device=device).share_memory_(),
                             'gaze_polytopes_buffer': torch.zeros((6000,3), dtype=torch.float32,device=device).share_memory_(),
                             'gaze_simplices_buffer': torch.zeros((6000,3), dtype=torch.int32,device=device).share_memory_(),
                            'gaze_indices_buffer': torch.zeros((500,4), dtype=torch.int32,device=device).share_memory_(),
                             }
                             for i in range(self.num_workers)]
    # gaze_polytopes_buffer = work_tensors['gaze_polytopes_buffer']
    # gaze_simplices_buffer = work_tensors['gaze_simplices_buffer']
    # gaze_indices_buffer = work_tensors['gaze_indices_buffer']
        worker_timeout = 20
        configurations = {
                        'robot_geometry_centers_function_filename_1':'robot_collision_geometry_positions_1',
                        'robot_geometry_centers_function_filename_2':'robot_collision_geometry_positions_2',
                          'robot_geometry_centers_function_name':self.robots[0].collision_model.all_geometry_casadi.name(),
                          'codegen_path':self.CODEGEN_FOLDER / 'misc_functions',
                          'camera_fov_1': 0.6,
                          'camera_fov_2': 0.6,
                          }

                        #   configurations['camera_fov_1'],configurations['camera_fov_2']
        self.message = lcmt_gaze_polytopes()
        self.workers = [mp.Process(target=point_cloud_worker, args=(self.work_tensors[i],self.comms_to_worker[i],self.comms_from_worker[i],configurations,worker_timeout,self.device)) for i in range(self.num_workers)]
        for worker in self.workers:
            worker.start()
        for i in range(self.num_workers):
            self.comms_to_worker[i].put(Message(MessageType.TEST,data=TestData(message=f'test_worker_{i}')))
        self.last_point_cloud = torch.zeros((0,3),dtype=torch.float32,device=self.device)
        self.processing = False
        self.placing_spot_hidden = False
        self.num_polytopes = 0
        self.meshcat.SetObject('/placing_spot',Box(0.1,0.05,0.05),)
        self.count = 0
        try:
            self.point_cloud_shared_memory = shared_memory.SharedMemory(name='point_cloud_shared_memory',create=True, size=point_cloud_max_size*3*np.dtype(np.float32).itemsize)
            
            self.shared_point_cloud = np.ndarray((point_cloud_max_size,3), dtype=np.float32, buffer=self.point_cloud_shared_memory.buf)
            self.shared_point_cloud[:] = 0.
            print("created shared memory")
        except:
            self.point_cloud_shared_memory = shared_memory.SharedMemory(name='point_cloud_shared_memory', size=point_cloud_max_size*3*np.dtype(np.float32).itemsize)
            
            self.shared_point_cloud = np.ndarray((point_cloud_max_size,3), dtype=np.float32, buffer=self.point_cloud_shared_memory.buf)
            self.shared_point_cloud[:] = 0.
            print("using pre existing shared memory")
    @property
    def device_o3d(self):
        return o3c.Device('cuda:0') if self.device == 'cuda:0' else o3c.Device('cpu:0')
    def calc_gaze_polytopes(self,context,output):
        
        output.set_value(self.message)
    def calc_point_cloud(self, context, output):
        # TODO: filter the point cloud based on the robot positions
        # CHATUBA
        t0 = time.perf_counter()
        voxel_size = self.voxel_size
        lower_camera = torch.tensor([-0.6,-0.6,0.05])
        upper_camera = torch.tensor([0.6,1,1.2])
        # lower_camera = torch.tensor([-0.6,-0.40,0.22]) #scenario 3
        # upper_camera = torch.tensor([0.6,-0.2,1.2])
        camera_1_to_world = self.camera_pose_1_input_port.Eval(context).GetAsMatrix4()
        camera_2_to_world = self.camera_pose_2_input_port.Eval(context).GetAsMatrix4()
        world_to_camera_1 = np.linalg.inv(camera_1_to_world)
        world_to_camera_2 = np.linalg.inv(camera_2_to_world)
        # original_point_cloud_1:PointCloud = self.point_cloud_1_input_port.Eval(context).Crop([-1,-1,0.01],[1,1,1.2])
        # original_point_cloud_2:PointCloud = self.point_cloud_2_input_port.Eval(context).Crop([-1,-1,0.01],[1,1,1.2])
        depth_1 = self.depth_image_1_input_port.Eval(context)
        depth_2 = self.depth_image_2_input_port.Eval(context)
        # fx,fy,cx,cy = 
        original_point_cloud_1 = depth_image_to_point_cloud(torch.as_tensor(depth_1.data).squeeze().cpu(),*self.camera_1_intrinsics)
        # fx,fy,cx,cy = 
        original_point_cloud_2 = depth_image_to_point_cloud(torch.as_tensor(depth_2.data).squeeze().cpu(),*self.camera_2_intrinsics)
        fov = 0.6
        crop_cone_mask_1 =  are_points_in_cone(torch.tensor([0,0,0],dtype=torch.float32),torch.tensor([0,0,1],dtype=torch.float32),original_point_cloud_1,torch.tensor(fov,dtype=torch.float32,))
        crop_cone_mask_2 =  are_points_in_cone(torch.tensor([0,0,0],dtype=torch.float32),torch.tensor([0,0,1],dtype=torch.float32),original_point_cloud_2,torch.tensor(fov,dtype=torch.float32,))
        original_point_cloud_1  = original_point_cloud_1[crop_cone_mask_1]
        original_point_cloud_2  = original_point_cloud_2[crop_cone_mask_2]
        # pc = depth_image_to_point_cloud(torch.as_tensor(realsense_reader.depth_image_1_output_port.Eval(rs_context).data).squeeze().cpu(),fx,fy,cx,cy)
        # R = torch.as_tensor(realsense_reader.transform_output_port.Eval(rs_context).rotation().matrix(),dtype=torch.float32)
        # t = torch.as_tensor(realsense_reader.transform_output_port.Eval(rs_context).translation(),dtype=torch.float32).reshape(3,1)
        # pc_cam = R@pc.unsqueeze(-1) + t
        # cropped = pc_cam[torch.all((pc_cam > l.reshape(3,1)) & (pc_cam < u.reshape(3,1)),dim=1).squeeze()]
        R = torch.as_tensor(camera_1_to_world[:3,:3],dtype=torch.float32)
        t = torch.as_tensor(camera_1_to_world[:3,3],dtype=torch.float32).reshape(3,1)
        original_point_cloud_1 = torch.as_tensor(RotationMatrix.MakeXRotation(-0.1*0).matrix(),dtype=torch.float32)@(R@original_point_cloud_1.unsqueeze(-1) + t)
        original_point_cloud_1 = original_point_cloud_1[torch.all((original_point_cloud_1 > lower_camera.reshape(3,1)) & (original_point_cloud_1 < upper_camera.reshape(3,1)),dim=1).squeeze()]
        
        R = torch.as_tensor(camera_2_to_world[:3,:3],dtype=torch.float32)
        t = torch.as_tensor(camera_2_to_world[:3,3],dtype=torch.float32).reshape(3,1)
        original_point_cloud_2 = torch.as_tensor(RotationMatrix.MakeYRotation(0.0).matrix(),dtype=torch.float32)@(R@original_point_cloud_2.unsqueeze(-1) + t)
        # original_point_cloud_2[:,2] += 0.04
        original_point_cloud_2 = original_point_cloud_2[torch.all((original_point_cloud_2 > lower_camera.reshape(3,1)) & (original_point_cloud_2 < upper_camera.reshape(3,1)),dim=1).squeeze()]

        original_point_cloud_1 = torch_to_drake_point_cloud(original_point_cloud_1)
        original_point_cloud_2 = torch_to_drake_point_cloud(original_point_cloud_2)

        robot_1_position_lcm = self.robot_1_position_port.Eval(context)
        robot_2_position_lcm = self.robot_2_position_port.Eval(context)
        output_point_cloud = self.last_point_cloud
        placing_spot_pose = self.final_pose_input_port.Eval(context)
        
        try:
            
            message = self.comms_from_worker[0].get_nowait()
            processed_point_cloud_size = message.data.processed_point_cloud_size
            processed_point_cloud = self.work_tensors[0]['point_cloud_result_buffer'][:processed_point_cloud_size].clone()
            gaze_indices_size = message.data.gaze_indices_size
            self.num_polytopes = gaze_indices_size

            self.placing_spot_hidden = message.data.placing_spot_hidden
            if self.num_polytopes > 0:
                message = lcmt_gaze_polytopes()
                gaze_indices = self.work_tensors[0]['gaze_indices_buffer'][:self.num_polytopes].cpu().numpy().copy()
                a,b,c,d = gaze_indices[-1]
                gaze_polytopes = self.work_tensors[0]['gaze_polytopes_buffer'][:b].cpu().numpy().copy()
                gaze_simplices = self.work_tensors[0]['gaze_simplices_buffer'][:d].cpu().numpy().copy()
                message.num_polytopes = self.num_polytopes
                message.num_vertices = b
                message.num_simplices = d
                # print(gaze_indices[-1])
                message.indices = gaze_indices
                message.vertices = gaze_polytopes
                message.simplices = gaze_simplices
                message.is_placing_spot_hidden = self.placing_spot_hidden
                self.message = message
#                 msg = self.message
#                 vertices = np.asarray(msg.vertices)
#                 simplices = np.asarray(msg.simplices)
#                 indices = np.asarray(msg.indices)
#                 print(indices)
#                 polytopes = []
#                 for i,(a,b,c,d) in enumerate(indices):
#                     polytope_vertices= vertices[a:b]

#                     polytope_simplices = simplices[c:d]
#                     # self.meshcat.SetTriangleMesh(f'/polytope_{i}',polytope_vertices.T,polytope_simplices.T)
# # 
#                     polytopes.append( { 'vertices': vertices[a:b], 'simplices': simplices[c:d] } )
#                     mean_vertices = np.mean(polytope_vertices,axis=0)
#                     # if np.linalg.norm(mean_vertices-placing_spot_pose.translation()) > 0.5:
#                         # continue
#                     polytope_vertices_ = polytope_vertices.astype(np.float64)
#                     faces = polytope_vertices[polytope_simplices]
#                     faces_ = np.array(faces).astype(np.float64)
            self.last_point_cloud = processed_point_cloud
            self.processing = False
            output_point_cloud = processed_point_cloud


            visualize_polytopes = True
            if visualize_polytopes:
                for i in range(self.num_polytopes):
                    self.meshcat.Delete(f'/polytope_{i}')
                for i in range(gaze_indices_size):                
                    a,b,c,d = self.work_tensors[0]['gaze_indices_buffer'][i]
                    vertices = self.work_tensors[0]['gaze_polytopes_buffer'][a:b]
                    simplices = self.work_tensors[0]['gaze_simplices_buffer'][c:d]
                    # faces = vertices[simplices]
                    self.meshcat.SetTriangleMesh(f'/polytope_{i}',vertices.T,simplices.T)
                    self.meshcat.SetProperty(f'/polytope_{i}',"color",[0,0,0,0.5])
                
            
            # self.meshcat.SetTriangleMesh('polytope',polytope_vertices.T,simplices.T)
            # print("Updated processed point cloud")
            # print("update proc pc", time.perf_counter() - t0)
        except:
            pass
        t0 = time.perf_counter()
        if not self.processing and self.count > 40:
            if not (original_point_cloud_1.size() == 0 or len(robot_1_position_lcm.joint_position) == 0 or len(robot_2_position_lcm.joint_position) == 0):
                point_cloud_1 = original_point_cloud_1.VoxelizedDownSample(voxel_size=voxel_size)
                point_cloud_2 = original_point_cloud_2.VoxelizedDownSample(voxel_size=voxel_size)
                # self.meshcat.SetObject('/original_point_cloud_1_ds',point_cloud_1,voxel_size,Rgba(0,0,1,0.5))
                # world_to_camera_1 = torch.from_numpy(np.linalg.inv(camera_1_to_world)).to(self.device)
                # world_to_camera_2 = torch.from_numpy(np.linalg.inv(camera_2_to_world)).to(self.device)
                robot_1_positions = torch.tensor(robot_1_position_lcm.joint_position,dtype=torch.float32,device=self.device)
                robot_2_positions = torch.tensor(robot_2_position_lcm.joint_position,dtype=torch.float32,device=self.device)
                point_cloud_1_torch = torch.from_numpy(point_cloud_1.xyzs()).to(self.device).T
                point_cloud_2_torch = torch.from_numpy(point_cloud_2.xyzs()).to(self.device).T
                

                self.work_tensors[0]['point_cloud_1_buffer'][:point_cloud_1_torch.shape[0]] = point_cloud_1_torch
                self.work_tensors[0]['point_cloud_2_buffer'][:point_cloud_2_torch.shape[0]] = point_cloud_2_torch
                self.work_tensors[0]['robot_1_positions_buffer'][:] = robot_1_positions
                self.work_tensors[0]['robot_2_positions_buffer'][:] = robot_2_positions
                self.work_tensors[0]['world_to_camera_1_buffer'][:] = torch.from_numpy(world_to_camera_1).to(self.device)    
                self.work_tensors[0]['world_to_camera_2_buffer'][:] = torch.from_numpy(world_to_camera_2).to(self.device)       
                self.work_tensors[0]['placing_spot_position_buffer'][:] = torch.from_numpy(placing_spot_pose.translation()).to(self.device)
                        
                self.comms_to_worker[0].put(Message(MessageType.WORK,data=WorkData(point_cloud_1_size = point_cloud_1_torch.shape[0],
                                                                                               point_cloud_2_size = point_cloud_2_torch.shape[0],
                                                                                               voxel_size=self.voxel_size,
                                                                                               margin= self.margin)))

                self.processing = True
                # print("sent pc for processing" ,time.perf_counter() - t0)
        # if output_point_cloud
        self.output_point_cloud = output_point_cloud
        # self.meshcat.SetObject('/processed_point_cloud',torch_to_drake_point_cloud(output_point_cloud),0.02,Rgba(1,0,0,0.5))
        try:
            # d = torch.cdist(output_point_cloud,output_point_cloud).fill_diagonal_(1000)
            d = torch.cdist(output_point_cloud,output_point_cloud).squeeze()
            # d[86]
            d.fill_diagonal_(self.voxel_size+0.1)
            pc2 = output_point_cloud[~torch.any(torch.triu(d < 0.01),dim = 1)]
            d = torch.cdist(pc2,pc2).squeeze()
            d.fill_diagonal_(self.voxel_size+0.1)
            output_point_cloud = pc2[~torch.all(d > self.voxel_size,dim = 1)]
            self.meshcat.SetObject('/processed_point_cloud_filtered',torch_to_drake_point_cloud(output_point_cloud),0.02,Rgba(0,1,0,0.5))
        except:
            pass
        # self.meshcat.SetObject("/aaaaa",torch_to_drake_point_cloud(pc2[~torch.all(d > point_cloud_filter.voxel_size,dim = 1)]),0.03,Rgba(0,0,0,1))
        
        output.set_value(output_point_cloud)
        self.meshcat.SetObject('/original_point_cloud_1',original_point_cloud_1,0.01,Rgba(1,1,1,0.5))
        self.meshcat.SetObject('/original_point_cloud_2',original_point_cloud_2,0.01,Rgba(1,0.8,0.8,0.5))
        

        
        
        
        if self.placing_spot_hidden:
            self.meshcat.SetProperty("/placing_spot","color",[1,0,0,1])
            
        else:
            self.meshcat.SetProperty("/placing_spot","color",[0,1,0,1])
        self.meshcat.SetTransform(f"/placing_spot", placing_spot_pose)
        self.count += 1
        # self.meshcat.SetObject('/last_point_cloud',torch_to_drake_point_cloud(last_point_cloud_torch),0.015,Rgba(0,1,0,0.5))
        self.shared_point_cloud[:] = np.nan
        self.shared_point_cloud[:output_point_cloud.shape[0]] = output_point_cloud.cpu().numpy()
