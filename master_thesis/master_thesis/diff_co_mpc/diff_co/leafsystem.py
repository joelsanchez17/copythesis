
from pydrake.all import LeafSystem, Value, AbstractValue, Rgba
import torch
import numpy as np
from .learning import SupportVectorWorker
from diff_co_mpc.diff_co_lcm import lcmt_support_vector
import time
from diff_co_mpc.helper_functions import torch_to_drake_point_cloud
from queue import Empty

class SVMPipeline(LeafSystem):
    def __init__(self, robots, workers: list[SupportVectorWorker],device, voxel_size, num_samples,prob_threshold,covariance,max_iterations_initialization,max_iterations_update, meshcat):
        super().__init__()
        self.max_iterations_initialization = max_iterations_initialization
        self.max_iterations_update = max_iterations_update
        self.prob_threshold = prob_threshold
        self.covariance = covariance
        self.robots = robots
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.device = device
        self.meshcat = meshcat
        # self.point_cloud_input_port = self.DeclareAbstractInputPort(name="point_cloud_in",
        #                               model_value=Value(PointCloud()))
        self.point_cloud_input_port = self.DeclareAbstractInputPort(name="point_cloud_in",
                                      model_value=Value(torch.empty((1,))))
        # self.DeclareForcedPublishEvent(self.publish)
        self.lcmt_support_vector_output_port = self.DeclareAbstractOutputPort(
            name="support_vectors",
            alloc=lambda: Value(lcmt_support_vector()),
            calc=self.calc_support_vectors)
        
        self.initialized = False
        

        # x = 
        load = np.load(str(TEMP_FOLDER / f'control_points.npz'),)
        self.initial_guess_control_points = [torch.from_numpy(load.get('control_points_1')),torch.from_numpy(load.get('control_points_2'))]
        self.initial_guess_control_points = [torch.concatenate([self.initial_guess_control_points[i], torch.zeros((self.initial_guess_control_points[i].shape[0],2))],axis=1).to(device = 'cuda',dtype=torch.float32) for i in range(2)]
        # self.x_initial_guesses = torch.from_numpy(np.hstack(list(v for v in initial_guess_by_quadrant.values() if v.size > 0)))

        self.workers = workers
        self.num_workers = len(workers)
        for worker in workers:
            worker.obstacle_radii.copy_(voxel_size)
            worker.X_buffer[:] = 0
            worker.W_buffer[:] = 0
            worker.H_s_buffer[:] = 0
            worker.num_support_vectors.copy_(0)
        
        self.processing = [False]*self.num_workers
        self.last_results = [(torch.zeros((0,support_vectors_dimen),dtype=torch.float32,device=device), torch.zeros((0,num_categories),dtype=torch.float32,device=device))]*self.num_workers
        self.message = None
        self.inputs = [None]*self.num_workers
        self.initialized = [False]*self.num_workers

        self.last_fk_support_vectors = [None,None]
        self.last_weights = [None,None]
        self.recording = {"r1":[],"r2":[]}
    def make_message(self,support_vectors_list,weights_list):
        message = lcmt_support_vector()
        message.num_support_vectors_1 = support_vectors_list[0].shape[0]
        message.support_vectors_1 = support_vectors_list[0]
        message.weights_1 = weights_list[0]
        message.num_weights_1 = weights_list[0].shape[0]
        message.num_support_vectors_2 = support_vectors_list[1].shape[0]
        message.support_vectors_2 = support_vectors_list[1]
        message.weights_2 = weights_list[1]
        message.num_weights_2 = weights_list[1].shape[0]
        message.support_vector_dimension = support_vectors_list[0].shape[1]
        message.weights_dimension = 1
        self.message = message
        return message
    def initialize_diff_co(self,point_cloud_torch):
        # return
        point_cloud_torch_gpu = point_cloud_torch.to('cuda')
        # obstacle_radii = self.obstacle_radii.expand(point_cloud_torch.shape[0]).to('cuda')
        t0 = time.perf_counter()
        num_exploration_samples = self.num_samples
        for i,worker in enumerate(self.workers):
            
            

            # lower_limits = torch.tensor(robot.plant.GetPositionLowerLimits(),dtype = torch.float32,device = obstacle_radii.device)
            # upper_limits = torch.tensor(robot.plant.GetPositionUpperLimits(),dtype = torch.float32,device = obstacle_radii.device)
            # sample_distribution = torch.distributions.uniform.Uniform(lower_limits,upper_limits)
            # exploration_samples = sample_distribution.sample((num_exploration_samples,))
            lower_limits = torch.tensor(self.robots[0].plant.GetPositionLowerLimits(),dtype = torch.float32,device = 'cuda')
            upper_limits = torch.tensor(self.robots[0].plant.GetPositionUpperLimits(),dtype = torch.float32,device = 'cuda')
            x_initial_guess = self.initial_guess_control_points[i]
            random_indices = torch.randint(0, x_initial_guess.shape[0], (num_exploration_samples,))
            exploration_samples = x_initial_guess[random_indices]
            exploration_samples += torch.randn_like(exploration_samples)*2
            exploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)
            assert exploration_samples.shape[0] <= worker.X_buffer.shape[0]
            worker.X_buffer[:exploration_samples.shape[0]] = exploration_samples
            worker.point_cloud_buffer[:point_cloud_torch_gpu.shape[0]] = point_cloud_torch_gpu

            worker.comm_in.put(diff_co.Message(diff_co.MessageType.WORK,data=diff_co.WorkData(sample_size = exploration_samples.shape[0],
                                                                                              max_iterations=self.max_iterations_initialization,
                                                                                            point_cloud_size=point_cloud_torch_gpu.shape[0],
                                                                                            covariance=self.covariance,
                                                                                            probability_threshold=self.prob_threshold,
                                                                                            exploitation_size=0,
                                                                                            
                                                                                            )))
        
        # new_support_vectors_list = []
        # new_weights_list = []
        
        
        for i,worker in enumerate(self.workers):
            message = worker.comm_out.get()
            
            result_size = message.data.result_size
            completed = message.data.completed
            max_margin = message.data.max_margin
            num_mislabeled = message.data.num_mislabeled
            support_vector_calculation_time = message.data.support_vector_calculation_time
            in_collision_calculation_time = message.data.in_collision_calculation_time
            kernel_calculation_time = message.data. kernel_calculation_time
            polyharmonic_weights_calculation_time = message.data.polyharmonic_weights_calculation_time
            total_time = message.data.total_time
            fk_support_vectors = worker.fk_X_buffer[:result_size].cpu().numpy().copy()
            poly_weights = worker.W_polyharmonic_buffer[:result_size +  fk_support_vectors.shape[1] + 1].cpu().numpy().copy()
            self.last_fk_support_vectors[i] = fk_support_vectors
            self.last_weights[i] = poly_weights
            # new_support_vectors_list.append(fk_support_vectors)
            # new_weights_list.append(poly_weights)
            print(f'Worker: {i}')
            print(f'Num. support vectors: {result_size}')
            print(f'Completed: {completed}')
            print(f'Max margin: {max_margin}')
            print(f'Num. mislabeled: {num_mislabeled}')
            print(f'Support vector calculation time: {support_vector_calculation_time}')
            print(f'In collision calculation time: {in_collision_calculation_time}')
            print(f'Kernel calculation time: {kernel_calculation_time}')
            print(f'Total time: {total_time}')
            print()
            self.initialized[i] = True
            self.last_fk_support_vectors[i] = fk_support_vectors
            self.last_weights[i] = poly_weights
        self.last_message = self.make_message(self.last_fk_support_vectors,self.last_weights)
        return self.last_message
        # return self.make_message(new_support_vectors_list,new_weights_list)

    def update_diff_co(self,point_cloud_torch):
        
        sigma_exploitation = 0.01
        num_exploration_samples = 2000
        
        exploitation_size = 0
        point_cloud_torch_gpu = None
        
        
        for i,worker in enumerate(self.workers):
            if (not self.processing[i]):
                t0 = time.perf_counter()
            # if (not any(self.processing)):
                if point_cloud_torch_gpu is None:
                    point_cloud_torch_gpu = point_cloud_torch.to('cuda')
                    self.point_cloud_torch_gpu = point_cloud_torch_gpu
                    

                # t0 = time.perf_counter()
                last_num_support_vectors = worker.num_support_vectors
                last_support_vectors = worker.X_buffer[:last_num_support_vectors]#$#.clone()
                last_weights = worker.W_buffer[:last_num_support_vectors]#.clone()
                last_H_s = worker.H_s_buffer[:last_num_support_vectors]#.clone()
                lower_limits = torch.tensor(self.robots[0].plant.GetPositionLowerLimits(),dtype = torch.float32,device = 'cuda')
                upper_limits = torch.tensor(self.robots[0].plant.GetPositionUpperLimits(),dtype = torch.float32,device = 'cuda')

                # get exploration samples based on saved trajectories
                
                x_initial_guess = self.initial_guess_control_points[i]
                random_indices = torch.randperm(x_initial_guess.shape[0])[:num_exploration_samples]
                exploration_samples = x_initial_guess[random_indices]
                exploration_samples += torch.randn_like(exploration_samples)*0.4
                exploration_samples = torch.clip(exploration_samples,lower_limits,upper_limits)
                exploration_samples = exploration_samples#.to('cuda')
                

                new_X = torch.vstack((last_support_vectors,exploration_samples))
                
                assert new_X.shape[0] <= worker.X_buffer.shape[0]
                worker.X_buffer[:new_X.shape[0]] = new_X
                
                worker.point_cloud_buffer[:point_cloud_torch_gpu.shape[0]] = point_cloud_torch_gpu[:worker.point_cloud_buffer.shape[0]]
                worker.comm_in.put(diff_co.Message(diff_co.MessageType.WORK,data=diff_co.WorkData(
                    sample_size = new_X.shape[0],
                    max_iterations=self.max_iterations_update,
                    point_cloud_size= min(point_cloud_torch_gpu.shape[0],worker.point_cloud_buffer.shape[0]),
                    covariance=self.covariance,
                    probability_threshold=self.prob_threshold,
                    exploitation_size=exploitation_size,
                    
                    ))) 
                # self.recording[f"r{i+1}"].append([point_cloud_torch_gpu.clone(),])
                self.processing[i] = True
                print("Total time to send SVM worker",i,time.perf_counter()-t0)
        
        
        
        
        for i,worker in enumerate(self.workers):
            try:
                message = worker.comm_out.get_nowait()
                t0 = time.perf_counter()
                result_size = message.data.result_size
                completed = message.data.completed
                max_margin = message.data.max_margin
                num_mislabeled = message.data.num_mislabeled
                support_vector_calculation_time = message.data.support_vector_calculation_time
                in_collision_calculation_time = message.data.in_collision_calculation_time
                kernel_calculation_time = message.data. kernel_calculation_time
                polyharmonic_weights_calculation_time = message.data.polyharmonic_weights_calculation_time
                total_time = message.data.total_time
                fk_support_vectors = worker.fk_X_buffer[:result_size].cpu().numpy().copy()
                poly_weights = worker.W_polyharmonic_buffer[:result_size +  fk_support_vectors.shape[1] + 1].cpu().numpy().copy()
                self.last_fk_support_vectors[i] = fk_support_vectors
                self.last_weights[i] = poly_weights

                # self.recording[f"r{i+1}"][-1]+=([fk_support_vectors.copy(),poly_weights.copy()])
                self.processing[i] = False
                print(f'Worker {i} update:')
                print(f'Num. support vectors: {result_size}')
                print(f'Completed: {completed}')
                print(f'Max margin: {max_margin}')
                print(f'Num. mislabeled: {num_mislabeled}')
                print(f'Support vector calculation time: {support_vector_calculation_time}')
                print(f'In collision calculation time: {in_collision_calculation_time}')
                print(f'Kernel calculation time: {kernel_calculation_time}')
                print(f'Total time: {total_time}')
                print(f"Receiving message total time",time.perf_counter()-t0)
                print()
                visualize = False
                if visualize:
                    self.meshcat.SetObject(f"/pc_svm_{i}", torch_to_drake_point_cloud(fk_support_vectors.reshape(-1,3)), 0.01,Rgba(0,0,1,0.2))
            except Empty:
                pass


        self.last_message = self.make_message(self.last_fk_support_vectors,self.last_weights)
        return self.last_message
    def calc_support_vectors(self, context,output):
        point_cloud_torch = self.point_cloud_input_port.Eval(context)

        # return
        if point_cloud_torch.numel() == 0:
            return
        # return
        if not all(self.initialized):
            print("Initializing DiffCo")
            t0 = time.perf_counter()
            message = self.initialize_diff_co(point_cloud_torch)
            # self.initialized = True
            output.set_value(message) 
        else:
            # asfsdfdfds
            t0 = time.perf_counter()
            message = self.update_diff_co(point_cloud_torch)
            output.set_value(message)
# import pyrealsense2 as rs

# import pyrealsense2 as rs

from pydrake.all import ConvertDepth16UTo32F,RotationMatrix
# from dt_apriltags import Detector

