from typing import Dict
from pydrake.multibody.tree import ModelInstanceIndex
# from pydrake.all import ModelInstanceIndex
from functools import partial
import numpy as np, sympy as sp, casadi as ca, typing as T
from projects.refactor_mpb.multibody_wrapper import MultiBodyPlantWrapper
import traceback, torch, os, sys, importlib, pathlib, copy, einops, ctypes, time
class Kernel(torch.nn.Module):
    def __init__(self,function,group_name,kernel):
        super(Kernel, self).__init__()
        self.kernel = kernel
        self.group_name = group_name
        self.function = function
        # self.register_buffer('_indices',torch.triu_indices(n,n).cpu())
    def polyharmonic_kernel(
        self, x: torch.Tensor, alpha: int
    ) -> torch.Tensor:
        xt = x.transpose(1,0)
        if alpha % 2 == 1:
            return torch.mean(torch.cdist(xt,xt) ** alpha,dim=0)
        else:
            r = torch.cdist(xt,xt)
            temp = (r**alpha) * torch.log(r)
            temp[torch.isnan(temp)] = 0.0

            return torch.mean(temp,dim=0)

    def rational_quadratic_kernel(
        self, x: torch.Tensor, alpha: int, length_scale: float
    ) -> torch.Tensor:
        # torch.mean(torch.cdist(fkx.transpose(1,0),fkx.transpose(1,0)),dim=0)
        xt = x.transpose(1,0)
        return torch.mean((
            (torch.cdist(xt,xt) ** 2.)
            / (2. * alpha * length_scale**2.)
            + 1.
        ) ** (-alpha),dim=0)
    def polyharmonic_kernel_cross(
        self, x: torch.Tensor,y: torch.Tensor, alpha: int
    ) -> torch.Tensor:
        xt = x.transpose(1,0)
        yt = y.transpose(1,0)
        if alpha % 2 == 1:
            return torch.mean(torch.cdist(xt,yt) ** alpha,dim=0)
        else:
            r = torch.cdist(xt,yt)
            temp = (r**alpha) * torch.log(r)
            temp[torch.isnan(temp)] = 0.0

            return torch.mean(temp,dim=0)

    def rational_quadratic_kernel_cross(
        self, x: torch.Tensor,y: torch.Tensor, alpha: int, length_scale: float
    ) -> torch.Tensor:
        # torch.mean(torch.cdist(fkx.transpose(1,0),fkx.transpose(1,0)),dim=0)
        xt = x.transpose(1,0)
        yt = y.transpose(1,0)
        # YouTubeVideo
        return torch.mean((
            (torch.cdist(xt,yt) ** 2.)
            / (2. * alpha * length_scale**2.)
            + 1.
        ) ** (-alpha),dim=0)
    # @torch.compile(dynamic=True)
    def forward(self,configuration,alpha,length_scale):
        if isinstance(configuration,(tuple,list)):
            fkx = self.function(configuration[0])
            fky = self.function(configuration[1])
            if self.kernel == "polyharmonic":
                K = self.polyharmonic_kernel_cross(fkx,fky, alpha)
            elif self.kernel == "rational_quadratic":
                K = self.rational_quadratic_kernel_cross(fkx,fky, alpha, length_scale)            
            return (fkx,fky),K
        fk = self.function(configuration)
        if self.kernel == "polyharmonic":
            K = self.polyharmonic_kernel(fk, alpha)
        elif self.kernel == "rational_quadratic":
            K = self.rational_quadratic_kernel(fk, alpha, length_scale)            
        return fk,K
class LinkCollisionChecker(torch.nn.Module):
    def __init__(self,function, radii):
        super(LinkCollisionChecker, self).__init__()
        self.function = function
        self.register_buffer('radii',radii)
        
    def forward(
        self,
        configuration: torch.Tensor,
        obstacle_points: torch.Tensor,
        obstacle_radii: torch.Tensor,
    ) -> torch.Tensor:
        robot_point_positions = self.function(configuration)
        robot_point_radii = self.radii
        obstacle_collision = (torch.cdist(robot_point_positions,obstacle_points) < (obstacle_radii.T + robot_point_radii) )
        obstacle_collision = torch.any(torch.any(obstacle_collision,dim=-1),dim=-1)
        del robot_point_positions
        return obstacle_collision

from scipy.special import gamma, loggamma
class ProbabilisticCollisionChecker(torch.nn.Module):
    def __init__(self,robot_geo_centers_functions,robot_point_radiis, indices,max_iterations = 10):
        super(ProbabilisticCollisionChecker, self).__init__()
        self.robot_geo_centers_functions = robot_geo_centers_functions

        radii = []
        for i,selection in enumerate(indices):
            radii.append(robot_point_radiis[i][selection])
        self.register_buffer("robot_point_radii", torch.cat(radii).reshape(-1,1))
        self.max_iterations = max_iterations
        self.register_buffer("K",torch.arange(self.max_iterations,dtype= torch.float32))
        self.INDICES_ = indices
    def collision_probability_geometries(self,r_1,s_1,E_o,mu_o,E_r,mu_r,):
        n = E_o.shape[1]
        Sigma = (E_o + E_r)#.to(torch.double)
        del E_o
        del E_r
        mu = (mu_r - mu_o)#.to(torch.double)
        del mu_r
        del mu_o
        radius_squared = (r_1+s_1)**2#.to(torch.double)
        del r_1
        del s_1
        diagonal_sigma = True
        if diagonal_sigma:
            eigenvalues = torch.diagonal (Sigma, dim1 = -2, dim2 = -1)
            b = (torch.diag_embed(eigenvalues**(-1/2))@mu.unsqueeze(-1)).reshape(mu.shape[0],mu.shape[1],-1)
        else:
            eigenvalues = torch.linalg.eigvalsh(Sigma)
            Sigma_chol = torch.linalg.cholesky(Sigma)
            b = (torch.diag_embed(eigenvalues**(-1/2)) @ Sigma_chol.transpose(-2,-1) @ torch.linalg.inv(Sigma_chol) @ mu.unsqueeze(-1)).reshape(mu.shape[0],mu.shape[1],-1)
            del Sigma_chol
        del Sigma
        b_squared = b**2
        b_sum = torch.sum(b_squared,dim=2)
        del b
        eig_prod_sqrt = torch.prod((2 * eigenvalues)**(-1/2),dim = -1)
        d0 = -1/2 * b_sum + torch.log(eig_prod_sqrt)
        c0 = torch.exp(-1/2 * b_sum) * eig_prod_sqrt
        del eig_prod_sqrt
        del b_sum
        cks = torch.empty_like(mu, ).resize_(mu_r.shape[0],mu_r.shape[1],self.max_iterations)
        dks = torch.empty_like(mu,).resize_(mu_r.shape[0],mu_r.shape[1],self.max_iterations)
        cks[...,0] = c0
        dks[...,0] = d0
        
        K = self.K
        dks = 0.5 * torch.sum((1 - K * b_squared.unsqueeze(-1)) * (2 * eigenvalues.unsqueeze(-1))**(-K), dim=-2)
        del eigenvalues
        del b_squared
        dks[...,0] = d0
        dks_flipped = torch.flip(dks, dims=[2])

        for k in range(1,self.max_iterations):
            cks[...,k] = 1/k * torch.sum(dks_flipped[..., -k-1:-1] * cks[..., :k], dim=2)
        del dks
        del dks_flipped
        P_C = torch.sum((-1) ** K * cks * radius_squared**(n/2+K) / torch.exp(torch.lgamma(n/2+K + 1)),dim = -1)
        del cks
        # return torch.clip(P_C,0,1)
        return P_C
    def collision_probability_geometries_simplified(self,r_1,s_1,eigenvalues,mu_o,mu_r,):
        n = 3
        mu = (mu_r - mu_o)#.to(torch.double)
        del mu_r
        del mu_o
        radius_squared = (r_1+s_1)**2#.to(torch.double)
        del r_1
        del s_1
        
        eig_inv = eigenvalues**(-1/2)
        mu[...,0] *=  eig_inv[...,0]
        mu[...,1] *=  eig_inv[...,1]
        mu[...,2] *=  eig_inv[...,2]
        b = mu
        b_squared = b**2
        b_sum = torch.sum(b_squared,dim=2)
        del b
        eig_prod_sqrt = torch.prod((2 * eigenvalues)**(-1/2),dim = -1)
        d0 = -1/2 * b_sum + torch.log(eig_prod_sqrt)
        c0 = torch.exp(-1/2 * b_sum) * eig_prod_sqrt
        del eig_prod_sqrt
        del b_sum
        # cks = torch.empty((mu.shape[0],mu.shape[1],self.max_iterations),device = mu.device, dtype= torch.float32)
        # dks = torch.empty((mu.shape[0],mu.shape[1],self.max_iterations),device = mu.device, dtype= torch.float32)
        cks = mu.new_empty(mu.shape[0],mu.shape[1],self.max_iterations)
        dks = mu.new_empty(mu.shape[0],mu.shape[1],self.max_iterations)
        cks[...,0] = c0
        dks[...,0] = d0
        K = self.K
        del mu
        dks = 0.5 * torch.sum((1 - K * b_squared.unsqueeze(-1)) * (2 * eigenvalues.unsqueeze(-1))**(-K), dim=-2)
        del eigenvalues
        del b_squared
        dks[...,0] = d0
        dks_flipped = torch.flip(dks, dims=[2])

        for k in range(1,self.max_iterations):
            cks[...,k] = 1/k * torch.sum(dks_flipped[..., -k-1:-1] * cks[..., :k], dim=2)
        del dks
        del dks_flipped
        P_C = torch.sum((-1) ** K * cks * radius_squared**(n/2+K) / torch.exp(torch.lgamma(n/2+K + 1)),dim = -1)
        del cks
        return torch.clip(P_C,0,1)
    def collision_probability_geometries_np(self, r_1, s_1, E_o, mu_o, E_r, mu_r):
        n = E_o.shape[1]
        Sigma = (E_o + E_r).astype(np.float128)
        mu = (mu_r - mu_o).astype(np.float128)
        radius_squared = ((r_1 + s_1) ** 2).astype(np.float128)

        diagonal_sigma = True
        if diagonal_sigma:
            eigenvalues = np.diagonal(Sigma, axis1=-2, axis2=-1)
            # b = (np.diag(eigenvalues**(-1/2)) @ mu[..., np.newaxis]).reshape(mu.shape[0], mu.shape[1], -1)
            b = (torch.diag_embed(torch.as_tensor(eigenvalues.astype(np.float64))**(-1/2)).numpy().astype(np.float128) @ mu[..., np.newaxis]).reshape(mu.shape[0], mu.shape[1], -1)
        else:
            eigenvalues = np.linalg.eigvalsh(Sigma)
            Sigma_chol = np.linalg.cholesky(Sigma)
            b = (np.diag(eigenvalues**(-1/2)) @ Sigma_chol.T @ np.linalg.inv(Sigma_chol) @ mu[..., np.newaxis]).reshape(mu.shape[0], mu.shape[1], -1)

        b_squared = b**2
        b_sum = np.sum(b_squared, axis=2)
        eig_prod_sqrt = np.prod((2 * eigenvalues) ** (-1/2), axis=-1)
        d0 = -1/2 * b_sum + np.log(eig_prod_sqrt)
        c0 = np.exp(-1/2 * b_sum) * eig_prod_sqrt

        cks = np.empty((mu.shape[0], mu.shape[1], self.max_iterations), dtype=np.float128)
        dks = np.empty((mu.shape[0], mu.shape[1], self.max_iterations), dtype=np.float128)
        cks[..., 0] = c0
        dks[..., 0] = d0
        K = np.arange(self.max_iterations, dtype=np.float128)

        for k in range(1, self.max_iterations):
            dks[..., k] = 0.5 * np.sum((1 - K[k] * b_squared) * (2 * eigenvalues) ** (-K[k]), axis=-1)
        
        dks_flipped = np.flip(dks, axis=2)

        for k in range(1, self.max_iterations):
            ck = 1/k * np.sum(dks_flipped[..., -k-1:-1] * cks[..., :k], axis=2)
            cks[..., k] = ck
        K_float64 = K.astype(np.float64)
        P_C = np.sum((-1) ** K * cks * radius_squared**(n/2 + K) / np.exp(loggamma((n/2 + K_float64 + 1).astype(np.float64)).astype(np.float128)), axis=-1)
         
        return P_C
    def forward(self,configuration, obstacle_points, obstacle_radii, obstacle_cov):
        # robot_point_positions = self.robot_geo_centers_function(configuration)[:,self.INDICES_,:]
        # robot_point_radii = self.robot_point_radii.reshape(-1,1)[self.INDICES_]
        robot_point_positions = []
        for i,selection in enumerate(self.INDICES_):
            robot_point_positions.append(self.robot_geo_centers_functions[i](configuration)[:,selection,:])
        robot_point_positions = torch.concatenate(robot_point_positions, dim=1)

        robot_point_radii = self.robot_point_radii
        obstacles_point_positions = obstacle_points
        obstacles_point_radii = obstacle_radii
        r_1 = robot_point_radii
        s_1 = obstacles_point_radii
        # cov_r = (torch.eye(3,device = configuration.device)*1e-12).repeat(robot_point_positions.shape[0],robot_point_radii.shape[0],1,1)
        cov_o = obstacle_cov
        eigenvalues = torch.diagonal(cov_o, dim1 = -2, dim2 = -1)

        indices1 = torch.arange(r_1.size(0),device = configuration.device).reshape(-1, 1)  # Reshape to n x 1
        indices2 = torch.arange(s_1.size(0),device = configuration.device).reshape(1, -1)  # Reshape to 1 x m
        indices1_grid = indices1.repeat(1 ,s_1.size(0))
        indices2_grid = indices2.repeat(r_1.size(0), 1)
        pair_indices = torch.stack((indices1_grid, indices2_grid), dim=-1).reshape(-1,2)
        r_1_paired = robot_point_radii.repeat(2,1)[pair_indices[:,0]]
        s_1_paired = obstacles_point_radii.repeat(2,1)[pair_indices[:,1]]
        # E_o_paired = cov_o.repeat(2,1,1)[pair_indices[:,1]]
        mu_o_paired = obstacles_point_positions.repeat(2,1)[pair_indices[:,1]]
        # E_r_paired = cov_r.repeat(2,1,1,1)[:robot_point_positions.shape[0],pair_indices[:,0]]
        mu_r_paired = robot_point_positions.repeat(2,1,1)[:robot_point_positions.shape[0],pair_indices[:,0]]
        eigenvalues = eigenvalues.repeat(2,1)[pair_indices[:,1]]
        # t0 = time.time()
        # collision_probabilities = self.collision_probability_geometries(r_1_paired,s_1_paired,E_o_paired,mu_o_paired,E_r_paired,mu_r_paired,)
        collision_probabilities = self.collision_probability_geometries_simplified(r_1_paired,s_1_paired,eigenvalues,mu_o_paired,mu_r_paired,)
        # print(time.time()-t0)
        del mu_r_paired
        del mu_o_paired
        del s_1_paired
        del r_1_paired
        prod = torch.prod(1 - collision_probabilities, dim = -1)
        out = torch.tensor([1.],device = configuration.device) - prod
        return out
    def collision_probabilities(self,configuration, obstacle_points, obstacle_radii, obstacle_cov):
        robot_point_positions = self.robot_geo_centers_function(configuration)#[:,self.INDICES_,:]
        robot_point_radii = self.robot_point_radii.reshape(-1,1)#[self.INDICES_]
        obstacles_point_positions = obstacle_points
        obstacles_point_radii = obstacle_radii
        r_1 = robot_point_radii
        s_1 = obstacles_point_radii
        cov_r = (torch.eye(3,device = configuration.device)*1e-6).expand(robot_point_positions.shape[0],robot_point_radii.shape[0],3,3)
        cov_o = obstacle_cov
        # cov_o = (torch.eye(3,device = configuration.device)*1e-2).expand(obstacles_point_radii.shape[0],3,3)
        
        indices1 = torch.arange(r_1.size(0),device = configuration.device).view(-1, 1)  # Reshape to n x 1
        indices2 = torch.arange(s_1.size(0),device = configuration.device).view(1, -1)  # Reshape to 1 x m
        indices1_grid = indices1.expand(-1, s_1.size(0))
        indices2_grid = indices2.expand(r_1.size(0), -1)
        pair_indices = torch.stack((indices1_grid, indices2_grid), dim=-1).view(-1,2)
        r_1_paired = robot_point_radii[pair_indices[:,0]]
        s_1_paired = obstacles_point_radii[pair_indices[:,1]]
        E_o_paired = cov_o[pair_indices[:,1]]
        mu_o_paired = obstacles_point_positions[pair_indices[:,1]]
        E_r_paired = cov_r[:,pair_indices[:,0]]
        mu_r_paired = robot_point_positions[:,pair_indices[:,0]]
        # t0 = time.time()
        collision_probabilities = self.collision_probability_geometries(r_1_paired,s_1_paired,E_o_paired,mu_o_paired,E_r_paired,mu_r_paired,)

        return collision_probabilities
    def collision_probabilities_np(self,configuration, obstacle_points, obstacle_radii, obstacle_cov):
        robot_point_positions = self.robot_geo_centers_function(configuration)[:,self.INDICES_,:]
        robot_point_radii = self.robot_point_radii.reshape(-1,1)[self.INDICES_]
        obstacles_point_positions = obstacle_points
        obstacles_point_radii = obstacle_radii
        r_1 = robot_point_radii
        s_1 = obstacles_point_radii
        cov_r = (torch.eye(3,device = configuration.device)*1e-6).expand(robot_point_positions.shape[0],robot_point_radii.shape[0],3,3)
        cov_o = obstacle_cov
        indices1 = torch.arange(r_1.size(0),device = configuration.device).view(-1, 1)  # Reshape to n x 1
        indices2 = torch.arange(s_1.size(0),device = configuration.device).view(1, -1)  # Reshape to 1 x m
        indices1_grid = indices1.expand(-1, s_1.size(0))
        indices2_grid = indices2.expand(r_1.size(0), -1)
        pair_indices = torch.stack((indices1_grid, indices2_grid), dim=-1).view(-1,2)
        r_1_paired = robot_point_radii[pair_indices[:,0]]
        s_1_paired = obstacles_point_radii[pair_indices[:,1]]
        E_o_paired = cov_o[pair_indices[:,1]]
        mu_o_paired = obstacles_point_positions[pair_indices[:,1]]
        E_r_paired = cov_r[:,pair_indices[:,0]]
        mu_r_paired = robot_point_positions[:,pair_indices[:,0]]
        collision_probabilities = self.collision_probability_geometries_np(r_1_paired.numpy(),s_1_paired.numpy(),E_o_paired.numpy(),mu_o_paired.numpy(),E_r_paired.numpy(),mu_r_paired.numpy(),)

        return collision_probabilities
    
class GeometricalModel:
    forward_kinematics_groups_torch: T.Dict[str, T.Callable]
    forward_kinematics_groups_casadi: T.Dict[str, ca.Function]
    links_by_group: T.Dict[str, T.List[str]]
    forward_kinematic_dimension_by_group: T.Dict[str, int]
    kernel_modules: Dict[str,Dict[str,Kernel]]
    link_collision_modules: Dict[str,LinkCollisionChecker]
    groups: T.List[str]
    all_geometries_torch: T.Callable[[torch.Tensor], torch.Tensor]
    all_geometry_radii_torch: torch.Tensor
    all_geometry_casadi: ca.Function
    all_geometry_radii_casadi: ca.DM
    robot_geometry_functions_torch: T.Dict[str, T.Callable[[torch.Tensor], torch.Tensor]]
    robot_geometry_radii_torch: T.Dict[str, torch.Tensor]
    robot_geometry_functions_casadi: T.Dict[str, ca.Function]
    robot_geometry_radii_casadi: T.Dict[str, ca.DM]
    
    def __init__(self, name, plant_wrapper, kernel_groups):
        diagram = plant_wrapper.diagram
        plant = diagram.GetSubsystemByName("plant")
        plant_wrapper = plant_wrapper
        self.name = name
        self.codegen_folder = plant_wrapper.TEMP_FOLDER
        self.geometries_folder = self.codegen_folder / self.name
        self.num_positions = plant.num_positions()
        self.suffix = "_" + self.name
        self.robot_geometry_functions_torch = {}
        self.robot_geometry_radii_torch = {}
        self.robot_geometry_functions_casadi = {}
        self.robot_geometry_radii_casadi = {}
        collision_geometries_robot = {}
        sys.path.append(str((self.geometries_folder / 'pytorch').resolve()))

        for body_index in plant.GetBodyIndices(ModelInstanceIndex(2)):
            body = plant.get_body(body_index)
            function_name = body.name() + self.suffix
            collision_geometries_robot[body.name()] = [
                geo for geo in plant.GetCollisionGeometriesForBody(body)
            ]
            functions = get_positions_function(
                "pytorch",
                plant_wrapper,
                collision_geometries_robot[body.name()],
                function_name,
                self.geometries_folder,
            )
            self.robot_geometry_functions_torch[body.name()] = (functions[
                "robot_point_positions_func"
            ])
            self.robot_geometry_radii_torch[body.name()] = functions[
                "robot_point_radii_func"
            ](
                torch.tensor(
                    [
                        0,
                    ]
                    * plant.num_positions(),
                    dtype=torch.float32,
                )
            ).reshape(
                -1, 1
            )
            functions = get_positions_function(
                "casadi",
                plant_wrapper,
                collision_geometries_robot[body.name()],
                function_name,
                self.geometries_folder,
            )
            self.robot_geometry_functions_casadi[body.name()] = functions[
                "robot_point_positions_func"
            ]
            self.robot_geometry_radii_casadi[body.name()] = functions[
                "robot_point_radii_func"
            ](
                ca.DM(
                    [
                        0,
                    ]
                    * plant.num_positions()
                )
            )
        all_geometries = [
            geo
            for body_index in plant.GetBodyIndices(ModelInstanceIndex(2))
            for geo in plant.GetCollisionGeometriesForBody(plant.get_body(body_index))
        ]

        functions = get_positions_function(
            "pytorch",
            plant_wrapper,
            all_geometries,
            "all_geometries",
            self.geometries_folder,
        )
        self.all_geometries_torch = (functions["robot_point_positions_func"])
        self.all_geometry_radii_torch = functions["robot_point_radii_func"](
            torch.tensor(
                [
                    0,
                ]
                * plant.num_positions(),
                dtype=torch.float32,
            )
        ).reshape(-1, 1)
        functions = get_positions_function(
            "casadi",
            plant_wrapper,
            all_geometries,
            "all_geometries",
            self.geometries_folder,
        )
        self.all_geometry_casadi = functions["robot_point_positions_func"]
        self.all_geometry_radii_casadi = functions["robot_point_radii_func"](
            ca.DM(
                [
                    0,
                ]
                * plant.num_positions()
            )
        )
        self.forward_kinematics_groups_torch = {}
        self.forward_kinematics_groups_casadi = {}
        self.links_by_group = {}
        self.forward_kinematic_dimension_by_group = {}
        self.groups = list(kernel_groups.keys())
        for group_name, group in kernel_groups.items():
            link_names = group["links"]
            sphere_indices_link = group["indices"]
            self.links_by_group[group_name] = link_names
            geometries = []
            dim = 0
            for link_name, sphere_indices in zip(link_names, sphere_indices_link):
                geometries += [
                    collision_geometries_robot[link_name][i] for i in sphere_indices
                ]
                dim += len(sphere_indices) * 3
            self.forward_kinematic_dimension_by_group[group_name] = dim
            functions = get_positions_function(
                "pytorch", plant_wrapper, geometries, group_name, self.geometries_folder
            )
            self.forward_kinematics_groups_torch[group_name] = (functions[
                "robot_point_positions_func"
            ])
            functions = get_positions_function(
                "casadi", plant_wrapper, geometries, group_name, self.geometries_folder
            )
            self.forward_kinematics_groups_casadi[group_name] = functions[
                "robot_point_positions_func"
            ]
        self.kernel_groups = kernel_groups
        self.make_modules()
    def make_modules(self):
        self.kernel_modules = {}
        self.probabilistic_collision_modules = {}
        for group_name in self.groups:
            self.kernel_modules[group_name] = {}
            for kernel_name in ['polyharmonic','rational_quadratic']:
                fk = self.forward_kinematics_groups_torch[group_name]
                self.kernel_modules[group_name][kernel_name] = Kernel(function = fk, group_name=group_name, kernel = kernel_name)
            if 'prob_indices' in self.kernel_groups[group_name]:
                prob_indices = self.kernel_groups[group_name]['prob_indices']
                group_link_functions = [self.robot_geometry_functions_torch[link] for i,link in enumerate(self.links_by_group[group_name]) if len(prob_indices[i]) > 0]
                group_link_radii = [self.robot_geometry_radii_torch[link] for i,link in enumerate(self.links_by_group[group_name]) if len(prob_indices[i]) > 0]
                indices = prob_indices
                indices = [torch.tensor(j) for i,j in enumerate(indices) if len(prob_indices[i]) > 0]
                prob = ProbabilisticCollisionChecker(robot_geo_centers_functions=group_link_functions,robot_point_radiis=group_link_radii,indices = indices,max_iterations = 10)
                self.probabilistic_collision_modules[group_name] = prob
        self.link_collision_modules = {}
        for link_name in self.robot_geometry_functions_torch.keys():
            self.link_collision_modules[link_name] = LinkCollisionChecker(self.robot_geometry_functions_torch[link_name],self.robot_geometry_radii_torch[link_name])
        self.link_collision_modules["all_geometries"] = LinkCollisionChecker(self.all_geometries_torch,self.all_geometry_radii_torch)
    def kernel_matrix(self,group_name,kernel_name,configuration,alpha,length_scale):
        return self.kernel_modules[group_name][kernel_name](configuration,alpha,length_scale)
    def is_group_in_collision(
        self,
        group_name: str,
        configuration: torch.Tensor,
        obstacle_points: torch.Tensor,
        obstacle_radii: torch.Tensor,
    ) -> torch.Tensor:
        in_collision = []
        for link_name in self.links_by_group[group_name]:
            in_collision.append(
                self.link_collision_modules[link_name](
                    configuration, obstacle_points, obstacle_radii
                )
            )
        in_collision = torch.any(torch.stack(in_collision), dim=0)
        return in_collision


    def __getstate__(self) -> object:
        state = {k: v for k, v in self.__dict__.items()}
        for k, v in self.__dict__.items():
            if k == "robot_geometry_functions_torch":
                state[k] = {}
                for body_name, func in v.items():
                    file_name = self.geometries_folder / (body_name + self.suffix)
                    state[k][body_name] = file_name
            if k == "all_geometries_torch":
                file_name = self.geometries_folder / "all_geometries"
                state[k] = file_name
            if k == "forward_kinematics_groups_torch":
                state[k] = {}
                for group_name, func in v.items():
                    file_name = self.geometries_folder / (group_name)
                    state[k][group_name] = file_name
            if k == "kernel_modules":
                state[k] = ''
            if k == "link_collision_modules":
                state[k] = ''
            if k == "probabilistic_collision_modules":
                state[k] = ''

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        sys.path.append(self.geometries_folder)
        for k, v in self.__dict__.items():
            if k == "robot_geometry_functions_torch":
                self.robot_geometry_functions_torch = {}
                for body_name, file_name in v.items():
                    module_name = body_name + self.suffix
                    file_name = (
                        self.geometries_folder
                        / "pytorch"
                        / (body_name + self.suffix + ".py")
                    )
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_name
                    )
                    foo = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = foo
                    spec.loader.exec_module(foo)
                    self.robot_geometry_functions_torch[body_name] = (getattr(
                        foo, module_name
                    ))
            if k == "all_geometries_torch":
                module_name = "all_geometries"
                file_name = self.geometries_folder / "pytorch" / ("all_geometries.py")
                spec = importlib.util.spec_from_file_location(module_name, file_name)
                foo = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = foo
                spec.loader.exec_module(foo)
                self.all_geometries_torch = getattr(foo, module_name)
            if k == "forward_kinematics_groups_torch":
                self.forward_kinematics_groups_torch = {}
                for group_name, file_name in v.items():
                    module_name = group_name
                    file_name = (
                        self.geometries_folder / "pytorch" / (group_name + ".py")
                    )
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_name
                    )
                    foo = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = foo
                    spec.loader.exec_module(foo)
                    self.forward_kinematics_groups_torch[group_name] = getattr(
                        foo, module_name
                    )
        self.make_modules()
    def partial_forward(
        self,
        group_name,
        configuration_1,
        configuration_2,
        kernel_type,
        alpha,
        length_scale,
    ):
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

        def rational_quadratic_kernel(
            x: torch.Tensor, y: torch.Tensor, alpha: int, length_scale: float
        ) -> torch.Tensor:
            return (
                (torch.linalg.norm(x - y, axis=-1, keepdim=False) ** 2)
                / (2 * alpha * length_scale**2)
                + 1
            ) ** (-alpha)
        fk_1 = self.forward_kinematics_groups_torch[group_name](configuration_1)
        fk_2 = self.forward_kinematics_groups_torch[group_name](configuration_2)
        indices1 = torch.arange(fk_1.size(0), device=fk_1.device).reshape(
            -1, 1
        )  # reshape to n x 1
        indices2 = torch.arange(fk_2.size(0), device=fk_1.device).reshape(
            1, -1
        )  # reshape to 1 x m
        # indices1_grid = indices1.repeat(1, fk.size(0))
        # indices2_grid = indices2.repeat(fk.size(0), 1)
        indices1_grid = indices1.expand(-1, fk_2.size(0))
        indices2_grid = indices2.expand(fk_1.size(0), -1)

        pair_indices = torch.stack((indices1_grid, indices2_grid), dim=-1).reshape(
            -1, 2
        )

        fk_1_paired = fk_1[pair_indices[:, 0]]
        fk_2_paired = fk_2[pair_indices[:, 1]]
        # del fk_1,fk_2
        # return torch.linalg.norm(a1-a2,axis=-1,keepdim=False)

        if kernel_type == "polyharmonic":
            K = torch.mean(
                polyharmonic_kernel(fk_1_paired, fk_2_paired, alpha), dim=-1
            ).reshape(configuration_1.size(0), configuration_2.size(0))
        elif kernel_type == "rational_quadratic":
            K = torch.mean(
                rational_quadratic_kernel(
                    fk_1_paired, fk_2_paired, alpha, length_scale
                ),
                dim=-1,
            ).reshape(configuration_1.size(0), configuration_2.size(0))
        # K = torch.mean(((torch.linalg.norm(fk_1_paired-fk_2_paired,axis=-1,keepdim=False)**2)/(2*alpha*length_scale**2)+1)**(-alpha),dim=-1).reshape(configuration_1.size(0),configuration_2.size(0))
        # for one configuration, fk returns a n x m x 2 tensor, where n is the number of FK "groups", and m is the number of points in each group
        del fk_1_paired, fk_2_paired
        # K = a1
        return fk_1, fk_2, K
def get_positions_function(
    module, plant_wrapper:MultiBodyPlantWrapper, collision_geometries, function_name, codegen_folder
):
    try:
        robot_point_positions_func = plant_wrapper.get_function_or_throw(
            function_name, module=module, path=codegen_folder
        )
        robot_point_radii_func = plant_wrapper.get_function_or_throw(
            function_name + "_radii", module=module, path=codegen_folder
        )
        return {
            "robot_point_positions_func": robot_point_positions_func,
            "robot_point_radii_func": robot_point_radii_func,
        }
    except:
        print(traceback.format_exc())
        pass
    try:
        robot_point_positions = plant_wrapper.get_function_or_throw(
            function_name, module="sympy", path=codegen_folder
        )
        robot_point_radii = plant_wrapper.get_function_or_throw(
            function_name + "_radii", module="sympy", path=codegen_folder
        )

    except:
        print(traceback.format_exc())
        expressions = make_position_robot_obstacle_expressions(
            plant_wrapper, collision_geometries
        )
        robot_point_positions = expressions["robot_positions"]
        robot_point_radii = expressions["robot_radii"]
    input_ = [sp.Matrix(plant_wrapper.sympy_position_variables)]
    robot_point_positions_func = plant_wrapper.make_function_from_expression(
        function_name, robot_point_positions, input_, path=codegen_folder, module=module
    )
    robot_point_radii_func = plant_wrapper.make_function_from_expression(
        function_name + "_radii",
        robot_point_radii,
        input_,
        path=codegen_folder,
        module=module,
    )
    return {
        "robot_point_positions_func": robot_point_positions_func,
        "robot_point_radii_func": robot_point_radii_func,
    }


def make_position_robot_obstacle_expressions(plant_wrapper, collision_geometries):
    scene_graph = plant_wrapper.diagram.GetSubsystemByName("scene_graph")
    inspector = scene_graph.model_inspector()
    plant = plant_wrapper.plant
    world_frame = plant.world_frame()
    robot_point_positions = []
    robot_point_radii = []
    for collision_geometry in collision_geometries:
        radius = inspector.GetShape(collision_geometry).radius()
        pose_in_frame = inspector.GetPoseInFrame(collision_geometry)
        col_frame = plant.GetBodyFromFrameId(
            inspector.GetFrameId(collision_geometry)
        ).body_frame()
        sympy_transform = (
            plant_wrapper.get_frame_pose_in_frame_function(
                col_frame, world_frame, 1e-6, module="sympy"
            )
            @ pose_in_frame.GetAsMatrix4()
        )
        robot_point_positions.append(sympy_transform[0:3, 3].reshape(1, 3))
        robot_point_radii.append([radius])
    robot_point_positions = sp.Matrix(robot_point_positions)
    robot_point_radii = sp.Matrix(robot_point_radii)

    return {"robot_positions": robot_point_positions, "robot_radii": robot_point_radii}

