import torch
import time
import numpy as np
from pydrake.all import PointCloud
def print_collision_score_on_samples(num_samples,planner,bspline_1,bspline_2):
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
        
    for s in np.linspace(0,1,num_samples):
        print(s)
        text= ''
        for robot in ['robot_1','robot_2']:
            for group_name in ['group_1','group_2','group_3','group_4']:
                svm = planner.lcm_subscription_handler.last_svm[robot][group_name][-1]
                
                weights = torch.as_tensor(svm['weights'])
                support_vectors = torch.as_tensor(svm['sv'])
                polynomial_weights = torch.as_tensor(svm['pol_weights']).squeeze()
                q = torch.as_tensor(bspline_2.evaluate(s) if robot == 'robot_2' else bspline_1.evaluate(s))
                col_model = planner.robot_1_collision_model if robot == 'robot_1' else planner.robot_2_collision_model
                fk_q = col_model.forward_kinematics_groups_torch[group_name](q)
                sample = fk_q.reshape(-1)
                dist = polyharmonic_kernel(sample, support_vectors.T, 1)
                score = torch.dot(weights.reshape(-1).to(torch.float64), dist.reshape(-1).to(torch.float64)) + polynomial_weights[0].reshape(-1).to(torch.float64) + torch.dot(polynomial_weights[1:].reshape(-1).to(torch.float64), sample.reshape(-1).to(torch.float64))
                # print(s,score)
                text += f'{group_name}: {score.item():.2f} '
            text+= '\n'
        print(text)
from diff_co_lcm import lcmt_global_solve

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
# print_collision_score_on_samples(10,planner,bspline_1_col,bspline_2_col)
def torch_to_drake_point_cloud(torch_pc):
    pc = PointCloud(torch_pc.shape[0])
    pc.mutable_xyzs()[:] = torch_pc.cpu().numpy().T
    return pc