
from pydrake.all import DrakeLcm,RigidTransform
from diff_co_lcm import lcmt_support_vector,lcmt_gaze_polytopes,lcmt_pose
import numpy as np
import threading
import time
from collections import deque
from typing import TypedDict

class SVM(TypedDict):
    weights: np.ndarray
    pol_weights: np.ndarray
    sv: np.ndarray
class VisionParameters(TypedDict):
    sampling_size_around_spot: int
    num_samples_spot: int
    num_samples_gaze: int
    num_samples_gaze_turn: int
    q7_guesses: int
    elevation: float
    robot_base_pose_1: np.ndarray

class LCMSubscriptionHandler:
    def __init__(
        self,
        collision_groups,
        sampling_size_around_spot,
        num_samples_spot,
        num_samples_gaze,
        num_samples_gaze_turn,
        q7_guesses,
        elevation,
        robot_base_pose_1,
        polyharmonic_kernel,
        kernel_gaze,
        meshcat,
    ):
        lcm = DrakeLcm()
        self.last_svm = {}

        for robot_name, group_names in collision_groups.items():
            self.last_svm[robot_name] = {}
            for group_name in group_names:
                lcm.Subscribe(
                    f"support_vectors_{robot_name}_{group_name}",
                    lambda msg, robot_name=robot_name, group_name=group_name: self.svm_callback(
                        msg, robot_name, group_name
                    ),
                )
                print(f"support_vectors_{robot_name}_{group_name}")
                self.last_svm[robot_name][group_name] = deque(maxlen=1000)
                # subscription = lcm.Subscribe(
                #     f"gaze_polytopes_{robot_name}_{group_name}",
                #     lambda msg, robot_name=robot_name, group_name=group_name: self.gaze_callback(
                #         msg, robot_name, group_name
                #     ),
                # )
        # subscription = lcm.Subscribe("support_vectors", self.svm_callback)
        subscription = lcm.Subscribe("gaze_polytopes", self.gaze_callback)
        subscription = lcm.Subscribe("final_pose", self.final_pose_callback)
        subscription = lcm.Subscribe("initial_pose", self.initial_pose_callback)
        self.lcm = lcm
        # self.last_svm = {deque(maxlen=1000)}
        # self.last_svm_vision = deque(maxlen=1000)
        self.recalculate_gaze_svm = True
        self.is_placing_spot_hidden = True
        self.placing_spot_pose = None
        self.initial_pose = None

        self.sampling_size_around_spot = sampling_size_around_spot
        self.num_samples_spot = num_samples_spot
        self.num_samples_gaze = num_samples_gaze
        self.num_samples_gaze_turn = num_samples_gaze_turn

        self.q7_guesses = q7_guesses
        self.elevation = elevation
        self.robot_base_pose_1 = robot_base_pose_1
        self.polyharmonic_kernel = polyharmonic_kernel
        self.kernel_gaze = kernel_gaze
        self.meshcat = meshcat
        self._stop_event = threading.Event()
        self.lcm_thread = threading.Thread(target=self.handle_subscriptions)
        self.lcm_thread.daemon = True
        self.lcm_thread.start()
    def handle_subscriptions(self):
        while not self._stop_event.is_set():
            # print('adfs')
            self.lcm.HandleSubscriptions(0)
            time.sleep(0.01)

    def stop(self):
        self._stop_event.set()
    def __del__(self):
        self.stop()
    def initial_pose_callback(self, *args):
        msg = lcmt_pose.decode(args[0])
        initial_pose = RigidTransform(np.asarray(msg.pose).reshape(4, 4)).GetAsMatrix4()
        self.initial_pose = initial_pose
        # print(initial_pose)

    def final_pose_callback(self, *args):
        msg = lcmt_pose.decode(args[0])
        placing_spot_pose = RigidTransform(np.asarray(msg.pose).reshape(4, 4)).GetAsMatrix4()
        self.placing_spot_pose = placing_spot_pose
        # print("Pose callback")

    def svm_callback(self, msg, robot_name, group_name):

        msg = lcmt_support_vector.decode(msg)

        support_vectors_1 = msg.support_vectors
        weights_1 = msg.weights[: msg.num_support_vectors]
        # constant_1 = msg.weights_1[msg.num_support_vectors_1]
        # pol_weights_1 = msg.weights_1[msg.num_support_vectors_1 + 1 :]
        pol_weights_1 = msg.weights[msg.num_support_vectors :]
        # print(msg.weights)
        sv_1 = np.array(support_vectors_1).T
        w_1 = np.array(weights_1).T
        # constant_1 = np.array(constant_1).T
        pols_1 = np.array(pol_weights_1).T
        self.last_svm[robot_name][group_name].append(
            SVM(weights=w_1, sv=sv_1, pol_weights=pols_1)
        )
    def display_polytopes(self):
        indices = self.last_indices
        simplices = self.last_simplices
        vertices = self.last_vertices
        for i in range(30):
            self.meshcat.Delete(f"/polytope_{i}")
        if indices is not None:
            for i, (a, b, c, d) in enumerate(indices):
                polytope_vertices = vertices[a:b]
                polytope_simplices = simplices[c:d]
                self.meshcat.SetTriangleMesh(
                    f"/polytope_{i}", polytope_vertices.T, polytope_simplices.T
                )
    def delete_polytopes(self):
        for i in range(30):
            self.meshcat.Delete(f"/polytope_{i}")
    def gaze_callback(self, *args):
        # global sampling_size_around_spot,num_samples_spot,num_samples_gaze,num_samples_gaze_turn,placing_spot_pose,q7_guesses,elevation,robot_base_pose_1
        msg = lcmt_gaze_polytopes.decode(args[0])
        placing_spot_pose = self.placing_spot_pose
        vertices = np.asarray(msg.vertices)
        simplices = np.asarray(msg.simplices)
        indices = np.asarray(msg.indices)
        # print(indices)
        self.last_vertices = vertices
        self.last_simplices = simplices
        self.last_indices = indices
        # TODO: calculate this once and just shift the points

        if self.is_placing_spot_hidden:
            self.is_placing_spot_hidden = msg.is_placing_spot_hidden
        # self.is_placing_spot_hidden = msg.is_placing_spot_hidden or (not self.is_placing_spot_hidden
        # if self.recalculate_gaze_svm and (self.placing_spot_pose is not None):
        #     pose_gaze_samples, iks_robot_1, transform_masks, ik_mask = (
        #         gaze.get_samples_around_placing_spot(
        #             sampling_size_around_spot,
        #             num_samples_spot,
        #             num_samples_gaze,
        #             num_samples_gaze_turn,
        #             placing_spot_pose,
        #             q7_guesses,
        #             elevation,
        #             robot_base_pose_1,
        #         )
        #     )
        #     line_point_ = pose_gaze_samples[:, :3, 3].astype(np.float64)
        #     line_direction_ = pose_gaze_samples[:, :3, 2].astype(np.float64)
        #     polytopes = []
        #     # print("fas")
        #     for i, (a, b, c, d) in enumerate(indices):
        #         polytope_vertices = vertices[a:b]

        #         polytope_simplices = simplices[c:d]

        #         polytopes.append(
        #             {"vertices": vertices[a:b], "simplices": simplices[c:d]}
        #         )
        #         mean_vertices = np.mean(polytope_vertices, axis=0)
        #         if (
        #             np.linalg.norm(mean_vertices - placing_spot_pose.translation())
        #             > 0.5
        #         ):
        #             continue
        #         polytope_vertices_ = polytope_vertices.astype(np.float64)
        #         faces = polytope_vertices[polytope_simplices]
        #         faces_ = np.array(faces).astype(np.float64)

        #         intersects = gaze.line_intersects_with_polytope_batch(
        #             polytope_vertices_,
        #             faces_,
        #             line_point_[transform_masks.squeeze()],
        #             line_direction_[transform_masks.squeeze()],
        #         )
        #         intersects_ = np.ones(transform_masks.shape[0], dtype=bool)
        #         intersects_[transform_masks.squeeze()] = intersects
        #         transform_masks = np.logical_and(
        #             transform_masks, ~intersects_.reshape(-1, 1)
        #         )
        #     # print("aaa")
        #     where_not_nan = np.nonzero(ik_mask[transform_masks.squeeze()])
        #     unique_rows, indices = np.unique(where_not_nan[0], return_index=True)
        #     unique_gazes = iks_robot_1[transform_masks.squeeze()][where_not_nan][
        #         indices
        #     ]
        #     support_vectors = torch.from_numpy(unique_gazes).to(torch.float32)
        #     Y_support_vectors = torch.ones(unique_gazes.shape[0])

        #     fk_support_vectors = (
        #         opt_collision.robots[0]
        #         .forward_kinematic_gaze(support_vectors)
        #         .squeeze()
        #     )
        #     mock_weights = torch.ones((fk_support_vectors.shape[0], 1))
        #     Y_support_vectors = torch.vmap(
        #         lambda sv, svv, w: self.kernel_gaze(sv, svv) @ w,
        #         in_dims=(0, None, None),
        #     )(
        #         fk_support_vectors.view(fk_support_vectors.shape[0], -1),
        #         fk_support_vectors.view(fk_support_vectors.shape[0], -1),
        #         mock_weights,
        #     )
        #     Y_support_vectors /= (Y_support_vectors).max()

        #     K_ph = torch.vmap(self.polyharmonic_kernel, in_dims=(0, None))(
        #         fk_support_vectors.view(fk_support_vectors.shape[0], -1),
        #         fk_support_vectors.view(fk_support_vectors.shape[0], -1),
        #     )
        #     polyharmonic_weights = torch.linalg.solve(
        #         K_ph, Y_support_vectors.to(torch.float32)
        #     )
        #     polyharmonic_weights = polyharmonic_weights - polyharmonic_weights.mean()
        #     polyharmonic_sv = fk_support_vectors.reshape(-1, 6)
        #     # is_placing_spot_hidden = msg.is_placing_spot_hidden

        #     polyharmonic_sv = polyharmonic_sv[
        #         : opt_collision.support_vectors_and_weights_gaze.size2_out(0)
        #     ]
        #     polyharmonic_weights = polyharmonic_weights[
        #         : opt_collision.support_vectors_and_weights_gaze.size2_in(1), ...
        #     ].reshape(opt_collision.support_vectors_and_weights_gaze.size1_in(1), -1)
        #     fk_support_vectors_1 = np.zeros(
        #         (opt_collision.support_vectors_and_weights_gaze.size_out(0))
        #     ).T
        #     fk_support_vectors_1[: polyharmonic_sv.shape[0]] = polyharmonic_sv
        #     weights_1 = np.zeros(
        #         opt_collision.support_vectors_and_weights_gaze.size_in(1)
        #     )
        #     weights_1[:, 0 : polyharmonic_weights.shape[1]] = polyharmonic_weights
        #     self.recalculate_gaze_svm = False
        #     self.last_svm_gaze = {"weights": weights_1, "fk_S": fk_support_vectors_1.T}

        #     # opt_collision.optimization_data['debug'].set_parameter('gaze_weights', weights_1)
        #     # opt_collision.optimization_data['debug'].set_parameter('gaze_support_vectors', fk_support_vectors_1.T)
            # opt_collision.optimization_data['debug'].set_parameter('gaze_cost_weight', np.array(-10.))

