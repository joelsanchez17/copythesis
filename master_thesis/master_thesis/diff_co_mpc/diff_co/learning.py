
import traceback, torch, os, sys, importlib, pathlib, copy, einops, ctypes, time, tempfile, string, random
from . import GeometricalModel
import numpy as np
import torch.multiprocessing as mp
import psutil

import concurrent.futures
from queue import Empty
# from pydrake.all import DrakeLcm
from pydrake.lcm import DrakeLcm
from diff_co_lcm.lcmt_support_vector import lcmt_support_vector
from diff_co_lcm.lcmt_global_solve import lcmt_global_solve
from utils.math.BSpline import BSpline
def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))
def recursive_mask(org, masks):
    if len(masks) == 0:
        return org
    return recursive_mask(org[masks[0]], masks[1:])


class GroupWorker:
    def __init__(
        self,
        group_i,
        robot_index,
        X_exploration_buffer,
        exploration_buffer_size,
        exploitation_buffer_size,
        so_file_d,
        max_num_support_vectors,
        device,
        dtype,
        affinity,

    ):
        
        self.affinity = affinity
        self.dtype = dtype
        self.so_file_d = so_file_d
        self.parent_pid = os.getpid()
        self.comm_in = mp.Queue()
        self.comm_out = mp.Queue()
        self.device = device
        self.X_exploration_buffer = X_exploration_buffer
        self.S_buffer = torch.zeros(
            (max_num_support_vectors, self.X_exploration_buffer.shape[1]),
            dtype=torch.float32,
            device=device,
        ).share_memory_()
        # self.fk_S_buffer = torch.zeros(
        #     (max_num_support_vectors, self.X_exploration_buffer.shape[1]),
        #     dtype=torch.float32,
        #     device=device,
        # ).share_memory_()
        self.X_exploitation_buffer = torch.zeros(
            (exploitation_buffer_size, self.X_exploration_buffer.shape[1]),
            dtype=torch.float32,
            device=device,
        ).share_memory_()
        # self.S_buffer = torch.zeros((max_num_support_vectors,sel), dtype=self.dtype,device=self.device).share_memory_()
        self.W_buffer = torch.zeros(
            (max_num_support_vectors, 1), dtype=self.dtype, device=self.device
        ).share_memory_()
        self.num_support_vectors_buffer = torch.zeros(
            1, dtype=torch.int32, device=self.device
        ).share_memory_()
        self.group_i = group_i
        self.robot_index = robot_index
        self.Y_by_group_buffer = torch.zeros(
            (exploitation_buffer_size + exploration_buffer_size, 1),
            dtype=self.dtype,
            device=self.device,
        ).share_memory_()
        self.K_by_group_buffer = torch.zeros(
            (
                exploitation_buffer_size + exploration_buffer_size,
                exploitation_buffer_size + exploration_buffer_size,
            ),
            dtype=self.dtype,
            device=self.device,
        ).share_memory_()
        self.thread = mp.Process(
            target=self.work,
            name=f"PID_{self.parent_pid}_GroupWorker_{robot_index}_{group_i}",
        )
        self.quit = False
        self.calculate_support_vectors_indexes = None
        self.time = mp.Value("d", 0.0)
    def start(self):
        self.thread.start()

    def stop(self):
        self.quit = True
        self.thread.join(1)
        if self.thread.exitcode is None:
            self.thread.terminate()

    def __del__(self):
        self.quit = True
        self.thread.join(1)
        if self.thread.exitcode is None:
            self.thread.terminate()

    def calculate_support_vectors_indexes_torch_float(
        self, Y, H, W, K, MAX_ITERATION=10000000
    ):
        rows, cols = Y.shape

        k_rows, k_cols = K.shape
        # cols = 1
        # # Convert numpy arrays to C arrays
        t0 = time.perf_counter()
        H_float = H.cpu().numpy()
        W_float = W.cpu().numpy()
        Y_float = Y.cpu().numpy()
        K_float = K.cpu().numpy()

        Y_c = Y_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        H_c = H_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        W_c = W_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        K_c = K_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        non_zero_W_indices = -np.ones(rows, dtype=np.int32)
        completed = np.zeros(cols, dtype=np.bool_)
        min_M = ctypes.c_float()
        zero_M_count = ctypes.c_int()
        iteration_number = ctypes.c_int()
        # print("time ctypes data as", time.perf_counter() - t0)

        t0 = time.perf_counter()
        
        self.lib_f.calculate_support_vectors_indexes(
            Y_c,
            H_c,
            W_c,
            K_c,
            rows,
            k_rows,
            k_cols,
            MAX_ITERATION,
            np.ctypeslib.as_ctypes(non_zero_W_indices),
            completed.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            ctypes.byref(min_M),
            ctypes.byref(zero_M_count),
            ctypes.byref(iteration_number),
        )
        # print("time in cpp", time.perf_counter() - t0)
        t0 = time.perf_counter()
        H.copy_(torch.as_tensor(H_float))
        W.copy_(torch.as_tensor(W_float))
        del H_float, W_float, Y_float, K_float
        # print("number of iterations: ", iteration_number.value)
        non_zero_W_indices = torch.as_tensor(
            non_zero_W_indices[non_zero_W_indices != -1]
        )
        return non_zero_W_indices, completed, min_M.value, zero_M_count.value

    def calculate_support_vectors_indexes_torch_double(
        self, Y, H, W, K, MAX_ITERATION=10000000
    ):
        rows, cols = Y.shape

        k_rows, k_cols = K.shape
        # cols = 1
        # # Convert numpy arrays to C arrays
        t0 = time.perf_counter()
        # H_double = H.to(self.dtype).cpu().numpy()
        # W_double = W.to(self.dtype).cpu().numpy()
        # Y_double = Y.to(self.dtype).cpu().numpy()
        # K_double = K.to(self.dtype).cpu().numpy()
        H_double = H.cpu().numpy()
        W_double = W.cpu().numpy()
        Y_double = Y.cpu().numpy()
        K_double = K.cpu().numpy()

        Y_c = Y_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        H_c = H_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        W_c = W_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        K_c = K_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # print("time ctypes data as", time.perf_counter() - t0)
        non_zero_W_indices = -np.ones(rows, dtype=np.int32)
        completed = np.zeros(cols, dtype=np.bool_)
        min_M = ctypes.c_double()
        zero_M_count = ctypes.c_int()
        iteration_number = ctypes.c_int()
        t0 = time.perf_counter()
        # print(K_double)
        # print(Y_double)
        self.lib_d.calculate_support_vectors_indexes(
            Y_c,
            H_c,
            W_c,
            K_c,
            rows,
            k_rows,
            k_cols,
            MAX_ITERATION,
            np.ctypeslib.as_ctypes(non_zero_W_indices),
            completed.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            ctypes.byref(min_M),
            ctypes.byref(zero_M_count),
            ctypes.byref(iteration_number),
        )
        print("time in cpp", time.perf_counter() - t0)
        t0 = time.perf_counter()
        H.copy_(torch.as_tensor(H_double))
        W.copy_(torch.as_tensor(W_double))
        del H_double, W_double, Y_double, K_double
        # H[:] = H_double#.to(torch.float32)
        # W[:] = W_double#.to(torch.float32)
        print("number of iterations: ", iteration_number.value)
        non_zero_W_indices = torch.as_tensor(
            non_zero_W_indices[non_zero_W_indices != -1]
        )
        print("time getting results back", time.perf_counter() - t0)
        return non_zero_W_indices, completed, min_M.value, zero_M_count.value
    
    @torch.compile(dynamic = True)
    def part_1_compile(self,Y_by_group_buffer,K_by_group_buffer,num_exploitation_samples):

        dtype = K_by_group_buffer.dtype
        device = K_by_group_buffer.device
        exploitation_K = K_by_group_buffer[
            : num_exploitation_samples :,
            : num_exploitation_samples,
        ].clone()
        exploitation_Y = Y_by_group_buffer[
            : num_exploitation_samples
        ].clone()
        exploitation_W = torch.zeros(num_exploitation_samples, Y_by_group_buffer.shape[1], dtype=dtype, device=device)
        exploitation_H = torch.zeros(num_exploitation_samples, Y_by_group_buffer.shape[1], dtype=dtype, device=device)
        return exploitation_Y, exploitation_H, exploitation_W, exploitation_K
    @torch.compile(dynamic = True)
    def part_2_compile(self, masks,final_SV_indices,K_group,W_group,Y_group,support_vector_indices,positive_similiarity_threshold,negative_similiarity_threshold,sample_removal_pos_H_multiplier,num_exploitation_samples):
        # masks = []
        num_support_vectors = support_vector_indices.shape[0]
        group_i = self.group_i
        support_vector_indices = support_vector_indices.reshape(-1)
        # print(f"{self.robot_index}_{group_i} partial time {time.perf_counter()-t_start}s")
        support_vector_mask = torch.zeros(
            W_group.shape[0], dtype=torch.bool, device=W_group.device
        )
        support_vector_mask[support_vector_indices] = True
        masks.append(support_vector_mask)
        # print('K_group shape',K_group.shape)
        # print('support_vector_indices',support_vector_indices,support_vector_indices.shape)
        gram_matrix = K_group[
            support_vector_indices
        ][:,support_vector_indices]
        # print('gram_matrix shape',gram_matrix.shape)
        W_group = W_group[support_vector_indices]
        Y_group = Y_group[support_vector_indices]
        mask = torch.ones(W_group.shape[0], dtype=bool, device=W_group.device)

        where_Y_positive = Y_group.reshape(-1) > 0
        where_Y_negative = Y_group.reshape(-1) < 0
        indices = torch.arange(0, Y_group.shape[0], device=Y_group.device)
        indices_positive = indices[where_Y_positive]
        indices_negative = indices[where_Y_negative]
        gram_matrix_positive = gram_matrix[where_Y_positive][:, where_Y_positive]
        gram_matrix_negative = gram_matrix[where_Y_negative][:, where_Y_negative]

        to_remove_positive = torch.unique(
            torch.argwhere(
                torch.triu(gram_matrix_positive, diagonal=1) > positive_similiarity_threshold
            )[:, 1]
        )
        to_remove_negative = torch.unique(
            torch.argwhere(
                torch.triu(gram_matrix_negative, diagonal=1) > negative_similiarity_threshold
            )[:, 1]
        )
        to_remove_positive_indices = indices_positive[to_remove_positive]
        to_remove_negative_indices = indices_negative[to_remove_negative]
        mask[to_remove_positive_indices] = False
        mask[to_remove_negative_indices] = False
        
        masks.append(mask)
        print(
            f"{self.robot_index}_{group_i} removing ",
            to_remove_positive_indices.size(),
            " in collision samples and ", to_remove_negative_indices.size(), " out of collision samples"
        )

        prev_H = (
            gram_matrix
            @ W_group
        )
        prev_H[Y_group.reshape(-1) > 0] *= sample_removal_pos_H_multiplier
        prev_H = prev_H[mask]
        filtered_gram_matrix = gram_matrix[mask][:, mask]
        num_support_vectors = mask.sum()
        # recovered_W = torch.linalg.inv(filtered_gram_matrix) @ prev_H
        recovered_W = torch.cholesky_solve(prev_H, torch.linalg.cholesky_ex(filtered_gram_matrix).L) #faster and more precise for float32 and symmetric matrices
        del prev_H, filtered_gram_matrix, gram_matrix
        # print(f"{self.robot_index}_{group_i} partial time {time.perf_counter()-t_start}s")


        indices = recursive_mask(final_SV_indices, masks)
        indices_exploitation = indices[indices < num_exploitation_samples]
        indices_exploration = (
            indices[indices >= num_exploitation_samples] - num_exploitation_samples
        )
        return masks, indices_exploitation, indices_exploration, recovered_W, num_support_vectors

    def compute_group(
        self,
        num_exploration_samples,
        num_exploitation_samples,
        positive_similiarity_threshold,
        negative_similiarity_threshold,
        sample_removal_pos_H_multiplier,
        max_iter,
    ):
        t_start = time.perf_counter()

        
        num_support_vectors_buffer = self.num_support_vectors_buffer
        W_buffer = self.W_buffer
        group_i = self.group_i
        worker_index = self.robot_index
        last_num_support_vectors = num_support_vectors_buffer.item()
        Y_group = self.Y_by_group_buffer[
            : num_exploration_samples + num_exploitation_samples
        ]
        # self.K_by_group_buffer[:] = 1.
        K_group = self.K_by_group_buffer[
            : num_exploration_samples + num_exploitation_samples,
            : num_exploration_samples + num_exploitation_samples,
        ]
        # print(self.K_by_group_buffer)
        final_SV_indices = torch.arange(
            0,
            num_exploitation_samples + num_exploration_samples,
            dtype=torch.int32,
            device=Y_group.device,
        )
        masks = []

        # W_group = torch.zeros(
        #     (num_exploitation_samples + num_exploration_samples, Y_group.shape[1]),
        #     dtype=self.dtype,
        #     device=Y_group.device,
        # )
        # W_group[:last_num_support_vectors] = W_buffer[
        #     :last_num_support_vectors
        # ]  # self.W_buffer[group_i][:last_num_support_vectors]

        
        # print(f"{self.robot_index}_{group_i} partial time {time.perf_counter()-t_start}s")
        if num_exploitation_samples > 0:
            exploitation_K = K_group[
                : num_exploitation_samples :,
                : num_exploitation_samples,
            ]
            exploitation_Y = Y_group[
                : num_exploitation_samples
            ]

            mask = torch.ones(Y_group.shape[0], dtype=bool, device=exploitation_Y.device)

            gram_matrix = exploitation_K
            where_Y_positive = exploitation_Y.reshape(-1) > 0
            where_Y_negative = exploitation_Y.reshape(-1) < 0
            indices = torch.arange(0, exploitation_Y.shape[0], device=exploitation_Y.device)
            indices_positive = indices[where_Y_positive]
            indices_negative = indices[where_Y_negative]
            gram_matrix_positive = gram_matrix[where_Y_positive][:, where_Y_positive]
            gram_matrix_negative = gram_matrix[where_Y_negative][:, where_Y_negative]

            to_remove_positive = torch.unique(
                torch.argwhere(
                    torch.triu(gram_matrix_positive, diagonal=1) > positive_similiarity_threshold
                )[:, 1]
            )
            to_remove_negative = torch.unique(
                torch.argwhere(
                    torch.triu(gram_matrix_negative, diagonal=1) > negative_similiarity_threshold
                )[:, 1]
            )

            to_remove_positive_indices = indices_positive[to_remove_positive]
            to_remove_negative_indices = indices_negative[to_remove_negative]
            
            mask[to_remove_positive_indices] = False
            mask[to_remove_negative_indices] = False
            masks.append(mask)
            mask = mask[:num_exploitation_samples]
            exploitation_K = gram_matrix[mask][:, mask].clone()
            exploitation_Y = exploitation_Y[mask].clone()
            exploitation_W = torch.zeros(exploitation_Y.shape[0], exploitation_Y.shape[1], dtype=self.dtype, device=exploitation_Y.device)
            
            exploitation_H = torch.zeros(exploitation_Y.shape[0], exploitation_Y.shape[1], dtype=self.dtype, device=exploitation_Y.device)
            
            # t0 = time.perf_counter()
            support_vector_indices, completed, max_margin, num_mislabeled = (
                self.calculate_support_vectors_indexes(
                    exploitation_Y,
                    exploitation_H,
                    exploitation_W,
                    exploitation_K,
                    max_iter * 1,
                )
            )
            # print(
            #     f"{self.robot_index}_{group_i} time 1st pass {worker_index} group {group_i}: {time.perf_counter()-t0}s"
            # )

            # print(completed, max_margin, num_mislabeled)
            # print(support_vector_indices.shape, last_num_support_vectors)

            support_vector_indices = support_vector_indices.reshape(-1)
            support_vector_mask = torch.zeros(
                num_exploration_samples + exploitation_Y.shape[0], dtype=torch.bool, device=self.device
            )
            support_vector_mask[support_vector_indices] = True
            support_vector_mask[-num_exploration_samples:] = True
            masks.append(support_vector_mask)

            indices = recursive_mask(final_SV_indices, masks)

            Y_group = Y_group[indices]
            W_group = torch.zeros(
                (Y_group.shape[0], Y_group.shape[1]),
                dtype=self.dtype,
                device=Y_group.device,
            )
            W_group[: support_vector_indices.shape[0]] = exploitation_W[
                support_vector_indices
            ]
            K_group = K_group[indices][:, indices]

            H_group = K_group @ W_group
            del exploitation_Y, exploitation_K, exploitation_W, exploitation_H
            # print(f"{self.robot_index}_{group_i} partial time {time.perf_counter()-t_start}s")
        else:

            H_group = torch.zeros(
                (Y_group.shape[0], Y_group.shape[1]), dtype=self.dtype
            )
            W_group = torch.zeros(
                (Y_group.shape[0], Y_group.shape[1]), dtype=self.dtype
            )
            
        support_vector_indices, completed, max_margin, num_mislabeled = (
            self.calculate_support_vectors_indexes(
                Y_group, H_group, W_group, K_group, max_iter
            )
        )
        
        num_support_vectors = support_vector_indices.shape[0]
        support_vector_indices = support_vector_indices.reshape(-1)
        # print(f"{self.robot_index}_{group_i} partial time {time.perf_counter()-t_start}s")
        support_vector_mask = torch.zeros(
            W_group.shape[0], dtype=torch.bool, device=W_group.device
        )
        support_vector_mask[support_vector_indices] = True
        masks.append(support_vector_mask)
        # print('K_group shape',K_group.shape)
        # print('support_vector_indices',support_vector_indices,support_vector_indices.shape)
        # gram_matrix = K_group[
        #     support_vector_indices
        # ][:,support_vector_indices]
        # # print('gram_matrix shape',gram_matrix.shape)
        # W_group = W_group[support_vector_indices]
        # Y_group = Y_group[support_vector_indices]
        # mask = torch.ones(W_group.shape[0], dtype=bool, device=W_group.device)

        # where_Y_positive = Y_group.reshape(-1) > 0
        # where_Y_negative = Y_group.reshape(-1) < 0
        # indices = torch.arange(0, Y_group.shape[0], device=Y_group.device)
        # indices_positive = indices[where_Y_positive]
        # indices_negative = indices[where_Y_negative]
        # gram_matrix_positive = gram_matrix[where_Y_positive][:, where_Y_positive]
        # gram_matrix_negative = gram_matrix[where_Y_negative][:, where_Y_negative]

        # to_remove_positive = torch.unique(
        #     torch.argwhere(
        #         torch.triu(gram_matrix_positive, diagonal=1) > positive_similiarity_threshold
        #     )[:, 1]
        # )
        # to_remove_negative = torch.unique(
        #     torch.argwhere(
        #         torch.triu(gram_matrix_negative, diagonal=1) > negative_similiarity_threshold
        #     )[:, 1]
        # )
        # to_remove_positive_indices = indices_positive[to_remove_positive]
        # to_remove_negative_indices = indices_negative[to_remove_negative]
        
        # mask[to_remove_positive_indices] = False
        # mask[to_remove_negative_indices] = False
        
        
        # masks.append(mask)
        # # print(
        # #     f"{self.robot_index}_{group_i} removing ",
        # #     to_remove_positive_indices.size(),
        # #     " in collision samples and ", to_remove_negative_indices.size(), " out of collision samples"
        # # )

        # prev_H = (
        #     gram_matrix
        #     @ W_group
        # )
        # prev_H[Y_group.reshape(-1) > 0] *= sample_removal_pos_H_multiplier
        # prev_H = prev_H[mask]
        # filtered_gram_matrix = gram_matrix[mask][:, mask]
        # num_support_vectors = mask.sum()
        # # recovered_W = torch.linalg.inv(filtered_gram_matrix) @ prev_H
        # recovered_W = torch.cholesky_solve(prev_H, torch.linalg.cholesky_ex(filtered_gram_matrix).L) #faster and more precise for float32 and symmetric matrices
        # del prev_H, filtered_gram_matrix, gram_matrix
        # print(f"{self.robot_index}_{group_i} partial time {time.perf_counter()-t_start}s")


        indices = recursive_mask(final_SV_indices, masks)
        indices_exploitation = indices[indices < num_exploitation_samples]
        indices_exploration = (
            indices[indices >= num_exploitation_samples] - num_exploitation_samples
        )

        if (
            indices_exploitation.shape[0] + indices_exploration.shape[0]
            > self.S_buffer.shape[0]
        ):
            print(
                f"{self.robot_index}_{group_i}, error indices_exploitation.shape[0] + indices_exploration.shape[0] > self.S_buffer.shape[0]"
            )
        else:


            Y_s = self.Y_by_group_buffer[indices]
            self.Y_by_group_buffer[: indices.shape[0]] = Y_s
            
            self.S_buffer[: indices_exploitation.shape[0]] = self.X_exploitation_buffer[
                indices_exploitation
            ]
            self.S_buffer[
                indices_exploitation.shape[0] : indices_exploitation.shape[0]
                + indices_exploration.shape[0]
            ] = self.X_exploration_buffer[indices_exploration]
            W_buffer[:num_support_vectors] = W_group[support_vector_mask, :]
            # W_buffer[:num_support_vectors] = recovered_W
            num_support_vectors_buffer[:] = num_support_vectors
            # print(f"{self.robot_index}_{group_i} num_support_vectors", num_support_vectors)
            
        del H_group, K_group, Y_group

        del W_group
        self.comm_out.put({"num_support_vectors": num_support_vectors})
        self.time.value = time.perf_counter() - t_start
        # print(f"{self.robot_index}_{group_i} total_time {time.perf_counter()-t_start}s")
        # print()
    def work(self):
        psutil.Process().cpu_affinity(self.affinity)

        if self.dtype == torch.float32:
            print("loading float")
            self.lib_f = ctypes.CDLL(str(self.so_file_d))
            self.lib_f.calculate_support_vectors_indexes.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_bool),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            self.calculate_support_vectors_indexes = self.calculate_support_vectors_indexes_torch_float
        else:
            print("loading double")
            self.lib_d = ctypes.CDLL(str(self.so_file_d))
            self.lib_d.calculate_support_vectors_indexes.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_bool),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
        ]
            self.calculate_support_vectors_indexes = self.calculate_support_vectors_indexes_torch_double
            
        while True:
            try:
                if not os.path.exists(f"/proc/{self.parent_pid}"):
                    print(
                        f"Parent process {self.parent_pid} is dead. {self.robot_index}_{self.group_i} worker exiting."
                    )
                    break
                if self.quit:
                    print(f"Worker {self.robot_index}_{self.group_i} is quitting.")
                    break
                message = self.comm_in.get(block=True, timeout=0.01)
                num_exploration_samples = message["num_exploration_samples"]
                num_exploitation_samples = message["num_exploitation_samples"]
                positive_similiarity_threshold = message["positive_similiarity_threshold"]
                negative_similiarity_threshold = message["negative_similiarity_threshold"]
                sample_removal_pos_H_multiplier = message["sample_removal_pos_H_multiplier"]
                max_iter = message["max_iter"]
                # print(self.robot_index, self.group_i, 'starting compute')
                self.compute_group(
                    num_exploration_samples,
                    num_exploitation_samples,
                    positive_similiarity_threshold,
                    negative_similiarity_threshold,
                    sample_removal_pos_H_multiplier,
                    max_iter,
                )
                # self.compute_group_with_compile(
                #     num_exploration_samples,
                #     num_exploitation_samples,
                #     positive_similiarity_threshold,
                #     negative_similiarity_threshold,
                #     sample_removal_pos_H_multiplier,
                #     max_iter,
                # )
            except Empty:
                pass
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
from multiprocessing import shared_memory
class SupportVectorWorker:
    def __init__(
        self,
        index,
        point_cloud_max_size,
        exploration_buffer_size,
        exploitation_buffer_size,
        num_samples_max_size,
        max_num_support_vectors,
        collision_model: GeometricalModel,
        dtype,
        affinity,
        affinity_by_group,
        device="cuda",
        compile=True,
        probabilistic = False,

    ):
        self.probabilistic = probabilistic
        self.affinity = affinity
        self.affinity_by_group = affinity_by_group
        self.dtype = dtype
        self.collision_model = collision_model
        self.forward_kinematic_dimension_by_group = (
            collision_model.forward_kinematic_dimension_by_group
        )
        self.num_fk_groups = len(collision_model.links_by_group.keys())
        self.parent_pid = os.getpid()
        self.X_exploration_buffer = torch.zeros(
            (num_samples_max_size, collision_model.num_positions),
            dtype=torch.float32,
            device=device,
        ).share_memory_()
        # self.fk_X_buffer = torch.zeros((num_of_fk_groups,num_samples_max_size,support_vectors_dimen), dtype=torch.float32,device=device).share_memory_()
        # self.W_polyharmonic_buffer = torch.zeros((num_of_fk_groups,max_num_support_vectors,), dtype=torch.float32,device=device).share_memory_()
        self.point_cloud_buffer = torch.zeros(
            (point_cloud_max_size, 3), dtype=torch.float32, device=device
        ).share_memory_()
        self.num_support_vectors = torch.zeros(
            self.num_fk_groups, dtype=torch.int32, device=device
        ).share_memory_()
        self.quit = False
        self.index = index
        self.device = device
        self.comm_in = mp.Queue()
        self.comm_out = mp.Queue()
        self.compile = compile
        temp_folder = pathlib.Path(tempfile.mkdtemp())
        so_file_d = temp_folder / (randomword(10) + ".so")
        print("Compiling thingy")
        if dtype == torch.float32:
            print(
                os.system(
                    # f"bash -c 'source /opt/intel/oneapi/setvars.sh &&  icpx -shared -o {str(so_file_d)} -qmkl=sequential -ffast-math -Ofast -mfma -mavx -mavx2 -flto -march=native -fPIC ../diff_co_mpc/diff_co/source/calculate_support_vector_indexes_float.cpp -I /usr/include/eigen3 -I /usr/include/mkl' "
                    f"bash -c 'gcc -shared -o {str(so_file_d)} -ffast-math -Ofast -mfma -mavx -mavx2 -flto -march=native -fPIC ../diff_co_mpc/diff_co/source/calculate_support_vector_indexes_float.cpp -I /usr/include/eigen3 -I /usr/include/mkl' "
                )
            )
        else:
            print(
                os.system(
                    # f"bash -c 'source /opt/intel/oneapi/setvars.sh &&  icpx -shared -o {str(so_file_d)} -qmkl=sequential -ffast-math -Ofast -mfma -mavx -mavx2 -flto -march=native -fPIC ../diff_co_mpc/diff_co/source/calculate_support_vector_indexes_double.cpp -I /usr/include/eigen3 -I /usr/include/mkl' "
                    f"bash -c 'gcc -shared -o {str(so_file_d)} -ffast-math -Ofast -mfma -mavx -mavx2 -flto -march=native -fPIC ../diff_co_mpc/diff_co/source/calculate_support_vector_indexes_double.cpp -I /usr/include/eigen3 -I /usr/include/mkl' "
                )
            )
        self.group_workers = {}
        for i,group_name in enumerate(self.collision_model.groups):
            self.group_workers[group_name] = GroupWorker(
                group_i=group_name,
                robot_index=self.index,
                exploration_buffer_size=exploration_buffer_size,
                X_exploration_buffer=self.X_exploration_buffer,
                exploitation_buffer_size=exploitation_buffer_size,
                device=self.device,
                max_num_support_vectors=max_num_support_vectors,
                so_file_d=so_file_d,
                dtype = self.dtype,
                affinity=self.affinity_by_group[i],
            )
        self.thread = mp.Process(
            target=self.work,
            name=f"PID_{self.parent_pid}_SupportVectorWorker_{index}",
        )
        self.time = mp.Value("d", 0.0)
    # def polyharmonic_kernel(
    #     self, x: torch.Tensor, alpha: int
    # ) -> torch.Tensor:
    #     xt = x.transpose(1,0)
    #     if alpha % 2 == 1:
    #         return torch.mean(torch.cdist(xt,xt) ** alpha,dim=0)
    #     else:
    #         r = torch.cdist(xt,xt)
    #         temp = (r**alpha) * torch.log(r)
    #         temp[torch.isnan(temp)] = 0.0

    #         return torch.mean(temp,dim=0)
    def calculate_polyharmonic_weights(self,support_vectors, Y_support_vectors):
        N, d = support_vectors.shape
        K_ph = torch.vmap(lambda x,y: torch.linalg.norm(x-y,axis=-1)**1, in_dims = (0,None))(support_vectors.view(support_vectors.shape[0],-1),support_vectors.view(support_vectors.shape[0],-1))
        B = torch.cat([torch.ones((N, 1),device=support_vectors.device), support_vectors], dim=1)
        top = torch.cat([K_ph, B], dim=1)
        bottom = torch.cat([B.T, torch.zeros((d+1, d+1), device=support_vectors.device)], dim=1)
        M = torch.cat([top, bottom], dim=0)
        f = torch.cat([Y_support_vectors, torch.zeros((d+1, 1), device=support_vectors.device)])
        # print(M)
        try:
            polyharmonic_weights = torch.linalg.solve(M, f)
        except:
            print(traceback.format_exc())
            M_inv = torch.linalg.pinv(M)
            polyharmonic_weights = M_inv @ f

        return polyharmonic_weights
    def update_point_cloud(self, point_cloud):
        if point_cloud.shape[0] > self.point_cloud_buffer.shape[0]:
            print(
                f"Point cloud size {point_cloud.shape[0]} is larger than the buffer size {self.point_cloud_buffer.shape[0]}."
            )
            self.point_cloud_buffer[:] = point_cloud[: self.point_cloud_buffer.shape[0]]
        else:
            self.point_cloud_buffer[: point_cloud.shape[0]] = point_cloud

    def start(self):
        self.thread.start()
        for worker in self.group_workers.values():
            worker.start()

    def __del__(self):
        self.quit = True
        for worker in self.group_workers.values():
            worker.stop()
        self.thread.join(1)
        if self.thread.exitcode is None:
            self.thread.terminate()

    def stop(self):
        self.quit = True
        for worker in self.group_workers.values():
            worker.stop()
        self.thread.join(1)
        if self.thread.exitcode is None:
            self.thread.terminate()

    def work(self):
        psutil.Process().cpu_affinity(self.affinity)
        remove_shm_from_resource_tracker()
        lcm = DrakeLcm()
        trajectory_points = None
        def global_trajectory_callback(msg):
            nonlocal  trajectory_points
            # print('hello')
            msg = lcmt_global_solve.decode(msg)
            if msg.bspline_robot_1.order == 0:
                trajectory_points = None
                # print(msg,msg.bspline_robot_1.order,msg.bspline_robot_1.control_points)
                return
            if self.index == 1:
                bspline = BSpline(np.asarray(msg.bspline_robot_1.control_points), msg.bspline_robot_1.order)
                trajectory_points = torch.as_tensor(bspline.fast_batch_evaluate(np.linspace(0,1,80)),dtype = self.dtype,device = self.device)
            elif self.index == 2:
                bspline = BSpline(np.asarray(msg.bspline_robot_2.control_points), msg.bspline_robot_2.order)
                trajectory_points = torch.as_tensor(bspline.fast_batch_evaluate(np.linspace(0,1,80)),dtype = self.dtype,device = self.device)
            else:
                print('Invalid robot index')
                
        lcm.Subscribe("global_trajectory", global_trajectory_callback)
        kernel_modules = {}
        link_collision_modules = {}
        probabilistic_collision_modules = {}
        fk_functions = {}
        for group_name in self.collision_model.groups:
            kernel_modules[group_name] = {}
            for kernel_name in ['polyharmonic','rational_quadratic']:
                kernel_modules[group_name][kernel_name] = self.collision_model.kernel_modules[group_name][kernel_name].to(self.device)
            fk_functions[group_name] = self.collision_model.forward_kinematics_groups_torch[group_name]
            if self.probabilistic:
                probabilistic_collision_modules[group_name] = self.collision_model.probabilistic_collision_modules[group_name].to(self.device)
                # self.collision_model.kernel_modules[group_name][kernel_name] = self.collision_model.kernel_modules[group_name][kernel_name].to(self.device)
        if not self.probabilistic:
            for link_name in self.collision_model.link_collision_modules:
                link_collision_modules[link_name] = self.collision_model.link_collision_modules[link_name].to(self.device)
        calculate_polyharmonic_weights = self.calculate_polyharmonic_weights
        # for group
            # self.collision_model.link_collision_modules[link_name] = self.collision_model.link_collision_modules[link_name].to(self.device)
        if self.compile:
            print('compiling shit')
            with torch.no_grad():
                alpha_ = torch.tensor(2., device=self.device,dtype = self.dtype)
                length_scale_ = torch.tensor(2., device=self.device,dtype = self.dtype)
                configuration_: torch.Tensor = torch.randn(1000, 9, device=self.device,dtype = self.dtype)
                obstacle_points_: torch.Tensor = torch.randn(300, 3, device=self.device,dtype = self.dtype)
                obstacle_radii_: torch.Tensor = torch.randn(300, 1, device=self.device,dtype = self.dtype)
                cov = torch.eye(3,device=self.device,dtype = self.dtype).repeat(obstacle_points_.shape[0],300,1)
                for group_name in self.collision_model.groups:
                    for kernel_name in ['polyharmonic','rational_quadratic']:
                        kernel_modules[group_name][kernel_name] = torch.compile(kernel_modules[group_name][kernel_name],dynamic=True)
                        kernel_modules[group_name][kernel_name](configuration_,alpha_,length_scale_)
                    fk_functions[group_name] = torch.compile(fk_functions[group_name],dynamic=True)
                    fk_functions[group_name](configuration_)
                if not self.probabilistic:
                    for link_name in self.collision_model.link_collision_modules:
                        link_collision_modules[link_name] = torch.compile(link_collision_modules[link_name],dynamic=True)
                        link_collision_modules[link_name](configuration_,obstacle_points_,obstacle_radii_)
                else:
                    for group_name in self.collision_model.groups:
                        probabilistic_collision_modules[group_name] = torch.compile(probabilistic_collision_modules[group_name],dynamic=True)
                        probabilistic_collision_modules[group_name](configuration_,obstacle_points_,obstacle_radii_,cov)
                        # torch.randn(samples,9,device =  'cuda')*0,obstacles,radii,cov
                alpha_ = torch.tensor(3., device=self.device,dtype = self.dtype)
                length_scale_ = torch.tensor(1., device=self.device,dtype = self.dtype)
                configuration_: torch.Tensor = torch.randn(3000, 9, device=self.device,dtype = self.dtype)
                obstacle_points_: torch.Tensor = torch.randn(400, 3, device=self.device,dtype = self.dtype)
                obstacle_radii_: torch.Tensor = torch.randn(400, 1, device=self.device,dtype = self.dtype)
                cov = torch.eye(3,device=self.device,dtype = self.dtype).repeat(obstacle_points_.shape[0],400,1)
                for group_name in self.collision_model.groups:
                    for kernel_name in ['polyharmonic','rational_quadratic']:
                        kernel_modules[group_name][kernel_name](configuration_,alpha_,length_scale_)
                    fk_functions[group_name](configuration_)
                if not self.probabilistic:
                    for link_name in self.collision_model.link_collision_modules:
                        link_collision_modules[link_name](configuration_,obstacle_points_,obstacle_radii_)
                else:
                    for group_name in self.collision_model.groups:                        
                        probabilistic_collision_modules[group_name](configuration_,obstacle_points_,obstacle_radii_,cov)
                alpha_ = torch.tensor(3., device=self.device,dtype = self.dtype)
                length_scale_ = torch.tensor(1., device=self.device,dtype = self.dtype)
                configuration_: torch.Tensor = torch.randn(50, 9, device=self.device,dtype = self.dtype)
                obstacle_points_: torch.Tensor = torch.randn(20, 3, device=self.device,dtype = self.dtype)
                obstacle_radii_: torch.Tensor = torch.randn(20, 1, device=self.device,dtype = self.dtype)
                cov = torch.eye(3,device=self.device,dtype = self.dtype).repeat(obstacle_points_.shape[0],20,1)
                for group_name in self.collision_model.groups:
                    for kernel_name in ['polyharmonic','rational_quadratic']:
                        kernel_modules[group_name][kernel_name](configuration_,alpha_,length_scale_)
                    fk_functions[group_name](configuration_)
                if not self.probabilistic:
                    for link_name in self.collision_model.link_collision_modules:
                        link_collision_modules[link_name](configuration_,obstacle_points_,obstacle_radii_)
                else:
                    for group_name in self.collision_model.groups:                        
                        probabilistic_collision_modules[group_name](configuration_,obstacle_points_,obstacle_radii_,cov)
        array_shape = (10000,3)
        dtype = np.float32
        shared_memory_name = "point_cloud_shared_memory"

        try:
            shm = shared_memory.SharedMemory(name=shared_memory_name, create=True, size=np.prod(array_shape) * np.dtype(dtype).itemsize)
            shared_array = np.ndarray(array_shape, dtype=dtype, buffer=shm.buf)
            shared_array[:] = np.nan
            print("Created new shared memory.")
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=shared_memory_name)
            shared_array = np.ndarray(array_shape, dtype=dtype, buffer=shm.buf)
            print("Linked to existing shared memory.")
        def is_group_in_collision(
            group_name: str,
            configuration: torch.Tensor,
            obstacle_points: torch.Tensor,
            obstacle_radii: torch.Tensor,
        ) -> torch.Tensor:
            in_collision = []
            for link_name in self.collision_model.links_by_group[group_name]:
                in_collision.append(
                    link_collision_modules[link_name](
                        configuration, obstacle_points, obstacle_radii
                    )
                )
            in_collision = torch.any(torch.stack(in_collision), dim=0)
            return in_collision
        lower_limits = torch.tensor(
            torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
        0.    ,  0.    ]), dtype=self.dtype, device=self.device
        )
        upper_limits = torch.tensor(
            torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,
        0.04  ,  0.04  ]), dtype=self.dtype, device=self.device
        )
        upper_limits[-2:] = 0.001
        lower_limits[-2:] = 0

        uniform_distribution = torch.distributions.uniform.Uniform(lower_limits, upper_limits)
        
        message = self.comm_in.get(block=True, timeout=600)
        num_exploration_samples = message["num_exploration_samples"]
        num_exploration_trajectory_samples = message["num_exploration_trajectory_samples"]
        num_exploitation_samples_by_group = message["num_exploitation_samples_by_group"]
        sample_removal_pos_H_multiplier_by_group = message["sample_removal_pos_H_multiplier_by_group"]
        negative_similiarity_threshold_by_group = message["negative_similiarity_threshold_by_group"]
        positive_similiarity_threshold_by_group = message["positive_similiarity_threshold_by_group"]
        length_scale_by_group = message["length_scale_by_group"]
        alpha_by_group = message["alpha_by_group"]
        max_num_SV_by_group = message["max_num_SV_by_group"]
        sigmoid_alpha = message["sigmoid_alpha"]                
        exploitation_covariance = message["exploitation_covariance"]
        max_iter = message["max_iter"]
        point_cloud_size = message["point_cloud_size"]
        obstacle_radii = message["obstacle_radii"]
        if self.probabilistic:
            obstacle_covariance = message["obstacle_covariance"]
            prob_threshold = message["prob_threshold"]
        # obstacle_points = self.point_cloud_buffer[:point_cloud_size]
        # obstacle_radii = torch.tensor(obstacle_radii, dtype=self.dtype, device=self.device).expand(point_cloud_size, 1)
        while True:
            if not os.path.exists(f"/proc/{self.parent_pid}"):
                print(
                    f"Parent process {self.parent_pid} is dead. {self.robot_index}_{self.group_i} worker exiting."
                )
                break
            if self.quit:
                print(f"Worker {self.robot_index}_{self.group_i} is quitting.")
                break
            t_start = time.perf_counter()
            
            try:
                message = self.comm_in.get(block=True, timeout=0.001)
                num_exploration_samples = message["num_exploration_samples"]
                num_exploration_trajectory_samples = message["num_exploration_trajectory_samples"]
                num_exploitation_samples_by_group = message["num_exploitation_samples_by_group"]
                sample_removal_pos_H_multiplier_by_group = message["sample_removal_pos_H_multiplier_by_group"]
                negative_similiarity_threshold_by_group = message["negative_similiarity_threshold_by_group"]
                positive_similiarity_threshold_by_group = message["positive_similiarity_threshold_by_group"]
                length_scale_by_group = message["length_scale_by_group"]
                alpha_by_group = message["alpha_by_group"]
                max_num_SV_by_group = message["max_num_SV_by_group"]
                sigmoid_alpha = message["sigmoid_alpha"]                
                exploitation_covariance = message["exploitation_covariance"]
                max_iter = message["max_iter"]
                point_cloud_size = message["point_cloud_size"]
                obstacle_radii = message["obstacle_radii"]
                if self.probabilistic:
                    obstacle_covariance = message["obstacle_covariance"]
                    prob_threshold = message["prob_threshold"]
            except Empty:
                pass
            # print(self.index, 'A', t_start - time.perf_counter())
            point_cloud = shared_array.copy()
            point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1)]
            self.update_point_cloud(torch.as_tensor(point_cloud,dtype=self.dtype, device=self.device))
            point_cloud_size = point_cloud.shape[0]
            if point_cloud_size == 0:
                print('No point cloud')
                continue
            obstacle_points = self.point_cloud_buffer[:point_cloud_size]
            obstacle_radii_exp = torch.tensor(obstacle_radii, dtype=self.dtype, device=self.device).expand(point_cloud_size, 1)
            if self.probabilistic:
                obstacle_covariance_exp = obstacle_covariance*torch.eye(3, dtype=self.dtype, device=self.device).expand(point_cloud_size, 3, 3)
            lcm.HandleSubscriptions(0)
            exploration_samples = uniform_distribution.sample((num_exploration_samples,))
            
            # print(self.index, 'A', t_start - time.perf_counter())
            if trajectory_points is not None:
                # print('hi')
                random_indices = torch.randint(0, trajectory_points.shape[0], (num_exploration_trajectory_samples,))
                # random_indices = torch.
                plan_exploration_samples = trajectory_points[random_indices]
                
                plan_exploration_samples += torch.randn_like(plan_exploration_samples)*exploitation_covariance
                plan_exploration_samples = torch.clip(plan_exploration_samples,lower_limits[:7],upper_limits[:7])
                plan_exploration_samples = torch.vstack([plan_exploration_samples,trajectory_points])
                plan_exploration_samples = torch.column_stack([plan_exploration_samples,torch.zeros(plan_exploration_samples.shape[0],2,dtype=self.dtype,device=self.device)])
                exploration_samples = torch.vstack([exploration_samples,plan_exploration_samples])
            self.X_exploration_buffer[: exploration_samples.shape[0]] = exploration_samples
            num_exploration_samples_actual = exploration_samples.shape[0]
            # print(self.index, 'B', t_start - time.perf_counter())
            groups_to_use = ['group_4','group_3']
            with torch.no_grad():
                for group_i, (group_name, group_worker) in enumerate(self.group_workers.items()):
                    if not group_name in groups_to_use:
                        continue
                    t_l = time.perf_counter()
                    exploitation_size = num_exploitation_samples_by_group[group_i]
                    # num_exploration_samples = num_exploration_samples_by_group[group_i]
                    max_num_SV = max_num_SV_by_group[group_i]

                    sample_removal_pos_H_multiplier = sample_removal_pos_H_multiplier_by_group[group_i]
                    alpha, length_scale = alpha_by_group[group_i], length_scale_by_group[group_i]
                    last_num_support_vectors = group_worker.num_support_vectors_buffer.item()
                    positive_similiarity_threshold = positive_similiarity_threshold_by_group[group_i] * 1/(1+np.exp(-sigmoid_alpha*(max_num_SV - last_num_support_vectors)/max_num_SV))
                    negative_similiarity_threshold = negative_similiarity_threshold_by_group[group_i] * 1/(1+np.exp(-sigmoid_alpha*(max_num_SV - last_num_support_vectors)/max_num_SV))

                    # print(negative_similiarity_threshold_by_group[group_i],negative_similiarity_threshold)
                    # print(positive_similiarity_threshold_by_group[group_i],positive_similiarity_threshold)
                    if last_num_support_vectors == 0:
                        exploitation_samples = uniform_distribution.sample((exploitation_size,)).to(
                            self.device
                        )
                    else:
                        last_support_vectors = group_worker.S_buffer[
                            :last_num_support_vectors
                        ]  # $#.clone()
                        last_weights = group_worker.W_buffer[:last_num_support_vectors]  # $#.clone()
                        random_indices = torch.randint(
                            0, last_num_support_vectors, (exploitation_size,)
                        )
                        exploitation_samples = last_support_vectors[random_indices].clone()
                        exploitation_samples += (
                            torch.randn_like(exploitation_samples) * exploitation_covariance
                        )
                        where_below = exploitation_samples < lower_limits
                        exploitation_samples = exploitation_samples[~torch.all(where_below, axis=-1)]
                        where_above = exploitation_samples > upper_limits
                        exploitation_samples = exploitation_samples[~torch.all(where_above, axis=-1)]

                        num_support_to_remove = last_num_support_vectors - max_num_SV
                        if num_support_to_remove > 0:
                            indices_to_remove = torch.randint(
                                0, last_num_support_vectors, (num_support_to_remove,)
                            )
                            mask = torch.ones(last_num_support_vectors, dtype=torch.bool)
                            mask[indices_to_remove] = False
                            last_support_vectors = last_support_vectors[mask]
                            group_worker.W_buffer[: last_support_vectors.shape[0]] = last_weights[mask]
                            group_worker.num_support_vectors = last_support_vectors.shape[0]
                            print(self.index,group_name,'removing support vectors',num_support_to_remove)
                        exploitation_samples = torch.vstack(
                            [last_support_vectors, exploitation_samples]
                        )

                    # print(self.index, group_name, 'A', t_l - time.perf_counter())
                    # print(self.index,group_name,'exploitation_samples.shape',exploitation_samples.shape)
                    # print(self.index,group_name,'exploration_samples.shape',exploration_samples.shape)
                    exploitation_samples = exploitation_samples[
                        : group_worker.X_exploitation_buffer.shape[0]
                    ]
                    group_worker.X_exploitation_buffer[: exploitation_samples.shape[0]] = (
                        exploitation_samples
                    )
                    # print(self.index, group_name, 'B', t_l - time.perf_counter())
                    # t0 = time.perf_counter()
                    # Y_exploitation = (in_collision_function(exploitation_samples,)*2-1  )[:,group_i:group_i+1]
                    all_samples = torch.vstack([exploitation_samples, exploration_samples])
                    if self.probabilistic:
                        Y_total = (
                            (probabilistic_collision_modules[group_name](
                                all_samples, obstacle_points, obstacle_radii_exp, obstacle_covariance_exp
                            ) > prob_threshold)
                            * 2.
                            - 1.
                        ).reshape(-1, 1)
                    else:
                        Y_total = (
                            is_group_in_collision(
                                group_name, all_samples, obstacle_points, obstacle_radii_exp
                            )
                            * 2.
                            - 1.
                        ).reshape(-1, 1)
                    # print(self.index,group_name,'Y_total.shape',Y_total.shape)
                    # Y_exploitation = (
                    #     is_group_in_collision(
                    #         group_name, exploitation_samples, obstacle_points, obstacle_radii_exp
                    #     )
                    #     * 2
                    #     - 1
                    # )
                    # print(self.index, group_name, 'C', t_l - time.perf_counter())

                    # Y_total = torch.vstack([Y_exploitation.view(-1, 1), Y_exploration.view(-1, 1)])


                    # fk_Xploitation, K_exploitation = kernel_matrix_function(exploitation_samples)
                    fk_X, K = kernel_modules[group_name]["rational_quadratic"](
                        all_samples,
                        alpha,
                        length_scale,
                    )

                    # make the elements of K ever so slightly more different
                    K *= 0.98
                    K = K.fill_diagonal_(1.)
                    group_worker.Y_by_group_buffer[: Y_total.shape[0]] = Y_total
                    group_worker.K_by_group_buffer[: K.shape[0], : K.shape[1]] = K
                    # torch.cuda.synchronize()
                    # print(self.index,group_name,'sending message')
                    num_exploitation_samples = exploitation_samples.shape[0]
                    # print(self.index, group_name, 'D', t_l - time.perf_counter())
                    message = {
                        "num_exploitation_samples": num_exploitation_samples,
                        "num_exploration_samples": num_exploration_samples_actual,
                        "group_i": group_i,
                        "positive_similiarity_threshold": positive_similiarity_threshold,
                        "negative_similiarity_threshold": negative_similiarity_threshold,
                        "sample_removal_pos_H_multiplier": sample_removal_pos_H_multiplier,
                        "max_iter": max_iter,
                    }
                    group_worker.comm_in.put(message)
                    # print(self.index, group_name, 'E', t_l - time.perf_counter())
                    del K,Y_total,fk_X,all_samples, exploitation_samples

                workers_done = []
                # while len(workers_done) < len(self.group_workers):

                while len(workers_done) < len(groups_to_use):
                    # all_queues_ok = True
                    for group_name, group_worker in self.group_workers.items():
                        if not group_name in groups_to_use:
                            continue
                        if group_name in workers_done or group_worker.comm_out.empty():
                            continue
                        result = group_worker.comm_out.get()
                        # if result is not None:
                        last_num_support_vectors = group_worker.num_support_vectors_buffer.item()
                        # print(self.index,'AAAAAAAAAA',group_name,last_num_support_vectors)
                        fk_S = fk_functions[group_name](group_worker.S_buffer[:last_num_support_vectors]).reshape(last_num_support_vectors,-1)
                        W_poly = calculate_polyharmonic_weights(fk_S, group_worker.Y_by_group_buffer[:last_num_support_vectors])
                        # torch.cuda.synchronize()
                        lcm_message = lcmt_support_vector()
                        lcm_message.num_support_vectors = last_num_support_vectors
                        lcm_message.support_vector_dimension = fk_S.shape[1]
                        lcm_message.support_vectors = fk_S.cpu().numpy()
                        lcm_message.weights = W_poly.cpu().numpy()
                        lcm_message.weights_dimension = W_poly.shape[1]
                        lcm_message.num_weights = W_poly.shape[0]
                        lcm.Publish(channel=f"support_vectors_robot_{self.index}_{group_name}", buffer = lcm_message.encode())
                        # print(self.index,'BBBBBBBBBB',group_name,last_num_support_vectors)
                        workers_done.append(group_name)
                        # print(f"Group {group_name} done")
                        # else:
                        #     all_queues_ok = False
                    # if all_queues_ok:
                    #     break
                    time.sleep(0.0001)
            self.time.value = time.perf_counter() - t_start
            # print(self.index,f"smv worker completed, time: {time.perf_counter()-t_start}s")
            

            
