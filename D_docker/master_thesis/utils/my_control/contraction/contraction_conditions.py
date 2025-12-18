
# contraction_condition = M_dot + (A + B@K).transpose(0,1)@M + M@(A + B@K) + 2 * self._lambda * M
def weak_contraction_condition(A, B, K, M, M_dot, lambda_, **kwargs):
    MABK = M@(A + B@K)
    result = M_dot + MABK.T + MABK + 2 * lambda_ * M
    return result
def weak_naturally_contracting_condition(A, W, W_dot, B_null, lambda_, **kwargs):
    # delta_x.T @ B @ M = 0 =>  directional_f_M + MA.T + MA + 2 * lambda_ * M < 0
    raise NotImplementedError()
    
def weak_inverse_metric_contraction_condition(A, B, K, W, W_dot, lambda_, **kwargs):
    ABKW = (A+B@K)@W
    result = (-W_dot + ABKW.T + ABKW + 2 * lambda_ * W)
    return result
def weak_inverse_metric_naturally_contracting_condition(A, W, W_dot, B_null, lambda_, **kwargs):
    AW = A@W
    result = B_null.T@(-W_dot + AW.T + AW  + 2 * lambda_ * W) @ B_null
    return result
# this being true implies C_1, but lets us find the controller at the same time

def simpler_controllers_contraction_conditions(partial_f_partial_x,B,K,M, directional_f_M, directional_B_M, partial_B_partial_x, lambda_, **kwargs):
    MdfdxBK = M@(partial_f_partial_x + B@K)
    C1 = (MdfdxBK + MdfdxBK.T + directional_f_M + 2 * lambda_ * M)
    
    C2s = []
    for i in range(0,B.shape[1]):
        b_i = B[:,i]
        partial_b_i_partial_x = partial_B_partial_x[:,i,:]
        directional_b_i_M = directional_B_M[:,:,i]
        Mdbdx = M@partial_b_i_partial_x
        C2_condition_i = (directional_b_i_M + Mdbdx + Mdbdx.T) # == 0 
        C2s.append(C2_condition_i)
    return C1,C2s

def inverse_metric_simpler_controllers_contraction_conditions(partial_f_x_partial_x,B,K,W, directional_f_W, directional_B_W, partial_B_partial_x,B_null, lambda_, **kwargs):
    """
    Returns 
    ```
    [
    -directional_f_W + (partial_f_x_partial_x + B@K)@W + (partial_f_x_partial_x + B@K).T@W + 2 * lambda_ * W,
    
    B_null@(-directional_f_W + partial_f_x_partial_x@W + (partial_f_x_partial_x@W).T + 2 * lambda_ * W)@B_null.T,
    
    partial_b_i_W - (partial_b_i_partial_x@W) - (partial_b_i_partial_x@W).T for b_i in columns of B
    ]
    ```
    """
    dfdxW = partial_f_x_partial_x@W
    C1_2 = -directional_f_W + dfdxW + dfdxW.T + 2 * lambda_ * W
    BKW = B@K@W
    # C1_1 = (-W_dot + ABKW.T + ABKW + 2 * lambda_ * W) but with C2 holding.
    C1_1 = C1_2 + BKW + BKW.T 
    # C1_2 = C1_1 for non actuable directions
    C1_2 = B_null@C1_2@B_null.T
    
    C2s = []
    for i in range(0,B.shape[1]):
        b_i = B[:,i]
        partial_b_i_partial_x = partial_B_partial_x[:,i,:]
        partial_b_i_W = directional_B_W[:,:,i]
        temp = partial_b_i_partial_x@W
        C2_condition_i = (partial_b_i_W - temp - temp.T)
        C2s.append(C2_condition_i)
    return C1_1,C1_2,C2s
