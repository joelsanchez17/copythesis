import torch
from torch import nn
import traceback
from utils.pytorch_modules import DynamicalSystem, ContractionMetric
from utils.pytorch_modules.Controllers import BaseController
# Control Contraction Metrics: Convex and Intrinsic Criteria for Nonlinear Feedback Design
# eq 4, condition 1: M_dot +(A + BK)'M + M (A + BK) < −2λM
# eq 5, condition 1.1: δ_x' MB = 0 ⇒ δ_x′(  ̇ M + A′M + MA +2λM )δ_x < 0. (eq 4 => eq5) then the system is contracting
# eq 5 tells us that, if in the directions of the null space of (MB)', the system is naturally contracting, then the system is stabilizable

def condition_1(M,M_dot,A,B,K,lambda_):
    """
    eq 4, condition 1: M_dot +(A + BK)'M + M (A + BK) < −2λM
    """
    return M_dot + (A + B.matmul(K)).transpose(0,1).matmul(M) + M.matmul(A + B.matmul(K)) + 2 * lambda_ * M
def condition_1_1(M_dot,A,B,K,lambda_):
    """
    eq 5, condition 1.1: δ_x' MB = 0 ⇒ δ_x'(  M_dot + A'M + MA +2λM )δ_x < 0
    """
    return None


def condition_2_1(M,dM_dt,delta_f_M,df_x_dx,B,K,lambda_):
    """
    Stronger Conditions Giving Simpler Controllers (delta_u = K delta_x)):
    δx'(∂M/∂t + ∂fM+ (∂f/∂x)'M + M∂x/∂f+2λM)δx<0 
    """
    return dM_dt +  delta_f_M + df_x_dx.transpose(0,1).matmul(M) + M.matmul(df_x_dx) + 2 * lambda_ * M
def condition_2_2(M,dM_dt,delta_f_M,f_x_dx,B,K,lambda_):
    """
    # For each i=1,2,…,mi=1,2,…,m, the following equation must be satisfied:
    # ∂biM + ∂bi'M + M∂bi = 0
    #        ∂x​​       ∂x∂
    """
    return None

def condition_3_1(W,W_dot,df_x_dx,lambda_,B_annihilator):
    """
    eq 9: B_annihilator'(-W_dot + AW + WA' + 2λW) B_annihilator < 0
    """
    W_condition = -W_dot + df_x_dx.matmul(W) + W.matmul(df_x_dx.transpose(0,1)) + 2 * lambda_ * W
    W_condition = B_annihilator.transpose(0,1)@W_condition@B_annihilator
    # W_condition = self.dyn_sys.B_null_(x).transpose(0,1)@W_condition@self.dyn_sys.B_null_(x)

# Stronger Conditions Giving Simpler Controllers
# f δx≠0 satisfies δx'MB=0, then the following inequality must hold:
# δx'(∂M/∂t​+∂f​M+ (∂f/∂x​)'M + M∂x/∂f​+2λM)δx<0 
# Condition C2:
# For each i=1,2,…,mi=1,2,…,m, the following equation must be satisfied:
# ∂biM + ∂bi'M + M∂bi = 0
#        ∂x​​       ∂x∂
# C1 can be equivalently written as:
# eq 8: ∂t/∂M​+ ∂f​M + (∂f/∂x)'M + M ∂f/∂x ​− ρ M BB'M + 2λM < 0
# where ρ is a scalar multiplier.
# K(x,t) = -1/2 ρ B'M
# u = integral of that

# dual metric:
# eq 9: B_annihilator'(-W_dot + AW + WA' + 2λW)B_annihilator < 0
# W = M^-1
# One can search directly for differential feedback δu = K(x, u, t)δx by way of W and Y (x, u, t) ∈ R ^(mxn) satisfying 
# eq 10: -W_dot + ∂f/∂x W + W∂f/∂x' + BY + Y'B' + 2λW < 0
# Condition C1 can be written similarly, it is equivalent to the existence of a scalar function ρ(x, t) satisfying the inequality 
# eq 11: −∂W/∂t − ∂fW + ∂f/∂x W + W ∂f/∂x' − ρBB' + 2λW <0.
# K = -1/2 ρB'W^−1 
# Condition C2 also transforms to a linear constraint on W :
# ∂biW − ∂bi/∂x W − W ∂bi/∂x' = 0

class ControlContractionMetric(torch.nn.Module):
    def __init__(self,dyn_sys:DynamicalSystem,controller:BaseController,contraction_metric:ContractionMetric,lambda_ = 1):
        super().__init__()
        self.dyn_sys = dyn_sys
        self.controller = controller
        self.contraction_metric = contraction_metric
        self._lambda = lambda_  
        
        
    def calculate_metric(self,x):
        return self.contraction_metric(x)
    def calculate_controller(self,x,x_ref,u_ref):
        return self.controller(x,x_ref,u_ref)
    
    def calculate_common_terms(self,x,x_ref,u_ref):
        K, u = torch.func.jacrev(lambda y: (self.controller(y,x_ref,u_ref),)*2, has_aux=True)(x)
        K = K.view(-1,x.shape[0])
        f,B,x_dot = self.dyn_sys.dynamics(x,u).values()
        return {'f':f,'B':B,'K':K,'u':u,'x_dot':x_dot}
    
    def condition_1(self,M,M_dot,A,B,K,lambda_):
        """
        eq 4, condition 1: M_dot +(A + BK)'M + M (A + BK) < −2λM
        """
        return M_dot + (A + B.matmul(K)).transpose(0,1).matmul(M) + M.matmul(A + B.matmul(K)) + 2 * lambda_ * M
    def condition_1_1(self,M_dot,A,B,K,lambda_):
        """
        eq 5, condition 1.1: δ_x' MB = 0 ⇒ δ_x'(  M_dot + A'M + MA +2λM )δ_x < 0
        """
        return None


    def condition_2_1(self,M,dM_dt,delta_f_M,df_x_dx,lambda_):
        """
        Stronger Conditions Giving Simpler Controllers (delta_u = K delta_x)):
        δx'(∂M/∂t + ∂fM+ (∂f/∂x)'M + M∂x/∂f + 2λM) δx<0 
        """
        return dM_dt +  delta_f_M + df_x_dx.transpose(0,1).matmul(M) + M.matmul(df_x_dx) + 2 * lambda_ * M
    def condition_2_2(self,M,dM_dt,delta_f_M,f_x_dx,B,K,lambda_):
        """
        # For each i=1,2,…,mi=1,2,…,m, the following equation must be satisfied:
        # ∂biM + ∂bi'M + M∂bi = 0
        #        ∂x​​       ∂x∂
        """
        return None

    def condition_3_1(self,W,W_dot,df_x_dx,lambda_,B_annihilator):
        """
        eq 9: B_annihilator'(-W_dot + AW + WA' + 2λW) B_annihilator < 0
        """
        W_condition = -W_dot + df_x_dx.matmul(W) + W.matmul(df_x_dx.transpose(0,1)) + 2 * lambda_ * W
        W_condition = B_annihilator.transpose(0,1)@W_condition@B_annihilator
        return W_condition
        # W_condition = self.dyn_sys.B_null_(x).transpose(0,1)@W_condition@self.dyn_sys.B_null_(x)
    def contraction_condition(self,x,x_ref,u_ref):
        try:
            f,B,K,u,x_dot = self.calculate_common_terms(x,x_ref,u_ref).values()
            M,M_dot = self.contraction_metric.M_M_dot(x,u).values()
            W,W_dot = self.contraction_metric.W_W_dot(x,u).values()
            A,df_dx = self.dyn_sys.differential_dynamics(x,u).values()
            W_condition = self.condition_3_1(W,W_dot,df_dx,self._lambda,self.dyn_sys.B_null_(x))
            contraction_condition = self.condition_1(M,M_dot,A,B,K,self._lambda)
            
        except Exception as e:
            traceback.print_exc()
            raise Exception("Error in calculating contraction condition")
        return W_condition,contraction_condition, f,B,K,u,x_dot,df_dx,M,M_dot
    def forward(self,x,x_ref,u_ref):
        return self.controller(x,x_ref,u_ref)