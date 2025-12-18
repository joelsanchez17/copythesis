

from functools import partial
from toolz import memoize, pipe, accumulate,groupby, compose,compose_left, merge
from collections import namedtuple
import pathlib,os
import typing as T
import numpy as np

import symforce
symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")
import sympy as sp
from sympy.codegen import CodeBlock,Assignment
import symforce.symbolic as sf
import casadi as ca
from projects.refactor_mpb.symforce_casadi.casadi_config import CasadiConfig
from symforce.codegen.backends.pytorch import PyTorchConfig, PyTorchCodePrinter
from symforce.ops.storage_ops import StorageOps



from utils.my_sympy.misc import *
from utils.my_sympy.conversions import *
from utils.misc import *
from projects.refactor_mpb.sympy_wrapper import SympyWrapper
        
class DynamicalSystemWrapper(SympyWrapper):
    
    def __init__(self, state_variables:list[sp.Symbol],actuation_variables:list[sp.Symbol], temp_folder = None):
        super().__init__(temp_folder=temp_folder)
        self.state_variables = state_variables
        self.actuation_variables = actuation_variables
        self._state_lower_bounds = [-np.inf] * len(self.state_variables)
        self._state_upper_bounds = [np.inf] * len(self.state_variables)
        self._actuation_lower_bounds = [-np.inf] * len(self.actuation_variables)
        self._actuation_upper_bounds = [np.inf] * len(self.actuation_variables)
    @property
    def state_lower_bounds(self):
        return self.STATE_NAMEDVIEW(*self._state_lower_bounds)
    @property
    def state_upper_bounds(self):
        return self.STATE_NAMEDVIEW(*self._state_upper_bounds)
    @property
    def actuation_lower_bounds(self):
        return self.ACTUATION_NAMEDVIEW(*self._actuation_lower_bounds)
    @property
    def actuation_upper_bounds(self):
        return self.ACTUATION_NAMEDVIEW(*self._actuation_upper_bounds)
    @property
    def input_lower_bounds(self):
        return self.actuation_lower_bounds
    @property
    def input_upper_bounds(self):
        return self.actuation_upper_bounds

    @property
    def STATE_NAMEDVIEW(self):
        return namedtuple('State', self.STATE_NAMES)
    @property
    def ACTUATION_NAMEDVIEW(self):
        return namedtuple('Actuation', self.ACTUATION_NAMES)
    @property
    def INPUT_NAMEDVIEW(self):
        return self.ACTUATION_NAMEDVIEW
    

    @property
    def STATE_NAMES(self):
        return [str(state) for state in self.state_variables]
    @property
    def ACTUATION_NAMES(self):
        return [str(actuation) for actuation in self.actuation_variables]
    @property
    def INPUT_NAMES(self):
        return self.ACTUATION_NAMES
    def num_states(self):
        return len(self.state_variables)
    def num_inputs(self):
        return len(self.actuation_variables)
    
    def get_time_derivative_function(self, module = 'casadi', **kwargs):
        raise NotImplementedError("This function must be implemented")
    @memoize
    def get_f_x_u_sympy(self, **kwargs):
        return self.get_time_derivative_function(module = 'sympy', **kwargs)
    @memoize
    def get_f_x_u_function(self, module = 'casadi', **kwargs):
        return self.get_time_derivative_function(module = module, **kwargs)
    @property 
    def f_x_u_pytorch(self):
        return self.get_f_x_u_function(module = 'pytorch', )
    @memoize    
    def get_B_sympy(self, **kwargs):
        x_dot = self.get_time_derivative_function(module = 'sympy', **kwargs)
        return x_dot.jacobian(self.actuation_variables).reshape(self.num_states(),self.num_inputs())
    @memoize
    def get_B_function(self, module = 'casadi', **kwargs):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"      
        function_name = 'B'
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')
        except:
            sympy_expression = self.get_B_sympy(**kwargs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.state_variables),]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    @property 
    def B_pytorch(self):
        return self.get_B_function(module = 'pytorch', )
    
    @memoize
    def get_f_x_sympy(self, **kwargs):
        x_dot = self.get_time_derivative_function(module = 'sympy', **kwargs)
        return x_dot.subs({k:0 for k in self.actuation_variables})
    @memoize
    def get_f_x_function(self, module = 'casadi', **kwargs):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"      
        function_name = 'f_x'
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')
        except:
            sympy_expression = self.get_f_x_sympy(**kwargs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.state_variables),]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)

    @property
    def f_x_pytorch(self):
        return self.get_f_x_function(module = 'pytorch', )
    @memoize
    def get_partial_f_x_partial_x_sympy(self, **kwargs):
        x_dot = self.get_f_x_function(module = 'sympy', **kwargs)
        return x_dot.jacobian(self.state_variables)
    @memoize
    def get_partial_f_x_partial_x_function(self, module = 'casadi', **kwargs):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"      
        function_name = 'partial_f_x_partial_x'
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')
        except:
            sympy_expression = self.get_partial_f_x_partial_x_sympy( **kwargs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.state_variables),]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    @property 
    def partial_f_x_partial_x_pytorch(self):
        return self.get_partial_f_x_partial_x_function(module = 'pytorch', )
    @memoize
    def partial_f_x_u_partial_x_sympy(self, **kwargs):
        x_dot = self.get_time_derivative_function(module = 'sympy', **kwargs)
        return x_dot.jacobian(self.state_variables)
    @memoize
    def get_partial_f_x_u_partial_x_function(self, module = 'casadi', **kwargs):
        assert module in ['casadi','pytorch','sympy','numpy'] , "module must be one of ['casadi','pytorch','sympy','numpy']"      
        function_name = 'partial_f_x_u_partial_x'
        path = self.TEMP_FOLDER / "dynamic_functions"
        path.mkdir(exist_ok=True,)
        try:
            return self.get_function_or_throw(function_name, path = path, module = module)
        except:
            pass
        try:
            sympy_expression = self.get_function_or_throw(function_name,path = path, module = 'sympy')
        except:
            sympy_expression = self.partial_f_x_u_partial_x_sympy(**kwargs)
            self.make_function_from_expression(function_name, sympy_expression, None, path = path, module = 'sympy') #saves csed expression
        inputs = [sp.Matrix(self.state_variables),sp.Matrix(self.actuation_variables)]
        return self.make_function_from_expression(function_name, sympy_expression, inputs, path = path, module = module)
    @property 
    def partial_f_x_u_partial_x_pytorch(self):
        return self.get_partial_f_x_u_partial_x_function(module = 'pytorch', )
    def get_latex_code_affine_system(self,symbol_to_latex_name = {}):
        f_x = self.get_f_x_function(module='sympy')
        f_x_u = self.get_f_x_u_function(module='sympy')
        B = self.get_B_function(module='sympy')
        partial_f_x_partial_x = self.get_partial_f_x_partial_x_function(module='sympy')
        partial_f_x_u_partial_x = self.get_partial_f_x_u_partial_x_function(module='sympy')
        latex_code = f"""
\\begin{{align*}}
\\\\~\\\\
\\\\~\\\\
& x = {sp.latex(self.state_variables, symbol_names=symbol_to_latex_name)}\\\\~\\\\
& u = {sp.latex(self.actuation_variables, symbol_names=symbol_to_latex_name)}\\\\~\\\\
& \dot{{x}}(x,u) =  {sp.latex(f_x_u.reshape(len(f_x),1), symbol_names=symbol_to_latex_name)} \\\\~\\\\
& f(x) =  {sp.latex(f_x.reshape(len(f_x),1), symbol_names=symbol_to_latex_name)} \\\\~\\\\
& B(x) =  {sp.latex(B, symbol_names=symbol_to_latex_name)} \\\\~\\\\
&\\frac{{\\partial{{f}}}}{{\\partial{{x}}}} = {sp.latex(partial_f_x_partial_x, symbol_names=symbol_to_latex_name)} \\\\~\\\\
&\\frac{{\\partial{{\dot{{x}}}}}}{{\\partial{{x}}}} = A(x,u) = {sp.latex(partial_f_x_u_partial_x, symbol_names=symbol_to_latex_name)} \\\\~\\\\
"""
        
        latex_code += "\\end{align*}"
        return latex_code