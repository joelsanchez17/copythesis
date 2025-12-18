

from functools import partial
from toolz import memoize, pipe, accumulate,groupby, compose,compose_left, merge
from collections import namedtuple
import pathlib,os
import typing as T


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
import time


from utils.my_sympy.misc import *
from utils.my_sympy.conversions import *
from utils.misc import *

def replace_constants_pytorch(path):
    # TODO: doesn't catch numbers like 0.9412142e-2
    """
    Quick hax to replace constant tensors in a pytorch file with variables.
    """
    import re
    with open(path, 'r') as file:
        content = file.read()

    # Regular expression to match constant tensor declarations including fractions
    pattern = r"torch\.tensor\(([\d\.-]+|[\d\.-]+\ / [\d\.-]+), \*\*tensor_kwargs\)"
    matches = re.findall(pattern, content)

    # Evaluate the fractions and convert to float, create a map of unique constants to new variable names
    constant_to_variable = {}
    unique_constants = {}
    for match in set(matches):
        try:
            constant = float(eval(match))
            
            var_name = f"cte_{len(unique_constants) + 1}"
            if constant not in unique_constants:
                unique_constants[constant] = var_name
            constant_to_variable[match] = constant
        except Exception as e:
            print(f"Error evaluating match '{match}': {e}")
            continue

    # Replace the occurrences of constant tensor declarations with new variable names
    for match, constant in constant_to_variable.items():
        # constant_str = str(constant) if '/' not in str(constant) else f'({str(constant)})'
        var_name = unique_constants[constant]
        content = re.sub(
            pattern.replace(r'([\d\.-]+|[\d\.-]+\ / [\d\.-]+)', (match)),
            var_name,
            content
        )
    new_declarations = '\n'.join(f"    {var_name} = torch.tensor({constant}, **tensor_kwargs)" for constant, var_name in unique_constants.items())
    content = content.replace('# Input arrays','# Input arrays\n\n'+new_declarations)
    with open(path, 'w') as file:
        file.write(content)
        


def sympy_to_symforce(var,):
    # symengine.sympify might just do this better?
    # functions on matrices is fucked sp.Function('whatever')(sp.Matrix(...)) won't work
    if isinstance(var,(sp.Symbol,sp.Number)):
        return StorageOps.from_storage(sf.Symbol,[var])
    elif isinstance(var,(sp.MatrixSymbol)):
        # https://github.com/symforce-org/symforce/blob/42c4df720f85d212380ba2d4c57b6dbb3f5fbe76/symforce/databuffer.py#L14 ?
        raise ValueError("MatrixSymbols don't work with sympy->symengine conversion")
    elif isinstance(var,(sp.IndexedBase)):
        raise ValueError("IndexedBase don't work with sympy->symengine conversion")
    elif isinstance(var,(sp.Matrix,)):
        return sf.Matrix(var)
    else:
        try:
            return StorageOps.from_storage(sf.Expr,[var])
        except:
            print("Tried to convert",var, " with StorageOps.from_storage(sf.Expr,[var])")
            raise ValueError(f"Type {type(var)} not implemented in sympy_var_to_symforce")
        
def cse_then_print(expr,path):
    expr_cse = sp.cse(expr,order = 'none')
    temp_terms = [Assignment(*term) for term in (expr_cse)[0]]
    preamble = "from sympy import *"
    vars = sp.python(expr.free_symbols).split("\n")[:-1]
    intermediate_results = sp.pycode(CodeBlock(*temp_terms),fully_qualified_modules=False)
    output = sp.python(expr_cse[1][0]).split("\n")[-1]

    text = "\n".join([preamble] + vars + [intermediate_results] + [output])
    with open(path,'w') as f:
        f.write(text)
        
        
class SympyWrapper:
    def __init__(self, temp_folder = None):
        if temp_folder is None:
            self.TEMP_FOLDER = pathlib.Path(os.getcwd()) / 'temp'
        else:
            self.TEMP_FOLDER = pathlib.Path(temp_folder)
        self.TEMP_FOLDER.mkdir(exist_ok=True,parents=True)


    
    def get_function_or_throw(self, function_name, path: pathlib.Path = None, module = 'casadi'):
        if path is None: 
            path = self.TEMP_FOLDER
            
        path = path / module
        if not (path / f"{function_name}.py").exists():
            raise ValueError(f"Function {function_name} not found in {path}")
        modul = symforce.codegen.codegen_util.load_generated_package(function_name, path / f"{function_name}.py")
        if module == 'sympy':
            sympy_expr = modul.e
            return sympy_expr
        function = getattr(modul, function_name)
        return function
    def make_function_from_expression(self, function_name, expression, inputs, path: pathlib.Path = None, module = 'casadi'):
        if path is None: 
            path = self.TEMP_FOLDER
            
        path = path / module
        path.mkdir(exist_ok=True,parents=True)
        
        if module == 'sympy':
            cse_then_print(expression,str(path / f"{function_name}.py"))
            return expression
        if module == 'casadi':
            return self.sympy_to_casadi(function_name, expression, inputs, path)
        if module == 'pytorch':
            return self.sympy_to_pytorch(function_name, expression, inputs, path)
    def sympy_to_module(self, function_name, sympy_expr, inputs, module, path: pathlib.Path = None):
        if path is None: path = self.TEMP_FOLDER
        assert isinstance(inputs, (list,tuple)), "inputs must be a list or tuple of variables"
        assert module in ['casadi','pytorch'] , "module must be one of ['casadi','pytorch']"
        config = CasadiConfig() if module == 'casadi' else PyTorchConfig()
        
        codegen_input = symforce.values.Values()
        for i,var in enumerate(inputs):
            sf_var = sympy_to_symforce(var)
            codegen_input[f"_in{i}"] = sf_var
            
        sf_expr = sympy_to_symforce(sympy_expr)
            
        codegen_output = symforce.values.Values(out = sf_expr)
        
        codegen_obj = symforce.codegen.Codegen(
            inputs=codegen_input,
            outputs=codegen_output,
            config=config,
            name=function_name,
            return_key="out",
        )
        
        generated_data = codegen_obj.generate_function(str(path), skip_directory_nesting=True)
        if module == 'pytorch':
            replace_constants_pytorch(str(generated_data.function_dir / f"{function_name}.py"))
        gen_module = symforce.codegen.codegen_util.load_generated_package(
            function_name, generated_data.function_dir
        )
        time.sleep(0.2)
        function = getattr(gen_module, function_name)
        return function
    def sympy_to_pytorch(self, function_name, sympy_expr, inputs, path: pathlib.Path = None):
        return self.sympy_to_module(function_name, sympy_expr, inputs, 'pytorch', path)
    def sympy_to_casadi(self, function_name, sympy_expr, inputs, path: pathlib.Path = None):
        return self.sympy_to_module(function_name, sympy_expr, inputs, 'casadi', path)
    