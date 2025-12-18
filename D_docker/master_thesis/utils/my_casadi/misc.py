from collections.abc import MutableMapping
from enum import Enum
import casadi as ca
import numpy as np
from toolz import pipe
import pathlib, os, string, random, tempfile, shutil, atexit,subprocess

def delete_casadi_function(ob,locals):
    
    try:
        idx = list(locals.values()).index(ob)
        key = list(locals.keys())[idx]
        import gc
        gc.collect(2)
        ca._casadi.delete_Function(ob)
        del locals[key]
        gc.collect(2)
    except:
        pass
    return None

def fix_code(path):
    import re
    # Define the pattern to find and the replacement pattern
    pattern = re.compile(r'if \((mid = (.*?))flag\) return 1;', re.MULTILINE | re.DOTALL)
    # pattern = re.compile(r'if \((mid(.*?))flag\) return 1;', re.MULTILINE | re.DOTALL)
    replacement = r'\1if (flag) return 1;'

    # Read the file
    with open(path, 'r') as file:
        file_contents = file.read()
    
    # Apply the correction
    corrected_contents = re.sub(pattern, replacement, file_contents)
    with open(path, 'w') as file:
        file.write(corrected_contents)
        
def vertcat(*args):
    try:
        if len(args)==0:
            return ca.DM(0,1)
    except:
        return ca.vertcat(args)
    if len(args) == 1:
        if isinstance(args[0],(list,tuple,np.ndarray)):
            return ca.vertcat(*args[0])
        else:
            return ca.vertcat(args[0])
    else:
        return ca.vertcat(*args)
    
def veccat(*args):
    try:
        if len(args)==0:
            return ca.DM(0,1)
    except:
        return ca.veccat(args)
    if len(args) == 1:
        if isinstance(args[0],(list,tuple,np.ndarray)):
            return ca.veccat(*args[0])
        else:
            return ca.veccat(args[0])
    else:
        return ca.veccat(*args)
        
def horzcat(*args):
    try:
        if len(args)==0:
            return ca.DM(0,1)
    except:
        return ca.horzcat(args)
    if len(args) == 1:
        if isinstance(args[0],(list,tuple,np.ndarray)):
            return ca.horzcat(*args[0])
        else:
            return ca.horzcat(args[0])
    else:
        return ca.horzcat(*args)
    
def merge_variable_bounds(variables,parameters,g,lbg,ubg):
    """
    
    variables: list of variables
    parameters: list of parameters
    constraints: list of constraints
    
    Returns lbx and ubx such that it is the most restrictive of all the given bounds.
    
    ```
    x = ca.SX.sym("x",5)
    y = ca.SX.sym("y",2)
    p = ca.SX.sym("p",1)
    (merge_variable_bounds(ca.veccat(x,y),p,ca.vvcat([x,x,y]),ca.vvcat([x*0+p,x*0+5,y*0 + p]),ca.vvcat([x*0+30,x*0+50, y*0 + ca.inf])))
    
    (SX(@1=fmax(p,5), [@1, @1, @1, @1, @1, p, p]),
    SX(@1=30, @2=inf, [@1, @1, @1, @1, @1, @2, @2]))
    ```  
    """
    variables = veccat(variables)
    parameters = veccat(parameters)
    g = veccat(g)
    lbg = veccat(lbg)
    ubg = veccat(ubg)
    try:
        [non_simple_g_indices,lbx,ubx,lam_f,lam_b] = ca.detect_simple_bounds(variables,parameters,g,lbg,ubg)
    except RuntimeError as e:
        if """Assertion "n_groups>=1" failed:""" in str(e):
            print("No simple bounds detected so something is lbg/ubg not lbx/ubx.")
            raise e
        else:
            raise e
    assert len(non_simple_g_indices) == 0 , f"Indices {non_simple_g_indices} of the constraints are not simple bounds on the variables. Constraints[{non_simple_g_indices}] = {g[non_simple_g_indices].str()}"
    return ca.cse(lbx),ca.cse(ubx)
# try:
#     [gi,lbx,ubx,lam_f,lam_b] = ca.detect_simple_bounds(x,3,g,lbg,ubg)
# except RuntimeError as e:
#     if """Assertion "n_groups>=1" failed:""" in str(e):
#         pass
#     else:
#         raise e
def hessian_chain_rule(inner_function,outer_function,triu = True):
    # derived wrt. to first input of the functions
    # outer_value = ca.sum1(outer_function(inner_value_symbol,parameters))
    diff_wrt_inner_index = 0
    diff_wrt_outer_index = 0
    diff_output_inner_index = 0
    diff_output_outer_index = 0
    
    outer_in = outer_function.mx_in()
    inner_in = inner_function.mx_in()
    if not isinstance(outer_in,(list,tuple)):
        outer_in = [outer_in]
    if not isinstance(inner_in,(list,tuple)):
        inner_in = [inner_in]
    
    inner_value = inner_function(*inner_in)
    outer_value = outer_function(*outer_in)
    
    if inner_function.n_out() == 1:
        inner_value = [inner_value]
    if outer_function.n_out() > 1:
        outer_value = outer_value[diff_output_outer_index]
        
    jacobian_inner_value = ca.jacobian(ca.vertcat(*inner_value),inner_in[diff_wrt_inner_index])
    gradient_outer_value = ca.Function('gr_outer',outer_in,[ca.gradient(outer_value,outer_in[diff_wrt_outer_index])])(*outer_in[:diff_wrt_outer_index], inner_value[diff_wrt_inner_index], *outer_in[diff_wrt_outer_index+1:])
    hessian_outer_value = ca.Function('he_outer',outer_in,[ca.hessian(outer_value,outer_in[0])[diff_wrt_outer_index]])(*outer_in[:diff_wrt_outer_index], inner_value[diff_wrt_inner_index], *outer_in[diff_wrt_outer_index+1:])
    
    hessians_inner = [ca.hessian(inner_value[diff_output_inner_index][i],inner_in[diff_wrt_inner_index])[0] for i in range(0,inner_function.numel_out(diff_output_inner_index))]
    hessian_chain = jacobian_inner_value.T@hessian_outer_value@jacobian_inner_value
    for i in range(0,inner_function.numel_out(diff_output_inner_index)):
        hessian_chain+= gradient_outer_value[i].T@hessians_inner[i]
    if triu:
        hessian_chain = ca.triu(hessian_chain)
    hessian_final = ca.Function('hessian_chain',[*inner_in, *outer_in[:diff_wrt_outer_index], *outer_in[diff_wrt_outer_index+1:]],[ca.cse(hessian_chain)],)
    return hessian_final

class CasadiDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        for k,v in self.store.items():
            if ca.is_equal(k,key):
                return self.store[k]
        raise KeyError(key)
        # return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        for k,v in self.store.items():
            if ca.is_equal(k,key):
                self.store[k] = value
                return
        self.store[key] = value
        
    def __copy__(self):
        new = CasadiDict()
        for k,v in self.store.items():
            new[k] = v
        return new
    def __delitem__(self, key):
        for k,v in self.store.items():
            if ca.is_equal(k,key):
                break
        del self.store[k]

    def __iter__(self):
        return iter(self.store)
    
    def __repr__(self):
        return self.store.__repr__()
    def __len__(self):
        return len(self.store)
    
homegeneous_transform_sp = ca.Sparsity(4,4)
for i in range(3):
    for j in range(3):
        homegeneous_transform_sp.add_nz(i,j)
    homegeneous_transform_sp.add_nz(i,3)
homegeneous_transform_sp.add_nz(3,3)
casadi_3x3_inverse = pipe(ca.SX.sym('in',3,3),
                          lambda x: ca.Function('inverse3x3',[x],[ca.inv(x)]))
casadi_4x4_inverse = pipe(ca.SX.sym('in',homegeneous_transform_sp),
                          lambda x: ca.Function('inverse4x4',[x],[ca.inv(x)]),)
T_1 = ca.SX.sym('in',homegeneous_transform_sp)
T_inv = ca.vertcat(ca.horzcat(T_1[0:3,0:3].T,-T_1[0:3,0:3].T@T_1[0:3,3]),
                   ca.horzcat(0,0,0,1))
casadi_inverse_homogeneous_transform = ca.Function('inverse4x4',[T_1],[T_inv],['T'],['T_inv'])

T_2 = ca.SX.sym('in',homegeneous_transform_sp)
T_2_in_T_1 = casadi_inverse_homogeneous_transform(T_1)@T_2
casadi_frame_A_in_frame_B = ca.Function('frame_1_in_frame_2',[T_1,T_2],
                                        [T_2_in_T_1,],
                                        ['T_A','T_B'],
                                        ['T_A_in_T_B'],)
_casadi = ca._casadi
class Casadi_Enum(Enum):
    OPTI_GENERIC_EQUALITY = _casadi.OPTI_GENERIC_EQUALITY
    OPTI_GENERIC_INEQUALITY = _casadi.OPTI_GENERIC_INEQUALITY
    OPTI_EQUALITY = _casadi.OPTI_EQUALITY
    OPTI_INEQUALITY = _casadi.OPTI_INEQUALITY
    OPTI_DOUBLE_INEQUALITY = _casadi.OPTI_DOUBLE_INEQUALITY
    OPTI_PSD = _casadi.OPTI_PSD
    OPTI_UNKNOWN = _casadi.OPTI_UNKNOWN
    OPTI_VAR = _casadi.OPTI_VAR
    OPTI_PAR = _casadi.OPTI_PAR
    OPTI_DUAL_G = _casadi.OPTI_DUAL_G
    OP_ASSIGN = _casadi.OP_ASSIGN
    OP_ADD = _casadi.OP_ADD
    OP_SUB = _casadi.OP_SUB
    OP_MUL = _casadi.OP_MUL
    OP_DIV = _casadi.OP_DIV
    OP_NEG = _casadi.OP_NEG
    OP_EXP = _casadi.OP_EXP
    OP_LOG = _casadi.OP_LOG
    OP_POW = _casadi.OP_POW
    OP_CONSTPOW = _casadi.OP_CONSTPOW
    OP_SQRT = _casadi.OP_SQRT
    OP_SQ = _casadi.OP_SQ
    OP_TWICE = _casadi.OP_TWICE
    OP_SIN = _casadi.OP_SIN
    OP_COS = _casadi.OP_COS
    OP_TAN = _casadi.OP_TAN
    OP_ASIN = _casadi.OP_ASIN
    OP_ACOS = _casadi.OP_ACOS
    OP_ATAN = _casadi.OP_ATAN
    OP_LT = _casadi.OP_LT
    OP_LE = _casadi.OP_LE
    OP_EQ = _casadi.OP_EQ
    OP_NE = _casadi.OP_NE
    OP_NOT = _casadi.OP_NOT
    OP_AND = _casadi.OP_AND
    OP_OR = _casadi.OP_OR
    OP_FLOOR = _casadi.OP_FLOOR
    OP_CEIL = _casadi.OP_CEIL
    OP_FMOD = _casadi.OP_FMOD
    OP_FABS = _casadi.OP_FABS
    OP_SIGN = _casadi.OP_SIGN
    OP_COPYSIGN = _casadi.OP_COPYSIGN
    OP_IF_ELSE_ZERO = _casadi.OP_IF_ELSE_ZERO
    OP_ERF = _casadi.OP_ERF
    OP_FMIN = _casadi.OP_FMIN
    OP_FMAX = _casadi.OP_FMAX
    OP_INV = _casadi.OP_INV
    OP_SINH = _casadi.OP_SINH
    OP_COSH = _casadi.OP_COSH
    OP_TANH = _casadi.OP_TANH
    OP_ASINH = _casadi.OP_ASINH
    OP_ACOSH = _casadi.OP_ACOSH
    OP_ATANH = _casadi.OP_ATANH
    OP_ATAN2 = _casadi.OP_ATAN2
    OP_CONST = _casadi.OP_CONST
    OP_INPUT = _casadi.OP_INPUT
    OP_OUTPUT = _casadi.OP_OUTPUT
    OP_PARAMETER = _casadi.OP_PARAMETER
    OP_CALL = _casadi.OP_CALL
    OP_FIND = _casadi.OP_FIND
    OP_LOW = _casadi.OP_LOW
    OP_MAP = _casadi.OP_MAP
    OP_MTIMES = _casadi.OP_MTIMES
    OP_SOLVE = _casadi.OP_SOLVE
    OP_TRANSPOSE = _casadi.OP_TRANSPOSE
    OP_DETERMINANT = _casadi.OP_DETERMINANT
    OP_INVERSE = _casadi.OP_INVERSE
    OP_DOT = _casadi.OP_DOT
    OP_BILIN = _casadi.OP_BILIN
    OP_RANK1 = _casadi.OP_RANK1
    OP_HORZCAT = _casadi.OP_HORZCAT
    OP_VERTCAT = _casadi.OP_VERTCAT
    OP_DIAGCAT = _casadi.OP_DIAGCAT
    OP_HORZSPLIT = _casadi.OP_HORZSPLIT
    OP_VERTSPLIT = _casadi.OP_VERTSPLIT
    OP_DIAGSPLIT = _casadi.OP_DIAGSPLIT
    OP_RESHAPE = _casadi.OP_RESHAPE
    OP_SUBREF = _casadi.OP_SUBREF
    OP_SUBASSIGN = _casadi.OP_SUBASSIGN
    OP_GETNONZEROS = _casadi.OP_GETNONZEROS
    OP_GETNONZEROS_PARAM = _casadi.OP_GETNONZEROS_PARAM
    OP_ADDNONZEROS = _casadi.OP_ADDNONZEROS
    OP_ADDNONZEROS_PARAM = _casadi.OP_ADDNONZEROS_PARAM
    OP_SETNONZEROS = _casadi.OP_SETNONZEROS
    OP_SETNONZEROS_PARAM = _casadi.OP_SETNONZEROS_PARAM
    OP_PROJECT = _casadi.OP_PROJECT
    OP_ASSERTION = _casadi.OP_ASSERTION
    OP_MONITOR = _casadi.OP_MONITOR
    OP_NORM2 = _casadi.OP_NORM2
    OP_NORM1 = _casadi.OP_NORM1
    OP_NORMINF = _casadi.OP_NORMINF
    OP_NORMF = _casadi.OP_NORMF
    OP_MMIN = _casadi.OP_MMIN
    OP_MMAX = _casadi.OP_MMAX
    OP_HORZREPMAT = _casadi.OP_HORZREPMAT
    OP_HORZREPSUM = _casadi.OP_HORZREPSUM
    OP_ERFINV = _casadi.OP_ERFINV
    OP_PRINTME = _casadi.OP_PRINTME
    OP_LIFT = _casadi.OP_LIFT
    OP_EINSTEIN = _casadi.OP_EINSTEIN
    OP_BSPLINE = _casadi.OP_BSPLINE
    OP_CONVEXIFY = _casadi.OP_CONVEXIFY
    OP_SPARSITY_CAST = _casadi.OP_SPARSITY_CAST
    OP_LOG1P = _casadi.OP_LOG1P
    OP_EXPM1 = _casadi.OP_EXPM1
    OP_HYPOT = _casadi.OP_HYPOT
    OP_LOGSUMEXP = _casadi.OP_LOGSUMEXP
    OP_REMAINDER = _casadi.OP_REMAINDER
    CASADI_INT_TYPE_STR = _casadi.CASADI_INT_TYPE_STR
    SOLVER_RET_SUCCESS = _casadi.SOLVER_RET_SUCCESS
    SOLVER_RET_UNKNOWN = _casadi.SOLVER_RET_UNKNOWN
    SOLVER_RET_LIMITED = _casadi.SOLVER_RET_LIMITED
    SOLVER_RET_NAN = _casadi.SOLVER_RET_NAN
    SOLVER_RET_INFEASIBLE = _casadi.SOLVER_RET_INFEASIBLE
    IS_GLOBAL = _casadi.IS_GLOBAL
    IS_MEMBER = _casadi.IS_MEMBER
    IS_SPARSITY = _casadi.IS_SPARSITY
    IS_DMATRIX = _casadi.IS_DMATRIX
    IS_IMATRIX = _casadi.IS_IMATRIX
    IS_SX = _casadi.IS_SX
    IS_MX = _casadi.IS_MX
    IS_DOUBLE = _casadi.IS_DOUBLE
    L_INT = _casadi.L_INT
    L_BOOL = _casadi.L_BOOL
    LL = _casadi.LL
    LR = _casadi.LR
    L_DICT = _casadi.L_DICT
    L_DOUBLE = _casadi.L_DOUBLE
    L_STR = _casadi.L_STR
    LABEL = _casadi.LABEL
def time_function(func,s = 1, return_times = False):
    # import time
    import timeit
    import numpy as np
    from scipy.sparse import csc_matrix
    setup = """
    
[buf,f_eval] = func.buffer()
res = []
for i in range(func.n_out()):
# 
    # res.append(csc_matrix((np.zeros(func.sparsity_in(i).nnz()).data, (func.sparsity_in(i).row(), func.sparsity_in(i).get_col())), ))
    res.append(np.zeros(func.sparsity_out(i).nnz()))
    buf.set_res(i, memoryview(res[-1]))
args = []
for i in range(func.n_in()):
    args.append(np.random.randn(func.sparsity_in(i).nnz()))
    buf.set_arg(i, memoryview(args[-1]))
times = []
for _ in range(5): f_eval()
"""
    timeit.timeit
    timer = timeit.Timer(stmt= 'f_eval()',setup = setup , globals = locals())
    total_time = 0
    times = []
    m = 0
    while total_time < s:
        n, time = timer.autorange()
        total_time += time
        m+= n 
        avg_time = time/n
        times.append(avg_time)
    print(f"Total running time: {total_time}s, repeated {m} times")
    times = np.array(times)
    median = np.median(times)
    unit = 's'
    d = 1
    if median < 1:
        d = 1e-3
        unit = 'ms'
    if median < 1e-3:
        d = 1e-6
        unit = 'us'
    if median < 1e-6:
        d = 1e-9
        unit = 'ns'
        
    print(f'Mean: {median/d: .3f} {unit}; standard deviation: {times.std()/d: .3f} {unit}; min: {times.min()/d: .3f} {unit}; max: {times.max()/d: .3f} {unit}')
    if return_times:
        return times
import pathlib, os, string, random, tempfile, shutil, atexit
class Jit(object):
    def randomword(self,length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
    def __init__(self,function,flags = ["-Ofast", "-fopenmp", "-march=native"], delete_afterwards = True, postpone_compilation = False, substitutions = []):
        
        self.folder = pathlib.Path(tempfile.mkdtemp())
        self.delete_afterwards = delete_afterwards
        self.c_file_original = self.folder / 'original.c'
        self.versions = [{'name': 'original', 'file': self.c_file_original, 'so_file':None, 'external': None, 'flags': flags, }]
        # self.so_file = None
        self.flags = flags
        self.original_function = function
        cg = ca.CodeGenerator(self.c_file_original.name)
        cg.add(function)
        cg.generate(str(self.folder) + "/") 
        atexit.register(lambda: shutil.rmtree(str(self.folder)))
        self.jitted_function = ca.Function()
        if not postpone_compilation:
            self.compile(0)

    def copy_original_file(self, substitutions = []):
        dest = self.folder / f"version_{len(self.versions)}.c"
        try:
            shutil.copy(self.c_file_original, dest)
            print(f"File copied from {self.c_file_original} to {dest}")
        except IOError as e:
            print(f"Unable to copy file. {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        self.versions.append({'name': f'version_{len(self.versions)}', 'file': dest, 'so_file':None, 'external': None,'flags': self.flags})
        self.replace_in_file(len(self.versions)-1,substitutions)

    def replace_in_file(self,version,substitutions):
        if len(substitutions) == 0: return
        
        file_path = self.versions[version]['file']
        
        with open(file_path, 'r') as file:
            content = file.read()
        for old,new in substitutions:
            content = content.replace(old,new)
            
        with open(file_path, 'w') as file:
            file.write(content)

    def compile(self, version = -1, flags = None):
        assert version < len(self.versions), f"Version {version} does not exist. The last version is {len(self.versions)-1}"

        if flags is None:
            flags = self.versions[version]['flags']
        else:
            self.versions[version]['flags'] = flags
        self.versions[version]['so_file'] = self.folder / (self.randomword(5) + '.so')
        compile_command = f"""gcc {" ".join(self.versions[version]['flags'])} -shared {str(self.versions[version]['file'])} -o {str(self.versions[version]['so_file'])}"""
        print(compile_command)
        print(os.system(compile_command))
        # self.versions[version]['so_file'] = self.versions[version]['so_file']
        self.versions[version]['external'] = ca.external(self.original_function.name(),str(self.versions[version]['so_file']))
        self.jitted_function = self.versions[version]['external']

    def __del__(self):
        if self.delete_afterwards:
            shutil.rmtree(str(self.folder))
            
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        if "jitted_function" in self.__dict__:
            return getattr(self.jitted_function, attr)
        raise AttributeError(f"'Jit' object has no attribute '{attr}'")
    def __str__(self):
        return self.jitted_function.__str__()

    def __repr__(self):
        return self.jitted_function.__repr__()
class Compile(object):
    def randomword(self,length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
    def __init__(self,file_name,path:pathlib.Path,function_name = None,function = None, get_cached_or_throw = False,flags = ["-Ofast", "-fopenmp", "-march=native"], ):
        self.temp_folder = pathlib.Path(tempfile.mkdtemp())
        self.path = path
        self.name_ = file_name
        self.so_name = file_name+'.so'
        self.flags = flags
        self.original_function = function
        self.c_file_original = self.temp_folder / (file_name + '.c')
        if get_cached_or_throw:
            assert function_name is not None, "Function name is None"
            if (self.path / self.so_name).exists():
                # Make a symlink so that the external function properly loads the function and doesn't use a cached one
                self.fake_so_file = self.temp_folder / (self.randomword(5) + '.so')
                os.symlink(str(self.path / self.so_name), str(self.fake_so_file))
                self.jitted_function = ca.external(function_name,str(self.fake_so_file))
                return
            else:
                raise Exception("Cached version does not exist")
        if function is None:
            raise Exception("Function is None")
        cg = ca.CodeGenerator(self.c_file_original.name)
        cg.add(function)
        cg.generate(str(self.temp_folder) + "/")         
        self.compile()
        
    def compile(self,  flags = None):        
        flags = self.flags
        so_file = self.path / self.so_name
        compile_command = f"""gcc {" ".join(flags)} -shared {str(self.c_file_original)} -o {str(so_file)}"""
        # compile_command = f"""gcc  -shared -Ofast -ffast-math -march=native -lamdlibmfast -lamdlibm -lm  -L/home/user/aocc-compiler-4.2.0/lib {str(self.c_file_original)} -o {str(so_file)}"""
        print("Compiling ",self.name_)
        print(compile_command)
        try:
            result = subprocess.run(compile_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("stdout:\n",result.stdout)
            print("stderr:\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Command failed with return code", e.returncode)
            print("Output:\n", e.output)
            print("Errors:\n", e.stderr)
            raise Exception(self.name_," compilation failed")
        print(self.name_," compiled successfully")
        
        self.fake_so_file = self.temp_folder / (self.randomword(5) + '.so')
        os.symlink(so_file, str(self.fake_so_file))        
        self.jitted_function = ca.external(self.original_function.name(),str(self.fake_so_file))
    def __set_state__(self,state):
        self.__dict__.update(state)


    def __del__(self):
        shutil.rmtree(str(self.temp_folder))
    def __call__(self,*args):
        return self.jitted_function(*args)
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        if "jitted_function" in self.__dict__:
            return getattr(self.jitted_function, attr)
        raise AttributeError(f"'Jit' object has no attribute '{attr}'")
    def __str__(self):
        return self.jitted_function.__str__()

    def __repr__(self):
        return self.jitted_function.__repr__()