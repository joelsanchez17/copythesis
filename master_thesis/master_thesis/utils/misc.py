from typing import Callable
import os, sys, pathlib
import sympy as sp

def get_root_directory():
    path = pathlib.Path(os.getcwd()).resolve()
    while not (root_file := path / "root_toys").is_file():
        path = path.parent
        if path == path.parent:
            return None
    # sys.path.insert(0, str(path))
    return (root_folder:=path.resolve())

TOYS_ROOT = get_root_directory()

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# this sort of works ok
def get_annotations(func:Callable,depth:int=1 , current_depth = 0) -> str:
    annotation = ''
    if depth == 0:
        return annotation
    if not hasattr(func,'__annotations__'):
        if hasattr(func,'__args__'):
            for arg in func.__args__:
                
                annotation += '\n' + '\t'*current_depth + ('↳ ' if current_depth!=0 else '') + str(arg)
                # annotation += str(arg)
                annotation += get_annotations(arg,depth,current_depth+1)
        return annotation
    if len(func.__annotations__.keys()) == 1 and 'return' in func.__annotations__.keys():
        return annotation
    for k,v in func.__annotations__.items():
        if k == 'return':
            
            annotation += '\n'+  '\t'*current_depth + ('↳ ' if current_depth!=0 else '') +'Return: '
            annotation += f'{v}'
        else:
            annotation += '\n'+'\t'*current_depth+ ('↳ ' if current_depth!=0 else '') +'Parameter: '
            annotation += f'{k}: {v}'
        # annotation += '\t'*current_depth + f'{v}' + '\t'*current_depth + get_annotations(v,depth-1,current_depth+1)
        annotation += get_annotations(v,depth-1,current_depth+1)
    return annotation


def get_urdf_path(path) -> str:
    root = get_root_directory()
    if ".urdf" not in path:
        path = path + ".urdf"
    path = root / "urdf_files" / path
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
    return path.resolve().as_posix()


def latex_to_image(latex_code,filename,output = 'png', dvioptions=['-D','1200'], **kwargs):
    sp.preview(latex_code, viewer='file',output=output, filename=filename, dvioptions=dvioptions,**kwargs)
def time_function(func,r = None, s = 1):
    import time
    import numpy as np
    [buf,f_eval] = func.buffer()
    res = []
    for i in range(func.n_out()):
        res.append(np.zeros(func.sparsity_out(i).shape))
        buf.set_res(i, memoryview(res[-1]))
    args = []
    for i in range(func.n_in()):
        args.append(np.random.randn(*func.sparsity_in(i).shape))
        buf.set_arg(i, memoryview(args[-1]))
    times = []
    tt = time.time()
    if not r:
        t0 = time.time()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        t1 = time.time()
        r = int(np.ceil(s/(t1-t0),))
    for it in range(r):
        t0 = time.time()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        f_eval()
        t1 = time.time()
        times.append((t1-t0)/10 )
    times = np.array(times)
    print("Total time:")
    print(time.time()-tt)
    d = 1
    unit = 's'
    mean = times.mean()
    # mean = 0.0001
    if mean < 1:
        d = 1e-3
        unit = 'ms'
    if mean < 1e-3:
        d = 1e-6
        unit = 'us'
    if mean < 1e-6:
        d = 1e-9
        unit = 'ns'
    print(f'Number of repetitions: {r*10}')
    print(f'Mean: {mean/d: .3f} {unit}; standard deviation: {times.std()/d: .3f} {unit}; min: {times.min()/d: .3f} {unit}; max: {times.max()/d: .3f} {unit}')
 