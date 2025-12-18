# import os, sys, pathlib
# def add_root_directory():
#     path = pathlib.Path(os.getcwd()).resolve()
#     while not (root_file := path / "root_toys").is_file():
#         path = path.parent
#         if path == path.parent:
#             return None
#     sys.path.insert(0, str(path))
# add_root_directory()

from utils.my_casadi.math import *

def delete_casadi_function(ob):
    
    try:
        idx = list(globals().values()).index(ob)
        key = list(globals().keys())[idx]
        import gc
        gc.collect(2)
        ca._casadi.delete_Function(ob)
        del globals()[key]
        gc.collect(2)
    except:
        pass
    return None