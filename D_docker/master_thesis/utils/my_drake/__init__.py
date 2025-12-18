import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from pathlib import Path

# def find_root_directory(start_path):
#     path = Path(start_path).resolve()

#     while True:
#         root_file = path / "root_toys"
#         if root_file.exists() and root_file.is_file():
#             return path
#         parent_path = path.parent
#         if parent_path == path:
#             break
#         path = parent_path

#     return None
# sys.path.insert(0, str(find_root_directory(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))))
# import os, sys, pathlib
# def add_root_directory():
#     path = pathlib.Path(os.getcwd()).resolve()
#     while not (root_file := path / "root_toys").is_file():
#         path = path.parent
#         if path == path.parent:
#             return None
#     if str(path) not in sys.path:
#         sys.path.insert(0, str(path))
#     return (root_folder:=path)
# add_root_directory()

# print('aaa')
# import misc
# import importlib
from .casadi import *
from .systems import *
# import controller, data, training, dynamical_system, training_parameters
# importlib.reload(controller)
# importlib.reload(data)
# importlib.reload(training)
# importlib.reload(dynamical_system)
# importlib.reload(training_parameters)
# from controller import *
# from data import *
# from training import *
# from dynamical_system import *
# from training_parameters import *