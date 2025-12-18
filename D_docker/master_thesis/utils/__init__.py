# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# from pathlib import Path

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

# import importlib

# import .misc
# import .runge_kutta
# import .sympy_to_torch
# import .torch_controller
# import .my_pytorch.misc as misc
# importlib.reload(misc)
# importlib.reload(runge_kutta)
# importlib.reload(sympy_to_torch)
# importlib.reload(torch_controller)
# importlib.reload(misc)


from .misc import *

__all__ = ['get_urdf_path',"TOYS_ROOT",]