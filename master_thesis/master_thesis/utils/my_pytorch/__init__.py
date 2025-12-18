# import os, sys, pathlib
# def add_root_directory():
#     path = pathlib.Path(os.getcwd()).resolve()
#     while not (root_file := path / "root_toys").is_file():
#         path = path.parent
#         if path == path.parent:
#             return None
#     sys.path.insert(0, str(path))
# add_root_directory()

