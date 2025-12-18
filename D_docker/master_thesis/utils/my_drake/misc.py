# import os, sys, pathlib
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# def add_root_directory():
#     path = pathlib.Path(os.getcwd()).resolve()
#     while not (root_file := path / "root_toys").is_file():
#         path = path.parent
#         if path == path.parent:
#             return None
#     sys.path.insert(0, str(path))
# add_root_directory()

# from utils.misc import is_notebook


from pydrake.all import (
    namedview, JointIndex, MultibodyPlant,SpatialInertia, UnitInertia, RigidTransform, CoulombFriction, RigidTransform,BodyIndex,FrameIndex,ModelInstanceIndex,
    InverseKinematics,Solve,SceneGraph,Frame,Context, Body)
import pydrake
import pydrake.geometry as pydgeo
import pydrake.multibody.tree as pydtr

import logging

from IPython.display import SVG, display
import pydot
import numpy as np
from pydrake.visualization import ApplyVisualizationConfig
from typing import List, Union,Optional, Tuple,Iterable,Callable
# import cairosvg
# from PIL import Image
# from io import BytesIO
# import matplotlib.pyplot as plt
import typing as T
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
class VisualizerHelper:
    def __init__(self,plant,diagram):
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.diagram_context)
        self.visualizer = diagram.GetSubsystemByName('meshcat_visualizer(visualizer)')
        self.visualizer_context = self.visualizer.GetMyContextFromRoot(self.diagram_context)
        self.plant = plant
        self.diagram = diagram
    def set_free_body_position(self, instance, transform:T.Union[RigidTransform, T.Iterable]):
        if isinstance(transform,RigidTransform):
            position = transform.translation()
            quaternion = transform.rotation().ToQuaternion().wxyz()
        if isinstance(transform, T.Iterable):
            if len(transform) == 3:
                position = transform
                quaternion = [1.0,0,0,0]
            else:
                position = transform[4:]
                quaternion = transform[:4]
        vector = np.hstack([quaternion,position])
        if isinstance(instance,str):
            instance = self.plant.GetModelInstanceByName(instance)
        self.plant.SetPositions(self.plant_context,instance,vector)
    def set_position(self,instance, position):
        if isinstance(instance,str):
            instance = self.plant.GetModelInstanceByName(instance)            
        self.plant.SetPositions(self.plant_context,instance,position)
    def publish(self):
        # TODO: somnething something collision etc
        # self.diagram.ForcedPublish(self.diagram_context)
        self.visualizer.ForcedPublish(self.visualizer_context)
    def publish_diagram(self):
        self.diagram.ForcedPublish(self.diagram_context)

def display_diagram(diagram,matplotlib = False,depth=1):
    if is_notebook() and not matplotlib:
       display(
            SVG(
                pydot.graph_from_dot_data(
                    diagram.GetGraphvizString(max_depth=depth))[0].create_svg()))
       
    else:
        print("install cairosvg and edit this file")
        # img_png = cairosvg.svg2png(pydot.graph_from_dot_data(
        #             diagram.GetGraphvizString(max_depth=1))[0].create_svg())
        # img = Image.open(BytesIO(img_png))
        # plt.imshow(img)
        # plt.show()
def is_listy(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False

def make_namedview_positions(
     mbp:MultibodyPlant, view_name:str, model_instance:Union[ModelInstanceIndex,List[ModelInstanceIndex],None] = None,add_suffix_if_single_position:bool=False, body_suffix:bool = True, model_suffix:bool = True
):
    """
    Create a namedview of the positions of the joints and floating base bodies in a MultibodyPlant.

    Args:
        mbp (MultibodyPlant): The MultibodyPlant instance to extract the positions from.
        view_name (str): The name of the resulting namedview.
        model_instance (ModelInstanceIndex or list of ModelInstanceIndex, optional): The model instance(s) to extract the positions from. If None, all model instances are used. Defaults to None.
        add_suffix_if_single_position (bool, optional): Whether to add a suffix to the joint name if it has only one position. Defaults to False.
        body_suffix (bool, optional): Whether to add the body name as a suffix to the floating base body position names. Defaults to True.
        model_suffix (bool, optional): Whether to add the model instance name as a suffix to the joint and floating base body position names. Defaults to True.

    Returns:
        NamedView: A namedview of the positions of the joints and floating base bodies in the MultibodyPlant.
    """
    if model_instance is not None:
        if not is_listy(model_instance):
            model_instance = [model_instance]
    names = [None] * mbp.num_positions()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        if model_instance is not None and joint.model_instance() not in model_instance:
            continue
        model_name = mbp.GetModelInstanceName(joint.model_instance())+"_" if model_suffix else ""
        suffix = model_name 
        if joint.num_positions() == 1 and not add_suffix_if_single_position:
            names[joint.position_start()] = f"{suffix}{joint.name()}"
        else:
            for i in range(joint.num_positions()):
                names[
                    joint.position_start() + i
                ] = f"{suffix}{joint.name()}_{joint.position_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        if model_instance is not None and body.model_instance() not in model_instance:
            continue
        model_name = mbp.GetModelInstanceName(body.model_instance())+"_" if model_suffix else ""
        body_name = body.name()+'_' if body_suffix else ""
        suffix = model_name  + body_name 
        start = body.floating_positions_start()
        for i in range(7):
            names[
                start + i
            ] = f"{suffix}{body.floating_position_suffix(i)}"
    return namedview(view_name, [names[i] for i in range(len(names)) if names[i] is not None])


def make_namedview_velocities(
    mbp:MultibodyPlant, view_name:str, model_instance:Union[ModelInstanceIndex,List[ModelInstanceIndex],None] = None,add_suffix_if_single_velocity:bool=False, body_suffix:bool = True, model_suffix:bool = True
):
    """
    Create a namedview of the velocities of the joints and floating base bodies in a MultibodyPlant.

    Args:
        mbp (MultibodyPlant): The MultibodyPlant instance to extract the velocities from.
        view_name (str): The name of the resulting namedview.
        model_instance (ModelInstanceIndex or list of ModelInstanceIndex, optional): The model instance(s) to extract the velocities from. If None, all model instances are used. Defaults to None.
        add_suffix_if_single_position (bool, optional): Whether to add a suffix to the joint name if it has only one velocity. Defaults to False.
        body_suffix (bool, optional): Whether to add the body name as a suffix to the floating base body velocity names. Defaults to True.
        model_suffix (bool, optional): Whether to add the model instance name as a suffix to the joint and floating base body velocities names. Defaults to True.

    Returns:
        NamedView: A namedview of the velocities of the joints and floating base bodies in the MultibodyPlant.
    """
    # if body_suffix is False and there's repeated names, then the view will give error but I ain't fixing it
    if model_instance is not None:
        if not is_listy(model_instance):
            model_instance = [model_instance]
    names = [None] * mbp.num_velocities()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        
        if model_instance is not None and joint.model_instance() not in model_instance:
            continue
        model_name = mbp.GetModelInstanceName(joint.model_instance())+"_" if model_suffix else ""
        suffix = model_name 
        if joint.num_velocities() == 1 and not add_suffix_if_single_velocity:
            names[joint.velocity_start()] = f"{suffix}{joint.name()}"
        else:
            for i in range(joint.num_velocities()):
                names[
                    joint.velocity_start() + i
                ] = f"{suffix}{joint.name()}_{joint.velocity_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        if model_instance is not None and body.model_instance() not in model_instance:
            
            continue
        model_name = mbp.GetModelInstanceName(body.model_instance())+"_" if model_suffix else ""
        body_name = body.name()+'_' if body_suffix else ""
        suffix = model_name  + body_name 
        start = body.floating_velocities_start() - mbp.num_positions()
        for i in range(6):
            names[
                start + i
            ] = f"{suffix}{body.floating_velocity_suffix(i)}"
    # print(names)
    return namedview(view_name, [names[i] for i in range(len(names)) if names[i] is not None])
    return names

def make_namedview_state(mbp: MultibodyPlant, view_name:str, model_instance:Union[ModelInstanceIndex,None]=None, body_suffix:bool=True, model_suffix:bool=True):
    """
    Creates a named view state for a given MultibodyPlant object.

    Args:
        mbp (MultibodyPlant): The MultibodyPlant object.
        view_name (str): The name of the view.
        model_instance (ModelInstanceIndex, optional): The model instance ID. Defaults to None.
        body_suffix (bool, optional): Whether to add a suffix to the body names. Defaults to True.
        model_suffix (bool, optional): Whether to add a suffix to the model names. Defaults to True.

    Returns:
        NamedView: The named view state.
    """
    
    pview = make_namedview_positions(mbp, f"{view_name}_pos", model_instance=model_instance, body_suffix=body_suffix, model_suffix=model_suffix, add_suffix_if_single_position=True)
    vview = make_namedview_velocities(mbp, f"{view_name}_vel", model_instance=model_instance, body_suffix=body_suffix, model_suffix=model_suffix, add_suffix_if_single_velocity=True)
    return namedview(view_name, pview.get_fields() + vview.get_fields())

# make_namedview_state(mbp, "state",model_instance = [robot1,robot2,obstacle_instance],body_suffix = False, model_suffix = True).get_fields()
def create_default_view(view: namedview):
    """
    Creates a default view with all states set to zero and if quaternion (has qw), then w = 1.

    Args:
        view (View): The view to create the default view for.

    Returns:
        View: The default view.
    """
    return view([1.  if name[-2:] == 'qw' else 0 for name in view.get_fields()])


def add_default_meshcat_visualization(builder,meshcat,apply_config=True) -> pydrake.all.MeshcatVisualizer:
    scene_graph = next(x for x in builder.GetSystems() if type(x) is pydrake.all.SceneGraph)
    if apply_config:
        ApplyVisualizationConfig(pydrake.visualization.VisualizationConfig(), builder,meshcat=meshcat)
        
    visualizer = pydrake.all.MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat
        )
    return visualizer

class CompositeShape:
    """
    A class for adding a composite shape to a MultibodyPlant.

    This class provides a context manager interface for adding a composite shape
    to a MultibodyPlant. The composite shape is defined as a collection of rigid
    bodies that are welded together to form a single rigid body. The composite
    shape can be used to represent complex geometries that cannot be represented
    by a single primitive shape.

    Args:
        plant (MultibodyPlant): The MultibodyPlant to which the composite shape will be added.
        scene_graph (SceneGraph): The SceneGraph associated with the MultibodyPlant.
        instance_name (str): The name of the model instance to be added to the MultibodyPlant.
        frame_name (str): The name of the frame to be added to the MultibodyPlant.

    Attributes:
        plant (MultibodyPlant): The MultibodyPlant to which the composite shape will be added.
        instance_name (str): The name of the model instance to be added to the MultibodyPlant.
        frame_name (str): The name of the frame to be added to the MultibodyPlant.
        model_instance (int): The index of the model instance added to the MultibodyPlant.
        main_frame (Frame): The frame of the main body of the composite shape.
        scene_graph (SceneGraph): The SceneGraph associated with the MultibodyPlant.

    Methods:
        __init__(self, plant: MultibodyPlant, scene_graph: SceneGraph, instance_name: str, frame_name: str)
            Initializes the CompositeShape object.
        __enter__(self)
            Adds the model instance and frame to the MultibodyPlant.
        frame_and_instance(self)
            Adds the model instance and frame to the MultibodyPlant.
        __exit__(self, exc_type, exc_val, exc_tb)
            Excludes collisions within the bodies of the model instance.
        add_welded_body(self, body_name, spatial_inertia, transform=RigidTransform())
            Adds a welded body to the composite shape. This one just automatically welds the body to the main frame.
        register_body_geometry(self, body, geometry, transform=RigidTransform(), color=np.array([0.5,0.5,0.5,1]), friction=CoulombFriction())
            Registers the visual and collision geometries of a body with the same geometry and transform.
            
        add_body(self, body_name, geometry, transform=RigidTransform(), spatial_inertia=None, color=np.array([0.5,0.5,0.5,1]), friction=CoulombFriction())
            Adds a body to the composite shape with a bunch of default parameters. Returns the body, visual geometry, and collision geometry.
    """
    def __init__(self, plant: MultibodyPlant, scene_graph:pydrake.all.SceneGraph,instance_name: str, frame_name: str):
        self.plant = plant
        self.instance_name = instance_name
        self.frame_name = frame_name
        self.model_instance = None
        self.main_frame = None
        self.scene_graph = scene_graph

    def __enter__(self):
        # Add the model instance and frame to the plant
        if self.model_instance is None:
            self.model_instance, self.main_frame = self.frame_and_instance()
        return self,self.model_instance, self.main_frame
    def frame_and_instance(self):
        self.model_instance = self.plant.AddModelInstance(self.instance_name)
        # spatial_inertia = SpatialInertia(
        #         mass=0.0001, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=UnitInertia.PointMass(np.array([0.0, 0.0, 0.0])))
        self.main_body = self.plant.AddRigidBody(self.frame_name, self.model_instance, SpatialInertia())
        self.main_frame = self.main_body.body_frame()
        # self.main_frame = self.plant.AddFrame(FixedOffsetFrame(self.frame_name, self.plant.world_frame(), X_PF=RigidTransform()))
        return self.model_instance, self.main_frame
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Gather all bodies added to the model instance
        bodies = [self.plant.get_body(body) for body in self.plant.GetBodyIndices(self.model_instance)]

        # Exclude collisions within the bodies of the model instance
        
        self.scene_graph.collision_filter_manager().Apply(
            pydgeo.CollisionFilterDeclaration().ExcludeWithin(self.plant.CollectRegisteredGeometries(bodies)))

    def add_welded_body(self, body_name:str, spatial_inertia:SpatialInertia,transform:Optional[RigidTransform]=RigidTransform()) -> pydtr.Body:
        body = self.plant.AddRigidBody(body_name, self.model_instance, spatial_inertia)
        # Attach the body to the main frame
        self.plant.WeldFrames(self.main_frame, body.body_frame(), transform)
        return body
    def register_body_geometry(self,body:pydtr.Body,geometry:pydgeo.Shape,transform:Optional[RigidTransform] = RigidTransform(),color:Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction:Optional[CoulombFriction] = CoulombFriction()) -> Tuple[pydgeo.GeometryId,pydgeo.GeometryId]:
        visual_geo = self.plant.RegisterVisualGeometry(body, transform, geometry, "visual",color)
        collision_geo = self.plant.RegisterCollisionGeometry(body, transform, geometry, "collision",friction)
        return visual_geo,collision_geo
    def add_body(self,body_name:str,geometry:pydgeo.Shape,transform:Optional[RigidTransform] = RigidTransform(),spatial_inertia: Optional[SpatialInertia] = None,color:Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction:Optional[CoulombFriction] = CoulombFriction()):
        if spatial_inertia is None:
            if isinstance(geometry, pydgeo.Box):
                inertia = UnitInertia.SolidBox(
                    geometry.width(), geometry.depth(), geometry.height()
                )
            elif isinstance(geometry, pydgeo.Cylinder):
                inertia = UnitInertia.SolidCylinder(geometry.radius(), geometry.length())
            elif isinstance(geometry, pydgeo.Sphere):
                inertia = UnitInertia.SolidSphere(geometry.radius())
            elif isinstance(geometry, pydgeo.Capsule):
                inertia = UnitInertia.SolidCylinder(geometry.radius(), geometry.length())
            spatial_inertia = SpatialInertia(
                mass=1., p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia
            )
        body = self.add_welded_body(body_name,spatial_inertia,transform)
        visual_geo,collision_geo = self.register_body_geometry(body,geometry,RigidTransform(),color,friction)
        return body,visual_geo,collision_geo
    
    
def get_all_bodies(plant: MultibodyPlant) -> List[pydtr.Body]:
    """Returns all bodies in the plant.
    
    Args:
        plant: A multibody plant object.
    
    Returns:
        A list of all bodies in the plant.
    """
    return [plant.get_body(BodyIndex(i)) for i in range(plant.num_bodies())]

def get_bodies_from_instance(plant: MultibodyPlant, model_instance: Union[ModelInstanceIndex, List[ModelInstanceIndex]]) -> List[pydtr.Body]:
    """Returns bodies associated with a specific model instance or instances.
    
    Args:
        plant: A multibody plant object.
        model_instance: A ModelInstanceIndex or list of ModelInstanceIndices.
    
    Returns:
        A list of bodies associated with the given model instance(s).
    """
    if type(model_instance) is ModelInstanceIndex:
        model_instance = [model_instance]
    return [plant.get_body(i) for instance in model_instance for i in plant.GetBodyIndices(instance)]

def get_bodies_from_instance_name(plant: MultibodyPlant, names: Union[str, List[str]]) -> List[pydtr.Body]:

    """Returns bodies associated with a specific model instance name or names.
    
    Args:
        plant: A multibody plant object.
        names: A string or list of strings representing model instance names.
    
    Returns:
        A list of bodies associated with the given model instance name(s).
    """
    if type(names) is str:
        names = [names]
    return [plant.get_body(i) for name in names for i in plant.GetBodyIndices(plant.GetModelInstanceByName(name))]

def get_all_instances(plant: 'MultibodyPlant') -> List['ModelInstanceIndex']:

    """Returns all model instances in the plant.
    
    Args:
        plant: A multibody plant object.
    
    Returns:
        A list of all model instances in the plant.
    """
    return [(ModelInstanceIndex(i)) for i in range(plant.num_model_instances())]

def get_all_frames(plant: 'MultibodyPlant') -> List[pydtr.Frame]:
    """Returns all frames in the plant.
    
    Args:
        plant: A multibody plant object.
    
    Returns:
        A list of all frames in the plant.
    """
    return [plant.get_frame(FrameIndex(i)) for i in range(plant.num_frames())]

def get_frames_from_instance(plant: 'MultibodyPlant', model_instance: Union['ModelInstanceIndex', List['ModelInstanceIndex']]) -> List[pydtr.Frame]:
    """Returns frames associated with a specific model instance or instances.
    
    Args:
        plant: A multibody plant object.
        model_instance: A ModelInstanceIndex or list of ModelInstanceIndices.
    
    Returns:
        A list of frames associated with the given model instance(s).
    """
    if type(model_instance) is ModelInstanceIndex:
        model_instance = [model_instance]
    return [plant.get_frame(FrameIndex(i)) for instance in model_instance for i in range(plant.num_frames()) if plant.get_frame(FrameIndex(i)).model_instance() == instance]

def get_frames_from_instance_name(plant: 'MultibodyPlant', names: Union[str, List[str]]) -> List[pydtr.Frame]:

    """Returns frames associated with a specific model instance name or names.
    
    Args:
        plant: A multibody plant object.
        names: A string or list of strings representing model instance names.
    
    Returns:
        A list of frames associated with the given model instance name(s).
    """
    if type(names) is str:
        names = [names]
    return [plant.get_frame(FrameIndex(i)) for name in names for i in  range(plant.num_frames()) if plant.get_frame(FrameIndex(i)).model_instance() == plant.GetModelInstanceByName(name)]

def get_all_joints(plant: 'MultibodyPlant') -> List[pydtr.Joint]:
    """Returns all joints in the plant.
    
    Args:
        plant: A multibody plant object.
    
    Returns:
        A list of all joints in the plant.
    """
    return [plant.get_joint(JointIndex(i)) for i in range(plant.num_joints())]
def get_joints_from_instance(plant: 'MultibodyPlant', model_instance: Union['ModelInstanceIndex', List['ModelInstanceIndex']]) -> List[pydtr.Joint]:

    """Returns joints associated with a specific model instance or instances.
    
    Args:
        plant: A multibody plant object.
        model_instance: A ModelInstanceIndex or list of ModelInstanceIndices.
    
    Returns:
        A list of joints associated with the given model instance(s).
    """
    if type(model_instance) is ModelInstanceIndex:
        model_instance = [model_instance]
    return [plant.get_joint(JointIndex(i)) for instance in model_instance for i in range(plant.num_joints()) if plant.get_joint(JointIndex(i)).model_instance() == instance]

def get_joints_from_instance_name(plant: 'MultibodyPlant', names: Union[str, List[str]]) -> List[pydtr.Joint]:

    """Returns joints associated with a specific model instance name or names.
    
    Args:
        plant: A multibody plant object.
        names: A string or list of strings representing model instance names.
    
    Returns:
        A list of joints associated with the given model instance name(s).
    """
    if type(names) is str:
        names = [names]
    return [plant.get_joint(JointIndex(i)) for name in names for i in range(plant.num_joints()) if plant.get_joint(JointIndex(i)).model_instance() == plant.GetModelInstanceByName(name)]

# def get_all_actuated_joints(plant: 'MultibodyPlant') -> List[pydtr.Joint]:
#     """Returns all joints in the plant.
    
#     Args:
#         plant: A multibody plant object.
    
#     Returns:
#         A list of all joints in the plant.
#     """
#     return [plant.get_joint(JointIndex(i)) for i in range(plant.num_actuated_dofs())]
def calculate_position_inverse_kinematics(plant: MultibodyPlant,desired_position: np.ndarray, frame: Union[pydtr.Frame,str]):
    """
    Computes the inverse kinematics solution for a desired position of a specified frame in the plant.
    
    Args:
        plant (MultibodyPlant): The multibody plant object.
        desired_position (np.ndarray): The desired 3D position for the frame.
        frame (Union['Frame', str]): The target frame or its name.
    
    Returns:
        np.ndarray: The solution of the inverse kinematics problem.
    
    Raises:
        ValueError: If the inverse kinematics solution fails.
    """
    if type(frame) == str:
        frame=  plant.GetFrameByName(frame)
    ik = InverseKinematics(plant, plant.CreateDefaultContext())
    ik.AddPositionConstraint(frame, [0, 0, 0], plant.world_frame(), desired_position, desired_position)
    result = Solve(ik.prog())
    if not result.is_success():
        raise ValueError("IK failed")
    return result.GetSolution()



class ObjectFactory:
    """TODO: Create automatically a controller that can be used to follow a trajectory, something something fake plant for control maybe https://github.com/RobotLocomotion/drake/blob/master/examples/planar_gripper/planar_gripper_simulation.cc
    TODO: Composite objects
    """
    @classmethod
    def plant_scene_graph(cls,plant,scene_graph) -> (0):
        if plant is None:
            plant = MultibodyPlant(0.0)
        if scene_graph is None:
            scene_graph = SceneGraph()
            try:
                plant.RegisterAsSourceForSceneGraph(scene_graph)
            except:
                logging.warning("Plant already registered as source for a scene graph")
        return plant,scene_graph
    
    @staticmethod
    def object_from_name(shape:str,**kwargs) -> Tuple[Iterable[pydtr.Body],Frame,MultibodyPlant,SceneGraph]:
        """
        Implemented objects: `box`, `sphere`, `cylinder`
        
        Get default arguments: `ObjectFactory.default_{object}_arguments()`
        """
        if shape == 'box':
            return ObjectFactory.box_object(**kwargs)
        elif shape == 'sphere':
            return ObjectFactory.sphere_object(**kwargs)
        elif shape == 'cylinder':
            return ObjectFactory.cylinder_object(**kwargs)
        else:
            raise ValueError("Unknown object shape")        
        
    @staticmethod
    def default_box_arguments():
        return {"name": 'box',"width":0.1,"depth":0.1,"height":0.1,"color":np.array([0.5,0.5,0.5,1]),"friction":CoulombFriction(0.9,0.8),"mass":1.,"collision":True}
    @staticmethod
    def box_object(name:str,plant:Optional[MultibodyPlant] = None,scene_graph:Optional[SceneGraph] = None,
                   width: Optional[float] = 0.1,depth: Optional[float] = 0.1,height: Optional[float] = 0.1,
                   color: Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction: Optional[CoulombFriction] = CoulombFriction(0.9,0.8),
                   mass: Optional[float] = 1., collision = True,**kwargs) -> Tuple[Iterable[pydtr.Body],Frame,MultibodyPlant,SceneGraph]:
        plant,scene_graph = ObjectFactory.plant_scene_graph(plant,scene_graph)
        box = pydgeo.Box(width, depth, height)
        inertia = UnitInertia.SolidBox(
                    width, depth, height
                )
        # UnitInertia.SolidBox()
        spatial_inertia = SpatialInertia(mass = mass,p_PScm_E = np.array([0.0, 0.0, 0.0]), G_SP_E = inertia)
        instance= plant.AddModelInstance(name)
        body = plant.AddRigidBody(name = name, M_BBo_B=spatial_inertia,model_instance = instance)
        if collision:
            visual_geo,collision_geo = ObjectFactory.register_collision_visual_geometry(plant,body,box,RigidTransform(),color,friction)
        else:
            visual_geo = ObjectFactory.register_visual_geometry(plant,body,box,RigidTransform(),color,friction)
        return body,body.body_frame(),plant,scene_graph
    @staticmethod
    def default_sphere_arguments():
        return {"name": 'sphere',"radius":0.1,"color":np.array([0.5,0.5,0.5,1]),"friction":CoulombFriction(0.9,0.8),"mass":1.,"collision":True}
    @staticmethod
    def sphere_object(name:str,plant:Optional[MultibodyPlant] = None,scene_graph:Optional[SceneGraph] = None,radius: Optional[float] = 0.1,color: Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction: Optional[CoulombFriction] = CoulombFriction(0.9,0.8),mass: Optional[float] = 1., collision = True,**kwargs) -> Tuple[Iterable[pydtr.Body],Frame,MultibodyPlant,SceneGraph]:
        plant,scene_graph = ObjectFactory.plant_scene_graph(plant,scene_graph)
        sphere = pydgeo.Sphere(radius)
        inertia = UnitInertia.SolidSphere(radius)
        spatial_inertia = SpatialInertia(mass = mass,p_PScm_E = np.array([0.0, 0.0, 0.0]), G_SP_E = inertia)
        instance= plant.AddModelInstance(name)
        body = plant.AddRigidBody(name = name, M_BBo_B=spatial_inertia,model_instance = instance)
        if collision:
            visual_geo,collision_geo = ObjectFactory.register_collision_visual_geometry(plant,body,sphere,RigidTransform(),color,friction)
        else:
            visual_geo = ObjectFactory.register_visual_geometry(plant,body,sphere,RigidTransform(),color,friction)
        return body,body.body_frame(),plant,scene_graph
    @staticmethod
    def default_cylinder_arguments():
        return {"name": 'cylinder',"radius":0.1,"length":0.1,"color":np.array([0.5,0.5,0.5,1]),"friction":CoulombFriction(0.9,0.8),"mass":1.,"collision":True}
    @staticmethod
    def cylinder_object(name:str,plant:Optional[MultibodyPlant] = None,scene_graph:Optional[SceneGraph] = None,radius: Optional[float] = 0.1,
                        length: Optional[float] = 0.1,color: Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction: Optional[CoulombFriction] = CoulombFriction(0.9,0.8),
                        mass: Optional[float] = 1., spatial_inertia: Optional[SpatialInertia] = None, collision = True,**kwargs) -> Tuple[Iterable[pydtr.Body],Frame,MultibodyPlant,SceneGraph]:
        plant,scene_graph = ObjectFactory.plant_scene_graph(plant,scene_graph)
        cylinder = pydgeo.Cylinder(radius, length)
        if spatial_inertia is None:
            inertia = UnitInertia.SolidCylinder(radius, length,unit_vector=[0,0,1])
            spatial_inertia = SpatialInertia(mass = mass,p_PScm_E = np.array([0.0, 0.0, 0.0]), G_SP_E = inertia)
        instance= plant.AddModelInstance(name)
        body = plant.AddRigidBody(name = name, M_BBo_B=spatial_inertia,model_instance = instance)
        if collision:
            visual_geo,collision_geo = ObjectFactory.register_collision_visual_geometry(plant,body,cylinder,RigidTransform(),color,friction)
        else:
            visual_geo = ObjectFactory.register_visual_geometry(plant,body,cylinder,RigidTransform(),color,friction)
        return body,body.body_frame(),plant,scene_graph
    @staticmethod
    def default_capsule_arguments():
        return {"name":'capsule', "radius":0.1,"length":0.1,"color":np.array([0.5,0.5,0.5,1]),"friction":CoulombFriction(0.9,0.8),"mass":1.,"collision":True}
    @staticmethod
    def capsule_object(name:str,plant:Optional[MultibodyPlant] = None,scene_graph:Optional[SceneGraph] = None,radius: Optional[float] = 0.1,length: Optional[float] = 0.1,color: Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction: Optional[CoulombFriction] = CoulombFriction(0.9,0.8),mass: Optional[float] = 1., collision = True,**kwargs) -> Tuple[Iterable[pydtr.Body],Frame,MultibodyPlant,SceneGraph]:
        plant,scene_graph = ObjectFactory.plant_scene_graph(plant,scene_graph)
        capsule = pydgeo.Capsule(radius, length)
        inertia = UnitInertia.SolidCylinder(radius, length)
        spatial_inertia = SpatialInertia(mass = mass,p_PScm_E = np.array([0.0, 0.0, 0.0]), G_SP_E = inertia)
        instance= plant.AddModelInstance(name)
        body = plant.AddRigidBody(name = name, M_BBo_B=spatial_inertia,model_instance = instance)
        if collision:
            visual_geo,collision_geo = ObjectFactory.register_collision_visual_geometry(plant,body,capsule,RigidTransform(),color,friction)
        else:
            visual_geo = ObjectFactory.register_visual_geometry(plant,body,capsule,RigidTransform(),color,friction)
        return body,body.body_frame(),plant,scene_graph
    
    @staticmethod
    def register_collision_visual_geometry(plant:MultibodyPlant,body:pydtr.Body,geometry:pydgeo.Shape,transform:Optional[RigidTransform] = RigidTransform(),color:Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction:Optional[CoulombFriction] = CoulombFriction()) -> Tuple[pydgeo.GeometryId,pydgeo.GeometryId]:
        visual_geo = plant.RegisterVisualGeometry(body, transform, geometry, "visual",color)
        collision_geo = plant.RegisterCollisionGeometry(body, transform, geometry, "collision",friction)
        return visual_geo,collision_geo
    @staticmethod
    def register_visual_geometry(plant:MultibodyPlant,body:pydtr.Body,geometry:pydgeo.Shape,transform:Optional[RigidTransform] = RigidTransform(),color:Optional[np.ndarray] = np.array([0.5,0.5,0.5,1]),friction:Optional[CoulombFriction] = CoulombFriction()) -> Tuple[pydgeo.GeometryId,pydgeo.GeometryId]:
        visual_geo = plant.RegisterVisualGeometry(body, transform, geometry, "visual",color)
        # collision_geo = plant.RegisterCollisionGeometry(body, transform, geometry, "collision",friction)
        return visual_geo
def get_continuous_state_vector_from_context(context:Context):
    return context.get_continuous_state_vector().CopyToVector()
def get_continuous_namedview_from_context(context:Context, namedview:'namedview'):
    return namedview(get_continuous_state_vector_from_context(context))
def get_discrete_state_vector_from_context(context:Context):
    return context.get_discrete_state_vector().CopyToVector()
def get_discrete_namedview_from_context(context:Context, namedview:'namedview'):
    return namedview(get_discrete_state_vector_from_context(context))
def get_input_vector_from_context(context:Context, input_port:Optional[Union[int,str]] = None):
    if isinstance(input_port, int):
        input_port = context.get_input_port(input_port)
    elif input_port is None:
        input_port = context.get_input_port()
    elif isinstance(input_port, str):
        input_port = context.Get(input_port)
    return input_port.Eval(context).CopyToVector()
def get_input_namedview_from_context(context:Context, namedview:'namedview', input_port:Optional[Union[int,str]] = None):
    return namedview(get_input_vector_from_context(context, input_port))
def copy_namedview(namedview:'namedview'):
    return type(namedview)(namedview[:])

def get_all_free_bodies(plant:MultibodyPlant) -> List[Body]:
    """
    Returns a list of free bodies (i.e, have quaternion dofs) in the plant
    """
    free_bodies = []
    for body_index in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(body_index))
        
        if body.has_quaternion_dofs():
            free_bodies.append(body)
    return free_bodies
def get_all_free_bodies_model_instances(plant:MultibodyPlant) -> List[ModelInstanceIndex]:
    """
    Returns a list of free bodies (i.e, have quaternion dofs) in the plant
    """
    free_bodies = []
    for body_index in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(body_index))
        if body.has_quaternion_dofs():
            free_bodies.append(body.model_instance())
    return free_bodies