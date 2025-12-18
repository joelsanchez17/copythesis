from collections import namedtuple
from pydrake.all import (
    FindResourceOrThrow, 
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    RigidTransform, 
    FixedOffsetFrame,
    RollPitchYaw,
    ColorRenderCamera,
    DepthRenderCamera,
    SpatialInertia,
    RgbdSensor,
    ColorizeDepthImage,
    Sphere,
    CoulombFriction,
    ColorizeLabelImage,
    UnitInertia,
    DepthImageToPointCloud,
    BaseField,
    PixelType,
    AbstractValue,
    LeafSystem,
    LightParameter, 
    Rgba,
    RenderEngineVtkParams,
    MeshcatPointCloudVisualizer,
    RenderCameraCore, CameraInfo, ClippingRange, DepthRange, MakeRenderEngineVtk
      )
from pydrake.common import FindResourceOrThrow
import sys
from pathlib import Path
sys.path.insert(0, str(Path("~/Projects/Dual_Camera_Perception_Module/master_thesis/master_thesis/drake").expanduser()))
from manipulation.scenarios import AddMultibodyTriad
import yaml
from utils.my_drake.misc import ObjectFactory,add_default_meshcat_visualization
import numpy as np

def make_complete_plant(base_transforms:list,obstacles:list, meshcat, camera_params:dict = None):
    """
    ```
        return {
            'plant':plant,
            'diagram':diagram,
            'visualizer':visualizer,
            'obstacle_frames':obstacle_frames,
            'obstacle_bodies':obstacle_bodies,
            'robot_opt_tuple':robot_opt_tuple,
            'carried_object_opt':namedtuple('carried_object_opt',['diagram','plant','viz_model_instance'])(diagram,plant, plant.GetModelInstanceByName(obj['name']) ),

        }

    ```
    """
    if camera_params:
        for camera_dict in camera_params:            
            if 'point_cloud_visualizer' not in camera_dict:
                camera_dict['point_cloud_visualizer'] = False
            assert 'core' in camera_dict
            assert 'depth_range' in camera_dict
            assert 'renderer' in camera_dict
            assert 'name' in camera_dict
            assert 'pose' in camera_dict
            assert 'attached_to' in camera_dict
        
            
    # franka_robot_urdf = FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm.urdf")
    # franka_robot_urdf = "package://drake_models/franka_description/urdf/panda_arm.urdf"
    # franka_robot_urdf = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
    # franka_robot_urdf = "/home/sysgen/Projects/Dual_Camera_Perception_Module/master_thesis/master_thesis/models/franka_description/urdf/panda_arm_hand.urdf"
    franka_robot_urdf = "/home/sysgen/Projects/Dual_Camera_Perception_Module/master_thesis/master_thesis/model2/franka_panda_description/robots/panda_arm_hand.urdf"
    # franka_robot_urdf = "/home/sysgen/Projects/Dual_Camera_Perception_Module/master_thesis/master_thesis/model3/PandaRobot.jl/deps/Panda/panda.urdf"
    # franka_robot_urdf = "package://franka_description/urdf/panda_arm_hand.urdf"
    import os
    assert os.path.exists(franka_robot_urdf), f"URDF file not found at: {franka_robot_urdf}"

    # PackageMap().GetPath("drake_models")
# (robot,) = Parser(plant).AddModelsFromUrl("package://drake_models/franka_description/urdf/panda_arm.urdf")
    EE_frame_transform = RigidTransform(p = [0,0,0.1034],rpy= RollPitchYaw([0,0,-np.pi/4]))
    EE_frame_offset_transform = RigidTransform(p = [0,0.05,-0.075])
    panda_link6_transform = RigidTransform(p = [0,0,-0.1])
    def make_one_robot_plant(base_transform:RigidTransform):
        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)
        parser = Parser(plant)
        parser.package_map().Add(
            "franka_panda_description",
            "/home/sysgen/Projects/Dual_Camera_Perception_Module/master_thesis/master_thesis/model2/franka_panda_description"
        )
        
        robot= Parser(plant).AddModelFromFile(franka_robot_urdf)
        plant.RenameModelInstance(robot,"robot")
        world_frame = plant.world_frame()
        
        
        plant.WeldFrames(world_frame, plant.GetFrameByName('panda_link0',model_instance=robot), base_transform)
        f = plant.GetFrameByName('panda_link8')
        EE_frame= plant.AddFrame(FixedOffsetFrame('EE_frame',f, X_PF=EE_frame_transform))
        plant.AddFrame(FixedOffsetFrame('EE_frame_offset',plant.GetFrameByName('EE_frame'), X_PF=EE_frame_offset_transform))
        plant.AddFrame(FixedOffsetFrame('panda_link6_offset',plant.GetFrameByName('panda_link6'), X_PF=panda_link6_transform))
        shape = Sphere(radius = 0.020)
        properties = CoulombFriction()
        num_geometries = 5
        for i in range(num_geometries):
            name = "object_collision_"+str(i)
            X_BG = RigidTransform(p = [-0.03*(i),0.0,0.1])
            plant.RegisterCollisionGeometry(plant.GetBodyByName('panda_hand'), X_BG,shape,name,properties)    
        plant.RegisterCollisionGeometry(plant.GetBodyByName('panda_link8'), RigidTransform(),shape,'panda_link8_extra',properties)
        plant.Finalize()
        
        #flag
        for body in plant.GetBodyIndices(plant.GetModelInstanceByName("robot")):
            print(body, plant.get_body(body).name())



        diagram = builder.Build()
        return diagram, plant
    def make_carried_object_plant(obj):
        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)

        body,frame,_,_ = ObjectFactory.object_from_name(**obj, plant = plant,scene_graph =scene_graph)     

        carried_object_EE_1_frame = plant.AddFrame(frame = FixedOffsetFrame(
                                                    name = f"carried_object_EE_1_frame",
                                                    P = frame,
                                                    X_PF = obj['transform_object_to_EE_1']))
        
        carried_object_EE_2_frame = plant.AddFrame(frame = FixedOffsetFrame(
                                                    name = f"carried_object_EE_2_frame",
                                                    P = frame,
                                                    X_PF = obj['transform_object_to_EE_2']))

        world_frame = plant.world_frame()

        plant.Finalize()
        diagram = builder.Build()
        return diagram, plant
    
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    obstacle_frames = []
    obstacle_bodies = []
    for i, obstacle in enumerate(obstacles):
        if 'name' not in obstacle:
            obstacle['name'] = 'obstacle_{}'.format(str(i))
            
        body,frame,_,_ = ObjectFactory.object_from_name(**obstacle,plant=plant,scene_graph = scene_graph)
        
        obstacle_bodies.append(body)
        obstacle_frame = plant.AddFrame(frame = FixedOffsetFrame(
                                                                name = "obstacle_{}_frame".format(obstacle['name']),
                                                                P = plant.world_frame(),
                                                                X_PF = RigidTransform.Identity()))
        obstacle_frames.append(obstacle_frame)
        

    
    obj_size = 0.4
    transform_object_to_EE_1 = RigidTransform(p=[obj_size/2, 0.0, 0.0],rpy=RollPitchYaw([np.pi,0,np.pi]))
    transform_object_to_EE_2 = RigidTransform(p=[-obj_size/2, 0.0, 0.0],rpy=RollPitchYaw([np.pi,0,np.pi]))
    obj = {'shape':'box',
           'width':obj_size,
           'height':0.05,
           'depth':0.05,
           'name':'carried_object',
           'transform_object_to_EE_1':transform_object_to_EE_1,
           'transform_object_to_EE_2':transform_object_to_EE_2
           }
    

            
    robot_opt_tuple = {}
    robot_instances = []
    for i, base_transform in enumerate(base_transforms):
        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)
        parser = Parser(plant)
        parser.package_map().Add(
            "franka_panda_description",
            "/home/sysgen/Projects/Dual_Camera_Perception_Module/master_thesis/master_thesis/model2/franka_panda_description"
        )
        robot = Parser(plant).AddModelFromFile(franka_robot_urdf)
        plant.RenameModelInstance(robot,f"robot_{i}")
        world_frame = plant.world_frame()
        plant.WeldFrames(world_frame, plant.GetFrameByName('panda_link0',model_instance=robot), base_transform)
        robot_only_diagram, robot_only = make_one_robot_plant(base_transform)
        robot_opt_tuple[i] = namedtuple('robot_opt',['diagram','plant','viz_model_instance'])(robot_only_diagram,robot_only, robot)
        f = plant.GetFrameByName('panda_link8',model_instance=robot)
        EE_frame = plant.AddFrame(FixedOffsetFrame('EE_frame',f, X_PF=EE_frame_transform))
        AddMultibodyTriad(EE_frame, scene_graph)
        # robot_instances.append(robot)

    body,frame,_,_ = ObjectFactory.object_from_name(**obj, plant = plant,scene_graph =scene_graph)     

    carried_object_EE_1_frame = plant.AddFrame(frame = FixedOffsetFrame(
                                                name = f"carried_object_EE_1_frame",
                                                P = frame,
                                                X_PF = obj['transform_object_to_EE_1']))
    
    carried_object_EE_2_frame = plant.AddFrame(frame = FixedOffsetFrame(
                                                name = f"carried_object_EE_2_frame",
                                                P = frame,
                                                X_PF = obj['transform_object_to_EE_2']))
    
    cameras = None
    if camera_params:
        cameras = []
        for camera_dict in camera_params:
            camera = Camera(builder,plant,scene_graph,camera_dict)
            if camera_dict['point_cloud_visualizer']:
                pc_visualizer = MeshcatPointCloudVisualizer(meshcat, "/point_cloud/",publish_period=0.1)
                builder.AddNamedSystem('point_cloud_visualizer' + camera_dict['name'],pc_visualizer)
                builder.Connect(camera._depth_to_point_cloud.point_cloud_output_port(),pc_visualizer.get_input_port(0))
            cameras.append(camera)

    plant.Finalize()
    if camera_params:
        for camera in cameras:
            builder.Connect(plant.get_body_poses_output_port(),camera._extract_pose.get_input_port())
        
    meshcat.Delete()

    visualizer = add_default_meshcat_visualization(builder,meshcat)

    diagram = builder.Build()
    return {
        'plant':plant,
        'diagram':diagram,
        'visualizer':visualizer,
        'obstacle_frames':obstacle_frames,
        'obstacle_bodies':obstacle_bodies,
        'robot_opt_tuple':robot_opt_tuple,
        'camera':cameras if camera_params else None,
        'carried_object_opt':namedtuple('carried_object_opt',['diagram','plant','viz_model_instance'])(*make_carried_object_plant(obj), plant.GetModelInstanceByName(obj['name']) ),

    }

class OnePosePublish(LeafSystem):    
    def __init__(self, index):
        super().__init__()
        self._index = index
        self._output_port = self.DeclareAbstractOutputPort(
            "geometry_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.body_pose,
        )     
        self._input_port = self.DeclareAbstractInputPort("poses", AbstractValue.Make([RigidTransform()]))
    def body_pose(self, context, output:AbstractValue):
        output.set_value(self._input_port.Eval(context,)[self._index])
class Camera:
    def __init__(self, builder, plant, scene_graph, camera_params):
        render_camera_core = camera_params['core']
        depth_range = camera_params['depth_range']
        name = camera_params['name']
        renderer = camera_params['renderer']
        self._plant = plant
        self._scene_graph = scene_graph
        self._name = name
        self._color_camera = ColorRenderCamera(render_camera_core, show_window=False)
        self._depth_camera = DepthRenderCamera(render_camera_core, depth_range)
        self._core = render_camera_core
        try:
            self._scene_graph.AddRenderer(
                self._core.renderer_name(), renderer)
        except:
            pass
        
        self._body = plant.AddRigidBody(name = f"{name}_camera_body", M_BBo_B=SpatialInertia(mass = 1,p_PScm_E = np.array([0.0, 0.0, 0.0]), G_SP_E = UnitInertia.SolidCylinder(1, 1,unit_vector=[0,0,1])),model_instance = plant.GetModelInstanceByName("DefaultModelInstance"))

        self._frame = plant.AddFrame(frame=FixedOffsetFrame(
                name=f"{camera_params['name']}_frame",
                P=plant.GetFrameByName(camera_params['attached_to']['frame_name'],plant.GetModelInstanceByName(camera_params['attached_to']['model_instance'])) if camera_params['attached_to']['frame_name'] != 'world_frame' else plant.world_frame(),
                X_PF=camera_params['pose'],))
        
        # we can change the position of FixedOffsetFrames directly, not of body frames
        self._plant.WeldFrames(self._frame, self._body.body_frame(), RigidTransform())
        
        self._sensor = RgbdSensor(
            plant.GetBodyFrameIdOrThrow(self._body.index()),
            X_PB=RigidTransform.Identity(),
            color_camera=self._color_camera,
            depth_camera=self._depth_camera,
        )
        builder.AddSystem(self._sensor)
        builder.Connect(
            scene_graph.get_query_output_port(),
            self._sensor.query_object_input_port(),
        )
        self._depth_to_point_cloud = builder.AddNamedSystem('depth_to_point_cloud_' + self._name,DepthImageToPointCloud(camera_info=self._core.intrinsics(), fields= BaseField.kXYZs , pixel_type=PixelType.kDepth32F))
        self._extract_pose = builder.AddSystem(OnePosePublish(self._body.index()))
        
        builder.Connect(self._extract_pose.get_output_port(),self._depth_to_point_cloud.camera_pose_input_port())
        # builder.Connect(self._sensor.GetOutputPort("depth_image_16u"),self._depth_to_point_cloud.depth_image_input_port())
        builder.Connect(self._sensor.GetOutputPort("depth_image_32f"),self._depth_to_point_cloud.depth_image_input_port())
    def set_transform(self,plant_context,transform:RigidTransform):
        self._frame.SetPoseInParentFrame(plant_context,transform)
    def get_transform(self,plant_context):
        return self._frame.GetPoseInParentFrame(plant_context)

def plant_from_yaml(file_path, camera:bool, meshcat, point_cloud_visualizer:bool = False):
    with open(file_path, 'r') as file:
        try:
            # Load the YAML content
            data = yaml.safe_load(file)
            transform_1 = RigidTransform(p = data['transforms']['robot_0']['translation'], rpy = RollPitchYaw(data['transforms']['robot_0']['rpy']))
            transform_2 = RigidTransform(p = data['transforms']['robot_1']['translation'], rpy = RollPitchYaw(data['transforms']['robot_1']['rpy']))
            if 'obstacles' in data:
                 obstacles = [{'name': name, **value} for name, value in data['obstacles'].items()]
                # obstacles = [{'name': name,} | value for name,value in data['obstacles'].items()]
            else:
                obstacles = []
            camera_params = None
            if camera:
                camera_params = []
                for key,params in data['sim_camera_params'].items():
                    camera_params_dict = params
                    intrinsics = CameraInfo(
                        width=camera_params_dict['width'],
                        height=camera_params_dict['height'],
                        fov_y=camera_params_dict['fov_y'],
                    )
                    # https://drake.mit.edu/pydrake/pydrake.geometry.html?highlight=rendercameracore#pydrake.geometry.RenderCameraCore
                    core = RenderCameraCore(
                        camera_params_dict['renderer']['name'],
                        intrinsics,
                        ClippingRange(camera_params_dict['clipping_range']['near'], camera_params_dict['clipping_range']['far']),
                        RigidTransform(),
                    )
                    depth_range = DepthRange(camera_params_dict['depth_range']['min'], camera_params_dict['depth_range']['max'])
                    renderer = MakeRenderEngineVtk(RenderEngineVtkParams(lights = [LightParameter(color=Rgba(*camera_params_dict['renderer']['lights']['color']), )]))
                    pose = RigidTransform(p = camera_params_dict['pose']['translation'], rpy = RollPitchYaw(camera_params_dict['pose']['rpy']))
                    attached_to_frame = camera_params_dict['attached_to']
                    if point_cloud_visualizer:
                        point_cloud_visualizer_obj = MeshcatPointCloudVisualizer(meshcat, path = '/point_cloud/', publish_period=0.1)
                    else:
                        point_cloud_visualizer_obj = False
                    camera_params.append({'core':core,
                                     'renderer':renderer,
                                     'depth_range':depth_range, 
                                     'name':camera_params_dict['name'],
                                     'pose':pose,
                                     'attached_to': attached_to_frame,
                                     'point_cloud_visualizer':point_cloud_visualizer_obj
                                     })


            plants_info = make_complete_plant([transform_1,transform_2],obstacles,meshcat, camera_params)
            return {'plants_info':plants_info, 'yaml':data, 'camera_params':camera_params}
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None
