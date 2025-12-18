-turn on the robots and FCI 
- connect the cameras with this serial num that is mentioned in the code (tip: when you connecting the camera, while the container is running, you must reopen the container)
run the launch file at this location:
location: catkin_frankaROS/src/franka_ros/controller_chen/launch ernesto2.launch
1- run ros master: roscore
2- source the workspace 
    cd catkin_frankaROS
    source devel/setup.bash
3- launch ernesto2:

run the docker:
1- in this directory : cd Projects/Dual_Camera_Perception_Module/D_docker/master_thesis
    then: code .
2- a window is pop up on the down right side that want to open up the Dev container, click on it(you can also bring it up by Ctrl+Shift+p then finding the dev container)
3- now running the scripts:
    - running the realsense_1 and 2 
        - open two terminal at the container 
        - first deactivate the environment then change directory to scripts 
        - then python3 realsense_1.py  and 2
    - run talker the whole notebook(run all)
    - run run the first four cells of camera_processing, then click on the local host link to see the visualization on meshcat 
    - run svm_learning_determinestic whole (run all)
    - run scenario_vision whole notebook(run all)
     

camera labels
    -10 fixed 
    - eye on the end-effector is fixed


#### things need to be installed after crash ####
-ros1
-numba
-concave_hull
-nvidia-smi
-glxinfo -B
-pyrealsense2


### the causes of some incompatibility ####
-sudo pip uninstall numpy
-sudo pip install numpy==1.26.4
-pip install --upgrade coverage