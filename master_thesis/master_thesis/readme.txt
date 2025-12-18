Be sure to add the root folder (the one with this readme) of the master thesis to the python path, using either .bashrc or import sys; sys.path.append(path_to_root) inside the python program themselves.
The included dockerfile might not work or not include all the necessary libraries, but it should provide a general view of most necessary libraries. 
To run the scenarios the following scripts/programs must be run:

python3 realsense_1.py
python3 realsense_2.py
camera_processing.ipynb
svm_learning_deterministic.ipynb / svm_learning_prob.ipynb (this outputs the support vectors and the weights, from the perspective of the control algorithm (next scripts) nothing changes, so all scenarios can be ran with different prob. parameters/deterministic without editing the control script)
talker.ipynb - defines the end and starting position

Uncertain Placing Spot Scenario: scenario_vision.ipynb
Dynamic Movement Scenario: scenario_3.ipynb
Varying Threshold Scenario: scenario_3.ipynb 
Probabilistic Algorithm Performance in Known Goal Scenario: scenario_vision.ipynb (by editing out the if/else regarding the apriltag)

A simulation can be ran by running simulation plant_simulation.ipynb and changing the simulation to true in the camera_processing script.

Each scenario might have a slightly different MPC formulation, parameters can be changed and the MPC can be rebuilt by switching the rebuild flag in the control scripts. If rebuild is false, it will simply load the previous .so files.
In general, the control scripts work as follows:
- the MPC is loaded/generated
- some initial trajectories are generated, this means solving the MPC starting from multiple initial guesses (based on a combination of all the possible the inverse kinematics from the starting position) , usually first ignoring collision and then mildly desconsidering it and then fully solving the proper problem (by selecting one of the previous solves and resolving it considering the latest collision info). This can be shortcut by simply saving one of the previous solves and reloading. One of my previous scripts has an automated procedure of generating multiple trajectories and selecting the one that best  fits the current initial/final pose reference (I changed a couple things afterwards and never updated this script, but that's the rough idea).
- the previously generated trajectory is used as the first initial solve of the MPC loop.

The svm_learning_prob and the scenario_vision programs have instructions inside them to show what cells to run. The other programs follow the same logic.

If something seems off do:
pfkill -f "python3"
and start all the programs again
Possible reasons: 
- some process (learning, camera processing) died in the background
- the interprocess shared memory got messed up.
If everything turns slow, it might be because the spawned processes didn't die, so run pfkill -f "from multiprocessing" to kill only them specifically.

- Programs:
-- camera_processing.ipynb
This script reads the data from the cameras and the data from the robots and uses that for the processing script. The processing program is defined in diff_co_mpc/point_cloud/. The PreprocessPointCloud system gets the data from the cameras and sends to a second process, the point cloud worker that is continously working in the backgroud. Data is sent to it and it processes it in a non blocking manner. When it is ready, it sends a message back and the main program reads the processed PC.

The algorithm described in the thesis is implemented in the point_cloud_worker() function, in point_cloud_processing.py.

-- svm_learning*
The main parts are defined in diff_co_mpc/diff_co. The source folder contains the perceptron learning algorithm in cpp. In learning.py the rest of the learning algorithm is defined as explained in the thesis. Each robot has a support vector worker and each of those has 4 group workers, each of those groups learns the support vector and the weights correspondent exclusively to the links as defined in the config.yaml. Each group worker is one instance of the algorithm as described in the thesis.
All of that is running using multiprocessing, so each worker is running in parallel.

-- control algorithm
The MPC and helper functions are defined in the mpc folder. In the optimisation.py the OptimizationInfo class contains the MPC functions and in the planner_implementation/make.py those functions are called to define the MPC itself. The optimisation variables and parameters are declared in the setup. 

The OptimizationInfo class has a bunch of functions that define the constraints and costs in such a way that Casadi is decently behaved. In addition, the MPC trajectory is defined in such a way that the constraints are applied at samples of the trajectory c(s_k), where c(s_k) is a function of the control points (dec. variables), to make Casadi behavel well, it is also necessary to sort of "manually"apply the chain rule to the constraint function and the inner function c(s_k). Those inner functions are defined in the class as well. The chain rule applied as a map to all the samples is done using the functions in custom_casadi.py. Finally, all the constraints and costs derivatives are put together using get_final_hessian and get_final_jacobian, that gets all those custom defined casadi functions and put them together to make the NLP work.

OptimizationData is a helper class to streamline setting the parameters, solving and getting the results. 
