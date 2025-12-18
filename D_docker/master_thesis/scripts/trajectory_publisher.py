import sys

sys.path += [".."]
from numpy import loadtxt
import sys
import rospy as ros
import numpy as np
# from gripper_operations import GripperOperations

from actionlib import SimpleActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, \
                             FollowJointTrajectoryGoal, FollowJointTrajectoryResult
# from controller_chen.msg import PoseBatch
# from controller_chen.srv import TriggerTrajectoryPublish
from utils.math.BSpline import BSpline


# data = loadtxt('/workspaces/toys/projects/thesis/cooperative_diff_co_mpc/8/control_points_1.txt', delimiter=' ')
# # print(data)
# bspline_1 = BSpline(data,3)
# bspline_1_dot = bspline_1.create_derivative_spline()
# print(bspline_1.evaluate(1))
# afdsdsf

import rospy as ros
ros.init_node('effort_joint_trajectory_controller')

from std_msgs.msg import String

rate = ros.Rate(10) 
action = ros.resolve_name('/panda_1/effort_joint_trajectory_controller/follow_joint_trajectory')
client = SimpleActionClient(action, FollowJointTrajectoryAction)
ros.loginfo("move_for_trajectory: Waiting for '" + action + "' action to come up")
client.wait_for_server()

topic = ros.resolve_name('/panda_1/franka_state_controller/joint_states')
ros.loginfo("move_for_trajectory: Waiting for message on topic '" + topic + "'")
joint_state = ros.wait_for_message(topic, JointState)
initial_pose = dict(zip(joint_state.name, joint_state.position))

joint_names = joint_state.name

# initial_q = bspline_1.evaluate(0)
# print(bspline_1.evaluate(0))
# print(bspline_1_dot.evaluate(0))
# asdfasdfdfs
print(initial_pose)
print(joint_names)
point = JointTrajectoryPoint()

point.time_from_start = ros.Duration.from_sec(
    # Use either the time to move the furthest joint with 'max_dq' or 500ms,
    # whatever is greater
    4
)
goal = FollowJointTrajectoryGoal()


initial_q_1 = np.array([ 0.55, -0.2620533 , -0.4839102 , -1.70406773,  0.33930883,
        1.82823052,  1.56379101,  0.        ,  0.        ])[:7]
goal.trajectory.joint_names = joint_names

initial_q_2 = np.array([ 1.55, -1.03936345,  0.08177012, -2.49036332,  0.03187163,
        1.38190466,  0.73227313,  0.        ,  0.        ])[:7]

point.positions[:7] = initial_q_1
# point.positions[9:] = initial_q_2
point.velocities = [0] * len(initial_q_1)


goal.trajectory.points.append(point)
goal.goal_time_tolerance = ros.Duration.from_sec(0.5)


ros.loginfo('Sending trajectory Goal to move into initial config')
client.send_goal_and_wait(goal)



# initial_q = bspline_1.evaluate(0)
# print(initial_pose)

# total_time = 10
# n = 400
# goal = FollowJointTrajectoryGoal()
# goal.trajectory.joint_names = joint_names
# for s in np.linspace(0,1,n):
#     point = JointTrajectoryPoint()
#     q = bspline_1.evaluate(s)
#     q_dot = bspline_1_dot.evaluate(s)/total_time
#     print(q,q_dot)
#     point.time_from_start = ros.Duration.from_sec(
#         # Use either the time to move the furthest joint with 'max_dq' or 500ms,
#         # whatever is greater
#         s*total_time
#     )
#     point.positions = q
#     point.velocities =  q_dot
#     goal.trajectory.points.append(point)
#     goal.goal_time_tolerance = ros.Duration.from_sec(0.5)


# client.send_goal_and_wait(goal)