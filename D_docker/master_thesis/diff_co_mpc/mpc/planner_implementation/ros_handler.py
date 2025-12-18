try:
    import sys
    sys.path += ["/opt/ros/noetic/lib/python3/dist-packages"]
    from actionlib import SimpleActionClient
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.msg import FollowJointTrajectoryAction, \
                                FollowJointTrajectoryGoal, FollowJointTrajectoryResult
    import rospy as ros
    from actionlib_msgs.msg import GoalStatus
except:
    print('ROS not available')
import numpy as np
from collections import deque
from mpc.helper_functions import integrate_q_trapezoidal
from utils.math.BSpline import BSpline
from utils.my_drake.misc import VisualizerHelper
class ROSHandler:
    def __init__(self, viz_helper:VisualizerHelper, max_store_seconds = 100):
        # global ROS_INITIALIZED
        # try:
        #     ROS_INITIALIZED 
        # except:
        #     ros.init_node('ros_handler',anonymous=True)
        #     ROS_INITIALIZED = True
        self.simulation = False
        self.viz_helper = viz_helper
        self.simulation_time = 0
        self.alpha = 0.2
        ros.Subscriber("/panda_1/franka_state_controller/joint_states", JointState, self.ros_callback_joints_1)
        ros.Subscriber("/panda_1/franka_gripper/joint_states", JointState, self.ros_callback_gripper_1 )
        ros.Subscriber("/panda_2/franka_state_controller/joint_states", JointState, self.ros_callback_joints_2)
        ros.Subscriber("/panda_2/franka_gripper/joint_states", JointState, self.ros_callback_gripper_2 )
        self.joints_1 = np.array([0.]*9)
        self.joints_2 = np.array([0.]*9)
        self.joints_dot_1 = np.array([0.]*7)
        self.joints_dot_2 = np.array([0.]*7)
        self.joints_dot_1_filtered = np.array([0.]*7)
        self.joints_dot_2_filtered = np.array([0.]*7)

        self.topic_frequency = 30
        max_size = max_store_seconds*self.topic_frequency
        self.joints_1_history = deque(maxlen=max_size)  
        self.joints_2_history = deque(maxlen=max_size)  
        self.joints_dot_1_history = deque(maxlen=max_size)  
        self.joints_dot_2_history = deque(maxlen=max_size)  
        self.joints_dot_1_filtered_history = deque(maxlen=max_size)  
        self.joints_dot_2_filtered_history = deque(maxlen=max_size)  

        self.joint_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,])
        self.joint_upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,])
        self.joint_vel_lower_limits = np.array([-2.175, -2.175, -2.175, -2.175, -2.61 , -2.61 , -2.61 ,])
        self.joint_vel_upper_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.61 , 2.61 , 2.61 ,])
        self.joint_acc_lower_limits = np.array([-15. ,  -7.5, -10. , -12.5, -15. , -20. , -20. ,  ])
        self.joint_acc_upper_limits = np.array([15. ,  7.5, 10. , 12.5, 15. , 20. , 20. , ])
        self.minimum_time = 20
        self.JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.setup_action_servers()
    def set_simulation(self,simulation):
        self.simulation = simulation

    def reset_simulation_time(self):
        self.simulation_time = 0

    def move_robots_to_configuration(self,q_1 = None,q_2 = None,time = 5):

        assert time>= self.minimum_time
        if q_1 is not None:
            point = JointTrajectoryPoint()

            point.time_from_start = ros.Duration.from_sec(
                    time
                )
            goal_1 = FollowJointTrajectoryGoal()
            goal_1.trajectory.joint_names = self.JOINT_NAMES
            point.positions = q_1
            point.velocities = [0] * len(self.JOINT_NAMES)
            goal_1.trajectory.points.append(point)
            goal_1.goal_time_tolerance = ros.Duration.from_sec(0.5)
        if q_2 is not None:
            point = JointTrajectoryPoint()

            point.time_from_start = ros.Duration.from_sec(
                    time
                )
            goal_2 = FollowJointTrajectoryGoal()
            goal_2.trajectory.joint_names = self.JOINT_NAMES
            point.positions = q_2
            point.velocities = [0] * len(self.JOINT_NAMES)
            goal_2.trajectory.points.append(point)
            goal_2.goal_time_tolerance = ros.Duration.from_sec(0.5)

        if q_1 is not None:
            ros.loginfo(f'Sending trajectory Goal to move robots to:\nq_1 = {q_1}',)
            self.client_1.send_goal(goal_1)
        if q_2 is not None:
            ros.loginfo(f'Sending trajectory Goal to move robots to:\nq_2 = {q_2}',)
            self.client_2.send_goal(goal_2)
        try:
            
            if q_1 is not None:
                self.client_1.wait_for_result()
            if q_2 is not None:
                self.client_2.wait_for_result()
        except:
            self.client_1.cancel_all_goals()
            self.client_2.cancel_all_goals()
    def cancel_current_goal(self):
        self.client_1.cancel_all_goals()
        self.client_2.cancel_all_goals()
    def setup_action_servers(self):
        self.action_1 = ros.resolve_name('/panda_1/effort_joint_trajectory_controller/follow_joint_trajectory')
        self.client_1 = SimpleActionClient(self.action_1, FollowJointTrajectoryAction)
        ros.loginfo("Waiting for '" + self.action_1 + "' action to come up")
        self.action_2 = ros.resolve_name('/panda_2/effort_joint_trajectory_controller/follow_joint_trajectory')
        self.client_2 = SimpleActionClient(self.action_2, FollowJointTrajectoryAction)
        ros.loginfo("Waiting for '" + self.action_2 + "' action to come up")
        # self.client_1.wait_for_server()
        # self.client_2.wait_for_server()

    def low_pass_filter(self, new_value, prev_filtered):
        return self.alpha * new_value + (1 - self.alpha) * prev_filtered
    def ros_callback_gripper_1(self,msg):
        self.joints_1[-2:]= msg.position

    def ros_callback_joints_1(self,msg):
        self.joints_1[:7]= msg.position
        self.joints_dot_1[:] = msg.velocity
        self.joints_dot_1_filtered[:] = self.low_pass_filter(self.joints_dot_1,self.joints_dot_1_filtered)
        self.joints_1_history.append(msg.position)
        self.joints_dot_1_history.append(msg.velocity)
        self.joints_dot_1_filtered_history.append(self.joints_dot_1_filtered.copy())
    def ros_callback_gripper_2(self,msg):
        self.joints_2[-2:]= msg.position
    def ros_callback_joints_2(self,msg):
        self.joints_2[:7]= msg.position
        self.joints_dot_2[:] = msg.velocity
        self.joints_dot_2_filtered[:] = self.low_pass_filter(self.joints_dot_2,self.joints_dot_2_filtered)
        self.joints_2_history.append(msg.position)
        self.joints_dot_2_history.append(msg.velocity)
        self.joints_dot_2_filtered_history.append(self.joints_dot_2_filtered.copy())
    def get_state(self):
        return self.client_1.get_status(), self.client_2.get_status()
    def goal_is_active(self):
        if self.simulation:
            return True
        return self.client_1.get_state() == GoalStatus.ACTIVE or self.client_2.get_state() == GoalStatus.ACTIVE
    def follow_pointwise_trajectory(self,q_1s,q_2s,q_1_dots,q_2_dots, time):
        if self.simulation:
            for q_1,q_1_dot,q_2,q_2_dot,t  in zip(q_1s,q_2_dots,q_1s,q_2_dots,np.linspace(0,time,num_steps)):
                self.viz_helper.set_position('robot_1',q_1)
                self.viz_helper.set_position('robot_2',q_2)
                self.viz_helper.diagram_context.SetTime(self.simulation_time + t)
                self.viz_helper.publish()
            self.simulation_time += time
            return 
        goal_1 = FollowJointTrajectoryGoal()
        goal_1.trajectory.joint_names = self.JOINT_NAMES
        num_steps = len(q_1s)
        tol = ros.Duration.from_sec(time/num_steps)
        for q_1,q_1_dot,t  in zip(q_1s,q_1_dots,np.linspace(0,time,num_steps)):
            msg_point = JointTrajectoryPoint()
            msg_point.positions = q_1
            msg_point.velocities = q_1_dot
            msg_point.time_from_start = ros.Duration.from_sec(t)
            goal_1.trajectory.points.append(msg_point)
            goal_1.goal_time_tolerance = tol
        goal_2 = FollowJointTrajectoryGoal()
        goal_2.trajectory.joint_names = self.JOINT_NAMES
        for q_2,q_2_dot,t  in zip(q_2s,q_2_dots,np.linspace(0,time,num_steps)):
            msg_point = JointTrajectoryPoint()
            msg_point.positions = q_2
            msg_point.velocities = q_2_dot
            msg_point.time_from_start = ros.Duration.from_sec(t)
            goal_2.trajectory.points.append(msg_point)
            goal_2.goal_time_tolerance = tol
        self.client_1.send_goal(goal_1)
        self.client_2.send_goal(goal_2)
    def follow_bspline_no_wait(self,bspline_1:BSpline,bspline_2:BSpline,time,num_steps,reverse = False):
        assert time>= self.minimum_time
        assert not reverse, "unimplemented"
        if bspline_1 is not None:
            bspline_1_dot = bspline_1.create_derivative_spline()
            goal_1 = FollowJointTrajectoryGoal()
            goal_1.trajectory.joint_names = self.JOINT_NAMES
            q_dots = bspline_1_dot.fast_batch_evaluate(np.linspace(0,1,num_steps))/time
            q_dots = np.clip(q_dots,self.joint_vel_lower_limits,self.joint_vel_upper_limits)
            qs = bspline_1.fast_batch_evaluate(np.linspace(0,1,num_steps))
            qs = np.clip(qs,self.joint_lower_limits,self.joint_upper_limits)
            tol = ros.Duration.from_sec(time/num_steps)
            assert np.allclose(qs[0],self.joints_1[:7] ,atol= 0.1,rtol=0.01)
            for q,q_dot,t in zip(qs,q_dots,np.linspace(0,time,num_steps)):
                point = JointTrajectoryPoint()
                
                point.time_from_start = ros.Duration.from_sec(
                    t
                )
                point.positions = q
                point.velocities =  q_dot
                goal_1.trajectory.points.append(point)
                goal_1.goal_time_tolerance = tol
        if bspline_2 is not None:
            bspline_2_dot = bspline_2.create_derivative_spline()
            goal_2 = FollowJointTrajectoryGoal()
            goal_2.trajectory.joint_names = self.JOINT_NAMES
            q_dots = bspline_2_dot.fast_batch_evaluate(np.linspace(0,1,num_steps))/time
            q_dots = np.clip(q_dots,self.joint_vel_lower_limits,self.joint_vel_upper_limits)
            qs = bspline_2.fast_batch_evaluate(np.linspace(0,1,num_steps))
            qs = np.clip(qs,self.joint_lower_limits,self.joint_upper_limits)
            tol = ros.Duration.from_sec(time/num_steps)
            assert np.allclose(qs[0],self.joints_2[:7] ,atol= 0.1,rtol=0.01)
            for q,q_dot,t in zip(qs,q_dots,np.linspace(0,time,num_steps)):
                point = JointTrajectoryPoint()                
                point.time_from_start = ros.Duration.from_sec(
                    t
                )
                point.positions = q
                point.velocities =  q_dot
                goal_2.trajectory.points.append(point)
                goal_2.goal_time_tolerance = tol
            # assert np.allclose(goal_2.trajectory.points[0].positions,self.joints_2[:7] ,atol= 0.05,rtol=0.01)
        if bspline_1 is not None:
            ros.loginfo(f'Robot 1 following trajectory',)
            self.client_1.send_goal(goal_1)
        if bspline_2 is not None:
            ros.loginfo(f'Robot 2 following trajectory',)
            self.client_2.send_goal(goal_2)
            
    def follow_derivative_bspline_no_wait(self,bspline_1 = None,bspline_2 = None,time = 5,num_steps = 10000, s_start = 0):
        assert time>= self.minimum_time
        
        dt = time/num_steps
        if bspline_1 is not None:
            
            goal_1 = FollowJointTrajectoryGoal()
            goal_1.trajectory.joint_names = self.JOINT_NAMES
            q  = self.joints_1[:7].copy()
            q_dots = evaluate_bspline(np.linspace(s_start,1,num_steps), bspline_1.control_points, bspline_1.knots,bspline_1.order)/time
            qs = (integrate_q_trapezoidal(q, q_dots, dt))[::50]
            for q,q_dot,t in zip(qs,q_dots[::50],np.linspace(s_start*time,time,qs.shape[0])):
                point = JointTrajectoryPoint()
                point.time_from_start = ros.Duration.from_sec(
                    t
                )
                
                q = np.clip(q,self.joint_lower_limits,self.joint_upper_limits)
                q_dot = np.clip(q_dot,self.joint_vel_lower_limits,self.joint_vel_upper_limits)
                point.positions = q
                point.velocities =  q_dot
                goal_1.trajectory.points.append(point)
                goal_1.goal_time_tolerance = ros.Duration.from_sec(
                    0.5
                )
            assert np.allclose(goal_1.trajectory.points[0].positions,self.joints_1[:7] ,atol= 0.05,rtol=0.01)
        if bspline_2 is not None:
            
            goal_2 = FollowJointTrajectoryGoal()
            goal_2.trajectory.joint_names = self.JOINT_NAMES
            q  = self.joints_2[:7].copy()
            q_dots = evaluate_bspline(np.linspace(s_start,1,num_steps), bspline_2.control_points, bspline_2.knots,bspline_2.order)/time
            qs = (integrate_q_trapezoidal(q, q_dots, dt))[::50]
            for q,q_dot,t in zip(qs,q_dots[::50],np.linspace(s_start*time,time,qs.shape[0])):
                point = JointTrajectoryPoint()
                point.time_from_start = ros.Duration.from_sec(
                    t
                )
                q = np.clip(q,self.joint_lower_limits,self.joint_upper_limits)
                q_dot = np.clip(q_dot,self.joint_vel_lower_limits,self.joint_vel_upper_limits)
                
                point.positions = q
                point.velocities =  q_dot
                goal_2.trajectory.points.append(point)
                goal_2.goal_time_tolerance = ros.Duration.from_sec(
                    0.5
                )
            assert np.allclose(goal_2.trajectory.points[0].positions,self.joints_2[:7] ,atol= 0.05,rtol=0.01)
        self.cancel_current_goal()
        if bspline_1 is not None:
            ros.loginfo(f'Robot 1 following trajectory',)
            self.client_1.send_goal(goal_1)
        if bspline_2 is not None:
            ros.loginfo(f'Robot 2 following trajectory',)
            self.client_2.send_goal(goal_2)
            
        # try:
        #     if bspline_1 is not None:
        #         self.client_1.wait_for_result()
        #     if bspline_2 is not None:
        #         self.client_2.wait_for_result()
        # except:
        #     self.client_1.cancel_all_goals()
        #     self.client_2.cancel_all_goals()
        
    def follow_acc_bspline_no_wait(self,bspline_1 = None,bspline_2 = None,time = 5,num_steps = 10000):
        assert time>= self.minimum_time
        # num_steps = 40
        dt = time/num_steps
        if bspline_1 is not None:
            
            goal_1 = FollowJointTrajectoryGoal()
            goal_1.trajectory.joint_names = self.JOINT_NAMES
            q  = self.joints_1[:7].copy()
            q_dot  = self.joints_dot_1[:7].copy()
            q_dots_dots = evaluate_bspline(np.linspace(0,1,num_steps), bspline_1.control_points, bspline_1.knots,bspline_1.order)/time**2
            q_dots = (integrate_q_trapezoidal(q_dot, q_dots_dots, dt))
            qs = (integrate_q_trapezoidal(q, q_dots, dt))[::200]
            for q,q_dot,t in zip(qs,q_dots[::200],np.linspace(0,time,qs.shape[0])):
                point = JointTrajectoryPoint()
                point.time_from_start = ros.Duration.from_sec(
                    t
                )
                
                q = np.clip(q,self.joint_lower_limits,self.joint_upper_limits)
                q_dot = np.clip(q_dot,self.joint_vel_lower_limits,self.joint_vel_upper_limits)
                point.positions = q
                point.velocities =  q_dot
                goal_1.trajectory.points.append(point)
                goal_1.goal_time_tolerance = ros.Duration.from_sec(
                    0.01
                )
            print(t,dt)
            assert np.allclose(goal_1.trajectory.points[0].positions,self.joints_1[:7] ,atol= 0.05,rtol=0.01)
        if bspline_2 is not None:
            
            goal_2 = FollowJointTrajectoryGoal()
            goal_2.trajectory.joint_names = self.JOINT_NAMES
            q  = self.joints_2[:7].copy()
            q_dot  = self.joints_dot_2[:7].copy()
            q_dots_dots = evaluate_bspline(np.linspace(0,1,num_steps), bspline_2.control_points, bspline_2.knots,bspline_2.order)/time**2
            q_dots = (integrate_q_trapezoidal(q_dot, q_dots_dots, dt))
            qs = (integrate_q_trapezoidal(q, q_dots, dt))[::200]
            for q,q_dot,t in zip(qs,q_dots[::200],np.linspace(0,time,qs.shape[0])):
                point = JointTrajectoryPoint()
                point.time_from_start = ros.Duration.from_sec(
                    t
                )
                q = np.clip(q,self.joint_lower_limits,self.joint_upper_limits)
                q_dot = np.clip(q_dot,self.joint_vel_lower_limits,self.joint_vel_upper_limits)

                # assert (q >= self.joint_lower_limits-0.001).all()
                # assert (q <= self.joint_upper_limits+0.001).all()
                # assert (q_dot >= self.joint_vel_lower_limits-0.001).all()
                # assert (q_dot <= self.joint_vel_upper_limits+0.001).all()
                point.positions = q
                point.velocities =  q_dot
                goal_2.trajectory.points.append(point)
                goal_2.goal_time_tolerance = ros.Duration.from_sec(
                    0.01
                )
            assert np.allclose(goal_2.trajectory.points[0].positions,self.joints_2[:7] ,atol= 0.05,rtol=0.01)
        self.cancel_current_goal()
        if bspline_1 is not None:
            ros.loginfo(f'Robot 1 following trajectory',)
            self.client_1.send_goal(goal_1)
        if bspline_2 is not None:
            ros.loginfo(f'Robot 2 following trajectory',)
            self.client_2.send_goal(goal_2)
            
        # try:
        #     if bspline_1 is not None:
        #         self.client_1.wait_for_result()
        #     if bspline_2 is not None:
        #         self.client_2.wait_for_result()
        # except:
        #     self.client_1.cancel_all_goals()
        #     self.client_2.cancel_all_goals()

    def follow_bspline(self,bspline_1 = None,bspline_2 = None,time = 5 , reverse = False):
        assert time>= self.minimum_time
        if bspline_1 is not None:
            bspline_1_dot = bspline_1.create_derivative_spline()
            goal_1 = FollowJointTrajectoryGoal()
            goal_1.trajectory.joint_names = self.JOINT_NAMES
            
            for s in np.linspace(0,1,20):
                point = JointTrajectoryPoint()
                # print(q,q_dot)
                point.time_from_start = ros.Duration.from_sec(
                    s*time
                )
                if reverse:
                    q = bspline_1.evaluate(1-s)
                    q_dot = -bspline_1_dot.evaluate(1-s)/time
                else:
                    q = bspline_1.evaluate(s)
                    q_dot = bspline_1_dot.evaluate(s)/time
                
                assert (q >= self.joint_lower_limits-0.001).all()
                assert (q <= self.joint_upper_limits+0.001).all()
                assert (q_dot >= self.joint_vel_lower_limits-0.001).all()
                assert (q_dot <= self.joint_vel_upper_limits+0.001).all()
                point.positions = q
                point.velocities =  q_dot
                goal_1.trajectory.points.append(point)
                goal_1.goal_time_tolerance = ros.Duration.from_sec(0.5)
            assert np.allclose(goal_1.trajectory.points[0].positions,self.joints_1[:7] ,atol= 0.01,rtol=0.01)
        if bspline_2 is not None:
            bspline_2_dot = bspline_2.create_derivative_spline()
            goal_2 = FollowJointTrajectoryGoal()
            goal_2.trajectory.joint_names = self.JOINT_NAMES
            for s in np.linspace(0,1,300):
                point = JointTrajectoryPoint()
                point.time_from_start = ros.Duration.from_sec(
                    s*time
                )
                if reverse:
                    q = bspline_2.evaluate(1-s)
                    q_dot = -bspline_2_dot.evaluate(1-s)/time
                else:
                    q = bspline_2.evaluate(s)
                    q_dot = bspline_2_dot.evaluate(s)/time
                assert (q >= self.joint_lower_limits-0.001).all()
                assert (q <= self.joint_upper_limits+0.001).all()
                assert (q_dot >= self.joint_vel_lower_limits-0.001).all()
                assert (q_dot <= self.joint_vel_upper_limits+0.001).all()
                point.positions = q
                point.velocities =  q_dot
                goal_2.trajectory.points.append(point)
                goal_2.goal_time_tolerance = ros.Duration.from_sec(0.5)
            assert np.allclose(goal_2.trajectory.points[0].positions,self.joints_2[:7] ,atol= 0.01,rtol=0.01)
        start_time = ros.Time.now().to_time() + 0.1
        tt = ros.Time.from_sec(start_time)
        if bspline_1 is not None:
            goal_1.trajectory.header.stamp = tt 
            ros.loginfo(f'Robot 1 following trajectory',)
            self.client_1.send_goal(goal_1)
        if bspline_2 is not None:
            goal_2.trajectory.header.stamp = tt
            
            ros.loginfo(f'Robot 2 following trajectory',)
            self.client_2.send_goal(goal_2)
            
        try:
            if bspline_1 is not None:
                self.client_1.wait_for_result()
            if bspline_2 is not None:
                self.client_2.wait_for_result()
        except:
            self.client_1.cancel_all_goals()
            self.client_2.cancel_all_goals()