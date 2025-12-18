#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys

sys.path += ["../diff_co_mpc/"]
from diff_co_lcm import lcmt_pose
from pydrake.all import DrakeLcm, RigidTransform, RollPitchYaw
import numpy as np
import time
lcm = DrakeLcm()

lcmt_start_pose = lcmt_pose()
lcmt_end_pose = lcmt_pose()

start_pose = RigidTransform(p=[0.0, 0.23, 0.15],rpy=RollPitchYaw([0, np.pi/4*0, 0]))
end_pose = RigidTransform(p=[0.0, -0.45, 0.15],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 
end_pose = RigidTransform(p=[0.0, 0.27, 0.15],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 


start_pose = RigidTransform(p=[0.0, -0.4, 0.2],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 

start_pose = RigidTransform(p=[0.0, -0.4, 0.25],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 
end_pose = RigidTransform(p=[0.0, 0.27, 0.25],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 

# start_pose = RigidTransform(p=[0.0, 0.27, 0.25],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 
# end_pose = RigidTransform(p=[0.0, -0.4, 0.2],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 
# end_pose = RigidTransform(p=[0.0, -0.4, 0.25],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 
# RigidTransform(p=[0.0, -0.4, 0.25],rpy=RollPitchYaw([0, 0, np.pi/4*0])) 

lcmt_start_pose.pose = start_pose.GetAsMatrix4().flatten()

lcmt_start_pose.timestamp = time.perf_counter_ns()
lcmt_end_pose.pose = end_pose.GetAsMatrix4().flatten()
lcmt_end_pose.timestamp = time.perf_counter_ns()

# state.
while True:
    lcm.Publish(channel="initial_pose", buffer = lcmt_start_pose.encode())
    lcm.Publish(channel="final_pose", buffer = lcmt_end_pose.encode())
    time.sleep(1)


# In[ ]:


import casadi
casadi.__version__

