#!/bin/bash
set -e

echo "Post-create script running..."

# Fix permissions if needed
sudo chown -R user:user /home/user

# (Optional) Clone repos, setup ROS, etc.
