#/usr/bash

sudo apt -y install git gnupg wget curl
python3 -m pip install catkin_pkg
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu jammy main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt-get -y install python3-rosdep python3-rosinstall-generator python3-vcstools python3-vcstool build-essential
sudo rosdep init
cd ~/
wget http://archive.ubuntu.com/ubuntu/pool/universe/h/hddtemp/hddtemp_0.3-beta15-53_amd64.deb
sudo apt install ~/hddtemp_0.3-beta15-53_amd64.deb
rm ~/hddtemp_0.3-beta15-53_amd64.deb
rm ~/base.yaml
wget https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/base.yaml
sudo python3 -c "import os, re; home = os.path.expanduser('~' + os.environ.get('SUDO_USER', '')); file_path = os.path.join(home, 'base.yaml'); content = open(file_path).read(); content = re.sub(r'(hddtemp:\n  arch: \[hddtemp\]\n  debian: \[hddtemp\]\n  fedora: \[hddtemp\]\n  freebsd: \[python27\]\n  gentoo: \[app-admin/hddtemp\]\n  macports: \[python27\]\n  nixos: \[hddtemp\]\n  openembedded: \[hddtemp@meta-oe\]\n  opensuse: \[hddtemp\]\n  rhel: \[hddtemp\]\n  slackware: \[hddtemp\]\n  ubuntu:\n    \'\\*\': null\n    bionic: \[hddtemp\]\n    focal: \[hddtemp\])', r'\1\n    impish: [hddtemp]\n    jammy: [hddtemp]', content); open(file_path, 'w').write(content)"
sudo python3 -c "import os, re; home = os.path.expanduser('~' + os.environ.get('SUDO_USER', '')); file_path = '/etc/ros/rosdep/sources.list.d/20-default.list'; content = open(file_path).read(); content = re.sub(r'yaml https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/base.yaml', rf'yaml file://{home}/base.yaml', content); open(file_path, 'w').write(content)"

rosdep update
rm -r -f ~/ros_catkin_ws
mkdir ~/ros_catkin_ws
cd ~/ros_catkin_ws
rosinstall_generator desktop --rosdistro noetic --deps --tar > noetic-desktop.rosinstall
mkdir ./src
vcs import --input noetic-desktop.rosinstall ./src
rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y
cd ~/ros_catkin_ws/

# https://github.com/ros/ros_comm/issues/2330
cd ~/ros_catkin_ws/src
rm -f -r rosconsole
rm -f -r urdf

git clone https://github.com/dreuter/rosconsole.git
cd rosconsole
git checkout noetic-jammy
cd ~/ros_catkin_ws/src

git clone https://github.com/dreuter/urdf.git
cd urdf
git checkout set-cxx-version
cd ~/ros_catkin_ws/src

# IF ERROR ABOUT LIBTBB2-DEV:
# sudo apt purge libtbb2-dev
# then rosdep:
cd ~/ros_catkin_ws
rosdep install --from-paths src --ignore-src -r -y
cd ~/ros_catkin_ws
sudo ./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release --install-space /opt/ros/noetic