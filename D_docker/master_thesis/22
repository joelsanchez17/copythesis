
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
COPY .env /tmp/.env

RUN bash -c "source /tmp/.env && rm /tmp/.env && \
    addgroup --gid \$GROUP_GID user && \
    adduser --disabled-password --gecos '' --uid \$USER_UID --gid \$GROUP_GID user && \ 
    apt-get update \
    && apt-get install -y sudo \
    && echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user"

# # Set the locale
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  


RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

    
RUN apt-get update && \
    apt-get install -y --no-install-recommends libblas-dev liblapack-dev libmetis-dev
RUN apt-get install gfortran -y
# Install Python packages
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get install python3-dev
RUN apt-get install -y --no-install-recommends git
RUN apt-get update && apt install -y cmake
RUN apt-get install wget -y
RUN apt-get -y install swig


# TODO: install BLIS for AMD systems, compile numpy from source
RUN apt-get -y install intel-mkl
RUN apt remove python3-numpy -y
RUN python3 -m pip install -i https://pypi.anaconda.org/intel/simple numpy
# TODO: releases/3.0.2
RUN cd /home/user/ && \
    git clone https://github.com/coin-or-tools/ThirdParty-Mumps && \
    cd ThirdParty-Mumps && \
    ./get.Mumps && \
    ./configure && \
    make && \
    make install && \
    make install

# https://github.com/coin-or-tools/ThirdParty-HSL
RUN git clone https://github.com/coin-or-tools/ThirdParty-HSL.git /home/user/ThirdParty-HSL
COPY ./linux/coinhsl-2023.11.17.tar.gz /home/user/ThirdParty-HSL/coinhsl-2023.11.17.tar.gz
RUN cd /home/user/ThirdParty-HSL && \
    gunzip /home/user/ThirdParty-HSL/coinhsl-2023.11.17.tar.gz && \
    tar xf /home/user/ThirdParty-HSL/coinhsl-2023.11.17.tar && \
    ln -s coinhsl-2023.11.17 coinhsl
RUN cd /home/user/ThirdParty-HSL && ./configure
RUN cd /home/user/ThirdParty-HSL && make
RUN cd /home/user/ThirdParty-HSL && make install
RUN ln -s /usr/local/lib/libcoinhsl.so /usr/local/lib/libhsl.so
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# TODO: Version 3.14.11
RUN cd /home/user/ && \
    git clone https://github.com/coin-or/Ipopt.git && \
    cd Ipopt && \
    export IPOPTDIR=$(pwd) && \
    mkdir $IPOPTDIR/build && \
    cd $IPOPTDIR/build && \
    $IPOPTDIR/configure && \
    make && \
    make install
ENV CASADI_SETUP_CMAKE_ARGS="-DWITH_THREAD=ON;-DWITH_BUILD_EIGEN3=ON;-DWITH_OPENMP=ON;-DWITH_IPOPT=ON;-DWITH_HSL=ON"

RUN cd /home/user/ && wget https://github.com/casadi/casadi/releases/download/3.6.5/casadi-3.6.5.tar.gz
RUN cd /home/user/ && pip install casadi-3.6.5.tar.gz
RUN cd /home/user/ && rm casadi-3.6.5.tar.gz
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/casadi/:$LD_LIBRARY_PATH

RUN pip install numpy==1.26.4
RUN pip install scipy==1.12.0
RUN pip install sympy==1.11.1
RUN pip install symforce==0.9.0
RUN pip install matplotlib==3.8.3
RUN pip install opencv-python==4.9.0.80
RUN pip install pandas==1.5.3
RUN pip install toolz
RUN pip install einops
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install drake==1.23.0
RUN pip install manipulation
RUN pip install CairoSVG

# 2.1.2
RUN pip install lightning

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 ssh  -y

# render latex stuff with sympy and matplotlib
RUN apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super -y

RUN apt-get install -y --no-install-recommends \
      libx11-6 libsm6 libglib2.0-0


RUN apt-get install -y pkg-config
RUN apt-get install -y graphviz 




# WSL GPU stuff
# ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
# ENV LIBVA_DRIVER_NAME=d3d12

# # Required for non-glvnd setups. https://gitlab.com/nvidia/container-images/opengl/-/blob/5191cf205d3e4bb1150091f9464499b076104354/base/Dockerfile
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV PYTHONPATH="${PYTHONPATH}:/workspaces/toys"

RUN git config --global user.name "Goghnelson"
RUN git config --global user.email gogh@goghluci_tv.com

# misc
RUN apt install -y net-tools iputils-ping

USER user

EXPOSE 11311