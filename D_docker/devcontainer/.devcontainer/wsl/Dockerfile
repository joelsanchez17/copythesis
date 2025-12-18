
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG USER_UID=1000
ARG GROUP_GID=1000

RUN bash -c "addgroup --gid \$GROUP_GID user && \
    adduser --disabled-password --gecos '' --uid \$USER_UID --gid \$GROUP_GID user && \ 
    apt-get update \
    && apt-get install -y sudo \
    && echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user"
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
    apt-get install -y --no-install-recommends 



RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    wget git \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libffi-dev liblzma-dev \
    software-properties-common lsb-release \
    build-essential gfortran pkg-config cmake swig\
    libblas-dev liblapack-dev libmetis-dev \
    net-tools iputils-ping \
    graphviz ffmpeg libsm6 libxext6 ssh dvipng texlive-latex-extra texlive-fonts-recommended cm-super libx11-6 libglib2.0-0 \
    intel-mkl && \
    apt-get clean
RUN cd /home/user && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py --prefix=/usr/ && \
    rm get-pip.py
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip

RUN pip install sympy==1.11.1 \
    symforce==0.9.0 \
    matplotlib==3.8.3 \
    opencv-python==4.9.0.80 \
    pandas==1.5.3 \
    toolz \
    einops \
    drake==1.23.0 \
    manipulation \
    CairoSVG \
    lightning==2.1.2 \
    meson==1.4.0
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
ENV MKL_CONFIG_NUMPY=mkl-dynamic-ilp64-iomp
ENV MKL_CONFIG_IPOPT=mkl-dynamic-lp64-iomp

RUN cd /home/user/ && \
    git clone --depth 1 --branch v1.26.5 https://github.com/numpy/numpy.git && \
    cd numpy && \
    git submodule update --init && \
    python3 -m pip install -r build_requirements.txt && \
    MAKEFLAGS="-j$(nproc)" python3 -m pip install . -Csetup-args=-Dblas=$MKL_CONFIG_NUMPY -Csetup-args=-Dlapack=$MKL_CONFIG_NUMPY && \
    cd .. && \
    rm -rf numpy
RUN cd /home/user/ && \
    git clone --branch v1.13.1 https://github.com/scipy/scipy.git && \
    cd scipy && \
    git submodule update --init && \
    python3 -m pip install -r requirements/all.txt && \
    MAKEFLAGS="-j$(nproc)" python3 -m pip install . -Csetup-args=-Dblas=$MKL_CONFIG_NUMPY -Csetup-args=-Dlapack=$MKL_CONFIG_NUMPY && \
    cd .. && \
    rm -rf scipy


ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libiomp5.so:$LD_PRELOAD



RUN cd /home/user/ && \
    git clone --branch releases/3.0.7 https://github.com/coin-or-tools/ThirdParty-Mumps && \
    cd ThirdParty-Mumps && \
    ./get.Mumps && \
    # ./configure && \
    # https://stackoverflow.com/questions/38038444/linking-ipopt-with-intel-mkl
    ./configure FFLAGS="-Ofast -march=native" CFLAGS="-Ofast -march=native" CPPFLAGS="-Ofast -march=native" --with-lapack-lflags="$(pkg-config --libs $MKL_CONFIG_IPOPT)" && \
    make -j$(nproc) && \
    make install && \
    cd /home/user/ && \
    rm -rf ThirdParty-Mumps
# ./configure CFLAGS="-Ofast -march=native" CPPFLAGS="-Ofast -march=native -flto" LDFLAGS="-O1 -flto" FCFLAGS="-Ofast -march=native" --with-lapack-lflags="$(pkg-config --libs mkl-dynamic-lp64-iomp)"
# https://github.com/coin-or-tools/ThirdParty-HSL
RUN git clone https://github.com/coin-or-tools/ThirdParty-HSL.git /home/user/ThirdParty-HSL
ENV COINHSL_VERSION=2023.11.17
COPY ./coinhsl-$COINHSL_VERSION.tar.gz /home/user/ThirdParty-HSL/coinhsl-$COINHSL_VERSION.tar.gz
RUN cd /home/user/ThirdParty-HSL && \
    # git checkout 9eac178 && \
    gunzip /home/user/ThirdParty-HSL/coinhsl-$COINHSL_VERSION.tar.gz && \
    tar xf /home/user/ThirdParty-HSL/coinhsl-$COINHSL_VERSION.tar && \
    ln -s coinhsl-$COINHSL_VERSION coinhsl && \
    ./configure FFLAGS="-Ofast -march=native" CFLAGS="-Ofast -march=native" CPPFLAGS="-Ofast -march=native" --with-lapack-lflags="$(pkg-config --libs $MKL_CONFIG_IPOPT)" && \
    make -j$(nproc) && \
    make install && \
    ln -s /usr/local/lib/libcoinhsl.so /usr/local/lib/libhsl.so && \
    cd /home/user/ && \
    rm -rf ThirdParty-HSL
    
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN cd /home/user/ && \
    git clone --branch releases/3.14.16 https://github.com/coin-or/Ipopt.git && \
    cd Ipopt && \
    mkdir build && \
    cd build && \
    # ../configure CFLAGS="-Ofast -march=native" CPPFLAGS="-Ofast -march=native" --with-lapack-lflags="$(pkg-config --libs $MKL_CONFIG_IPOPT)" --with-hsl-lflags="-L/usr/lib/x86_64-linux-gnu -lcoinhsl" --with-hsl-cflags=-I/usr/include
    ../configure FFLAGS="-Ofast -march=native" CFLAGS="-Ofast -march=native" CPPFLAGS="-Ofast -march=native" --with-lapack-lflags="$(pkg-config --libs $MKL_CONFIG_IPOPT)" && \
    make -j$(nproc) && \
    make install && \
    cd /home/user/ && \
    rm -rf Ipopt

ENV CASADI_SETUP_CMAKE_ARGS="-DWITH_THREAD=ON;-DWITH_BUILD_EIGEN3=ON;-DWITH_OPENMP=ON;-DWITH_IPOPT=ON;-DWITH_HSL=ON"
# ENV CASADI_SETUP_CMAKE_ARGS="-DWITH_THREAD=ON;-DWITH_BUILD_EIGEN3=ON;-DWITH_OPENMP=ON;-DWITH_IPOPT=ON;-DWITH_HSL=ON;-DWITH_QPOASES=ON;-DWITH_FATROP=ON;-DWITH_BUILD_FATROP=ON;-DWITH_LAPACK=ON"
# export CASADI_SETUP_CMAKE_ARGS="-DWITH_THREAD=ON;-DWITH_BUILD_EIGEN3=ON;-DWITH_OPENMP=ON;-DWITH_IPOPT=ON;-DWITH_HSL=ON;-DWITH_QPOASES=ON;-DWITH_FATROP=OFF;-DWITH_BUILD_FATROP=OFF;-DWITH_LAPACK=ON"
RUN cd /home/user/ && wget https://github.com/casadi/casadi/releases/download/3.6.5/casadi-3.6.5.tar.gz && \
    MAKEFLAGS="-j$(nproc)" pip install casadi-3.6.5.tar.gz && \
    rm casadi-3.6.5.tar.gz

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/casadi/:$LD_LIBRARY_PATH

# WSL GPU stuff
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
ENV LIBVA_DRIVER_NAME=d3d12

# # Required for non-glvnd setups. https://gitlab.com/nvidia/container-images/opengl/-/blob/5191cf205d3e4bb1150091f9464499b076104354/base/Dockerfile
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV PYTHONPATH="${PYTHONPATH}:/workspaces/toys"

RUN git config --global user.name "Goghnelson"
RUN git config --global user.email gogh@goghluci_tv.com




USER user

EXPOSE 11311

