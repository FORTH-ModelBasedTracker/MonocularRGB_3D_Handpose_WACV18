# WACV18 Monocular 3D hand pose estimation Docker 

FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" TZ="Europe/Athens" apt-get install -y tzdata
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y libboost-all-dev libpython-dev python-pip \
                    git cmake vim libgoogle-glog-dev libprotobuf-dev protobuf-compiler \
                    libhdf5-dev libatlas-base-dev liblmdb-dev libleveldb-dev \
                    libsnappy-dev wget unzip  apt-utils libpython-dev python-numpy \
                    libtbb-dev libglew-dev libopenni-dev libglm-dev freeglut3-dev libeigen3-dev \
                    ffmpeg x264 libx264-dev \
                    libgtk2.0-dev pkg-config

# needed by MonoHand3D
# RUN pip install scipy
ENV LC_ALL=C


RUN apt-get install -y libgoogle-glog-dev libtbb-dev libcholmod3.0.6 libatlas-base-dev libopenni0 libbulletdynamics2.83.6
RUN apt-get install -y python3-dev python3-pip

ENV MAKEFLAGS="-j$(nproc)"
RUN pip3 install scikit-build
RUN pip3 install opencv-python==3.4.3.18


ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/lib

# Enable OpenGL support 
# RUN apt-get install -y -qq --no-install-recommends libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Set the workspace location (where new code will go)
WORKDIR /workspace
