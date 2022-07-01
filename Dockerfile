# WACV18 Monocular 3D hand pose estimation Docker 

# FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04
# FROM nvidia/opengl:1.2-glvnd-devel-ubuntu16.04
FROM nvidia/cudagl:10.1-devel-ubuntu16.04

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update

ENV LC_ALL=C
ENV MAKEFLAGS="-j$(nproc)"

RUN DEBIAN_FRONTEND="noninteractive" TZ="Europe/Athens" apt-get install -y tzdata
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y libboost-all-dev libpython-dev \
                    git cmake vim libgoogle-glog-dev libprotobuf-dev protobuf-compiler \
                    libhdf5-dev libatlas-base-dev liblmdb-dev libleveldb-dev \
                    libsnappy-dev wget unzip  apt-utils \
                    libtbb-dev libglew-dev libopenni-dev libglm-dev freeglut3-dev libeigen3-dev \
                    ffmpeg x264 libx264-dev \
                    libgtk2.0-dev pkg-config

# needed by MonoHand3D

RUN apt-get install -y libcholmod3.0.6  
# libopenni0 

RUN apt-get install -y python3 python3-pip python3-distutils-extra

RUN pip3 install numpy==1.12.1
RUN pip3 install scikit-build==0.6.0
RUN pip3 install opencv-python==3.4.3.18

RUN apt-get install -y libopenexr-dev

RUN apt-get install -y libbulletdynamics2.83.6

# RUN pip install scipy

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/lib

# Enable OpenGL support 
# RUN apt-get install -y -qq --no-install-recommends libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


# Build opencv 
RUN mkdir opencv && cd opencv && wget https://github.com/opencv/opencv/archive/3.4.11.zip && unzip 3.4.11.zip && rm 3.4.11.zip  && \
    mkdir build && cd build && \
    cmake -DWITH_CUDA=ON -DBUILD_EXAMPLES=OFF -DOPENCV_GENERATE_PKGCONFIG=ON ../opencv-3.4.11 && \
    make -j`nproc` && make install


# WACV18 is an old project. The current version Openpose is not compatible any more.
# We clone and checkout the last compatible version (see PyOpenpose README).
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git && \
    cd openpose && git checkout e38269862f05beca9497960eef3d35f9eecc0808 && \
    git submodule update --init --recursive

# NOTE: Openpose comes with a CMake build system. 
# Unfortunatelly the commit we are using here has a bug that breaks the caffe build system for 
# GPUs newer than Pascal. So we are using the old Makefiles for this Dockerfile.

RUN apt-get install -y python3-dev python3-numpy

COPY cudnn/*.deb /tmp/
RUN dpkg -i /tmp/libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb /tmp/libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb 

# Build caffee 
COPY config/Makefile.config.caffe openpose/3rdparty/caffe/Makefile.config
RUN cd openpose/3rdparty/caffe/ && \
    make all -j`nproc` && make distribute -j`nproc`

# Build Openpose
COPY config/Makefile.config.openpose openpose/Makefile.config
RUN cd openpose && cp ubuntu_deprecated/Makefile.example Makefile && \
    make all -j`nproc` && make distribute -j`nproc` 

# This would be normally done by cmake but since we used the Makefiles for openpose build:
RUN cp -r openpose/3rdparty/caffe/distribute/* openpose/distribute && \
    ln -s /workspace/openpose_models openpose/distribute/models

# Setup environment variables needed by PyOpenpose
# Environment variables are set in the image and inherited by the container.
# applications running in the container have access to these environment vars.
ENV OPENPOSE_ROOT=/openpose/distribute
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${OPENPOSE_ROOT}/lib"

# Build PyOpenPose
RUN git clone https://github.com/FORTH-ModelBasedTracker/PyOpenPose.git && \
    mkdir PyOpenPose/build && cd PyOpenPose/build && cmake .. -DWITH_PYTHON3=1 && \
    make -j`nproc` && make install

ENV PYTHONPATH=/usr/local/lib:$PYTHONPATH
ENV LD_LIBRARY_PATH=/workspace/lib:/usr/local/lib:$LD_LIBRARY_PATH


# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Set the workspace location (where new code will go)
WORKDIR /workspace
