FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# ubuntu 20 has python 3.8
RUN apt-get update && apt-get install -y python3.8

RUN update-alternatives --remove python /usr/bin/python2
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10

RUN apt remove python-pip

RUN python -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools


RUN ln -s $(which python3) /usr/local/bin/python
RUN apt-get -y install libcusolver-11-0 git libgl1-mesa-glx

#RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.0/lib64
RUN echo '/usr/local/cuda-10.2/lib64' >> /etc/ld.so.conf.d/nvidia.conf
#RUN echo '/usr/local/cuda-11.0/lib64' >> /etc/ld.so.conf.d/nvidia.conf
RUN ldconfig


RUN pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install supervisely

RUN pip3 install mmcv-full
RUN pip3 install mmdet==2.14.0
RUN pip3 install mmsegmentation==0.14.1


# Install MMDetection3D
RUN git clone https://github.com/supervisely-ecosystem/mmdetection3d.git /mmdetection3d
WORKDIR /mmdetection3d
RUN git checkout sly
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

