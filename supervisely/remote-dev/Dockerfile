ARG PYTORCH="1.6.0"
ARG CUDA_VERSION="10.1"
ARG CUDNN_VERSION="7"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

RUN apt-get update && apt-get install -y python3.8 python3.8-dev
RUN update-alternatives --remove python /usr/bin/python3.6
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10

RUN apt remove python-pip
RUN apt install -y python3-pip
RUN ln -s $(which python3.8) /usr/local/bin/python
RUN python -m pip --no-cache-dir install --upgrade --force pip
RUN python -m pip --no-cache-dir install --upgrade setuptools
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y ffmpeg libsm6 libxext6

RUN pip install open3d
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
RUN pip install --no-cache-dir mmdet==2.14.0
RUN pip install --no-cache-dir mmsegmentation==0.14.1


 #Install MMDetection3D
RUN apt-get install -y git
RUN git clone https://github.com/supervisely-ecosystem/mmdetection3d.git /mmdetection3d
WORKDIR /mmdetection3d
RUN git checkout sly



ENV FORCE_CUDA="1"
RUN python -c "import torch; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(torch.cuda.get_device_properties(device))"
# INSTALL PACKAGES BELLOW MANUALLY

#RUN pip install -r requirements/build.txt
#RUN pip install --no-cache-dir -e .

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
#RUN pip uninstall pycocotools --no-cache-dir -y
#RUN pip install mmpycocotools --no-cache-dir --force --no-deps
