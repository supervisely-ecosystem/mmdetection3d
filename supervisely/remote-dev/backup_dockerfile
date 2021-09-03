ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -c anaconda python=3.8


# Install MMCV, MMDetection and MMSegmentation
RUN pip3 install mmcv-full
RUN pip3 install mmdet==2.14.0
RUN pip3 install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/supervisely-ecosystem/mmdetection3d.git /mmdetection3d
WORKDIR /mmdetection3d
RUN git checkout sly
ENV FORCE_CUDA="1"
RUN pip3 install -r requirements/build.txt
RUN pip3 install --no-cache-dir -e .

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip3 uninstall pycocotools --no-cache-dir -y
RUN pip3 install mmpycocotools --no-cache-dir --force --no-deps

RUN pip3 install --ignore-installed PyYAML
RUN pip3 install open3d
RUN pip3 install supervisely

RUN ln -s /opt/conda/bin/python3 /usr/local/bin/python
RUN ln -s /opt/conda/bin/pip3 /usr/local/bin/pip