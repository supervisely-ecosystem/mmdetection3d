ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


RUN conda create -n open-mmlab python=3.8 -y
RUN conda activate open-mmlab

# install latest PyTorch prebuilt with the default prebuilt CUDA version (usually the latest)
RUN conda install -c pytorch pytorch torchvision -y

# install mmcv
RUN pip install mmcv-full

# install mmdetection
RUN pip install git+https://github.com/open-mmlab/mmdetection.git

# install mmsegmentation
RUN pip install git+https://github.com/open-mmlab/mmsegmentation.git

# install mmdetection3d
ENV FORCE_CUDA="1"
RUN git clone https://github.com/open-mmlab/mmdetection3d.git
RUN cd mmdetection3d
RUN pip install -v -e .
RUN pip install supervisely


