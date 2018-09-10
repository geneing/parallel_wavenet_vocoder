FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

ARG PYTHON_VERSION=3.6
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake ccache \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev libeigen3-dev libnuma-dev libleveldb-dev liblmdb-dev libopencv-dev libsnappy-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy matplotlib ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -c mingfeima mkldnn && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda92 
     #/opt/conda/bin/conda install -c pytorch pytorch 
     #/opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -c anaconda tensorflow-gpu

ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
RUN git clone --recursive https://github.com/pytorch/pytorch 
WORKDIR /opt/pytorch/pytorch
RUN git checkout v0.4.1
RUN git submodule update --init

RUN TORCH_CUDA_ARCH_LIST="3.0;3.5;5.2;6.0;6.1" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    #python setup.py build && \
    pip install -v .

WORKDIR /opt/pytorch
RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

RUN pip install -U docopts nnmnkwii>=0.0.11 tqdm tensorboardX keras scikit-learn lws librosa

WORKDIR /workspace
RUN chmod -R a+w /workspace


