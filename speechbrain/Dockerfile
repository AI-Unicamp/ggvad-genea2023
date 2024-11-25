FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update 
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt-get update

RUN apt-get install -y wget git nano ffmpeg \
    libavcodec-extra \
    libavdevice58 \
    libavfilter7

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.8.3-Linux-x86_64.sh

RUN conda install -y python=3.9

RUN conda --version

WORKDIR /root
COPY environment.yml /root

RUN conda install tqdm -f
RUN conda update conda
RUN conda install pip
RUN conda --version

RUN rm -rf $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/ruamel*
RUN pip install speechbrain

RUN pip install torchvision