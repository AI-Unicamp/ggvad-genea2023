FROM nvidia/cuda:12.2.0-devel-ubuntu22.04


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget git nano ffmpeg

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.8.3-Linux-x86_64.sh

RUN conda --version

WORKDIR /root
COPY environment.yml /root

RUN conda install tqdm -f
RUN conda update conda
RUN conda install pip
RUN conda --version
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "ggvad", "/bin/bash", "-c"]
RUN pip install git+https://github.com/openai/CLIP.git
