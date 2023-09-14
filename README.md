# ggvad-genea2023

## Preparing environment

Git clone this repo

Get the GENEA Challenge 2023 dataset and put it into ./dataset/
(Our system is monadic so you'll only need the main-agent's data)

Download the bvhsdk from https://github.com/rltonoli/bvhsdk for motion processing
(Alternatively you could adapt the code to use PyMO: https://github.com/omimo/PyMO)

Enter the repo

Create docker image using 
docker build -t ggvad .

Run container using

```sh
docker run --rm -it --gpus device=GPU_NUMBER --userns=host --shm-size 64G -v /MY_DIR/ggvad-genea2023:/workspace/ggvad/ -p PORT_NUMBR --name CONTAINER_NAME ggvad:latest /bin/bash
```

for example:
```sh
nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 --runtime=nvidia --userns=host --shm-size 64G -v C:\ProgramFiles\ggvad-genea2023:/workspace/my_repo -p '8888:8888' --name my_container ggvad:latest /bin/bash
```

> ### Cuda version < 12.0:
> 
> If you have a previous cuda or nvcc release version you will need to adjust the Dockerfile. Change the first line to "FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel" and remove lines 10-14 (conda is already installed in the pythorch image)
> 
> ```sh
> nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=GPU_NUMBER > --runtime=nvidia --userns=host --shm-size 64G -v /work/rodolfo.tonoli/GestureDiffusion:/workspace/gesture-diffusion/ -p $port --name gestdiff_container$number multimodal-research-group-mdm:latest /bin/bash
> ```


OR use the shell script ggvad_container.sh (don't forget to change the volume) using the flags -g, -n, and -p
example:
```sh
sh ggvad_container.sh -g 0 -n my_container -p '8888:8888'
```

Activate environment:
```sh
source activate ggvad
```

## Dara pre-processing

Navigate to /workspace/ggvad and run

```sh
python ./data_loaders/gesture/scripts/genea_prep.py
```


