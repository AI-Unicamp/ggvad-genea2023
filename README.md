# ggvad-genea2023

## Preparing environment

Git clone this repo

Get the GENEA Challenge 2023 dataset and put it into ./dataset/
(Our system is monadic so you'll only need the main-agent's data)

Download the bvhsdk from https://github.com/rltonoli/bvhsdk for motion processing
(Alternatively you could adapt the code to use PyMO: https://github.com/omimo/PyMO)

Enter the repo

Create docker image using 

```sh
docker build -t ggvad .
```

Run container using

```sh
docker run --rm -it --gpus device=GPU_NUMBER --userns=host --shm-size 64G -v /MY_DIR/ggvad-genea2023:/workspace/ggvad/ -p PORT_NUMBR --name CONTAINER_NAME ggvad:latest /bin/bash
```

for example:
```sh
docker run --rm -it --gpus device=0 --userns=host --shm-size 64G -v C:\ProgramFiles\ggvad-genea2023:/workspace/my_repo -p '8888:8888' --name my_container ggvad:latest /bin/bash
```

> ### Cuda version < 12.0:
> 
> If you have a previous cuda or nvcc release version you will need to adjust the Dockerfile. Change the first line to `FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel` and remove lines 10-14 (conda is already installed in the pythorch image)
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

Download the [WavLM Base +](https://github.com/microsoft/unilm/tree/master/wavlm) and put it into the folder `/wavlm/`

Navigate to `/workspace/ggvad` and run

```sh
python ./data_loaders/gesture/scripts/genea_prep.py
```

This you convert bvh to npy representations, downsample wav files to 16k and save them as npy arrays, and convert these arrays to wavlm representations. The VAD data must be processed separetely due to python libraries incompatibility. 

### (Optional) Process VAD data

We provide the speech activity information (from speechbrain's VAD) data, but if you wish to process them yourself you should redo the steps of "Preparing environment" as before, but for the speechbrain environment: Build the image using the Dockerfile inside speechbrain (`docker build -t speechbrain .`), run the container (`docker run ... --name CONTAINER_NAME speechbrain:latest /bin/bash`) and run:

```sh
python ./data_loaders/gesture/scripts/genea_prep_vad.py
```

