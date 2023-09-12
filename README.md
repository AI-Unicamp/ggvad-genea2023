# ggvad-genea2023

Git clone this repo

Get the GENEA Challenge 2023 dataset and put it into ./dataset/
(Our system is monadic so you'll only need the main-agent's data)

Download the bvhsdk from https://github.com/rltonoli/bvhsdk for motion processing
(Alternatively you could adapt the code to use PyMO: https://github.com/omimo/PyMO)

Enter the repo

Create docker image using 
docker build -t ggvad .

Run container using
nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=INSERT_GPU_NUMBER --runtime=nvidia --userns=host --shm-size 64G -v OUT_DIR:CONTAINER_DIR -p PORT --name CONTAINER_NAME IMAGE_NAME /bin/bash
example:
nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 --runtime=nvidia --userns=host --shm-size 64G -v C:\ProgramFiles\ggvad-genea2023:/workspace/my_repo -p '8888:8888' --name my_container ggvad:latest /bin/bash

OR use the shell script ggvad_container.sh (don't forget to change the volume) using the flags -g, -n, and -p
example:
sh ggvad_container.sh -g 0 -n my_container -p '8888:8888'

Activate environment:
source activate ggvad

Navigate to /workspace/ggvad and run

python ./data_loaders/gesture/scripts/genea_prep.py
