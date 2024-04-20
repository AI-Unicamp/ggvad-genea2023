# ggvad-genea2023

Implementaion of the paper [Gesture Generation with Diffusion Models Aided by Speech Activity Information](https://openreview.net/forum?id=S9Efb3MoiZ)

## Preparing environment

1. Git clone this repo

2. Enter the repo and create docker image using 

```sh
docker build -t ggvad .
```

3. Run container using

```sh
docker run --rm -it --gpus device=GPU_NUMBER --userns=host --shm-size 64G -v /MY_DIR/ggvad-genea2023:/workspace/ggvad/ -p PORT_NUMBR --name CONTAINER_NAME ggvad:latest /bin/bash
```

for example:
```sh
docker run --rm -it --gpus device=0 --userns=host --shm-size 64G -v C:\ProgramFiles\ggvad-genea2023:/workspace/my_repo -p '8888:8888' --name my_container ggvad:latest /bin/bash
```

> ### Cuda version < 12.0:
> 
> If you have a previous cuda or nvcc release version you will need to adjust the Dockerfile. Change the first line to `FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel` and remove lines 10-14 (conda is already installed in the pythorch image). Then, run container using:
> 
> ```sh
> nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=GPU_NUMBER --runtime=nvidia --userns=host --shm-size 64G -v /work/rodolfo.tonoli/GestureDiffusion:/workspace/gesture-diffusion/ -p $port --name gestdiff_container$number multimodal-research-group-mdm:latest /bin/bash
> ```


OR use the shell script ggvad_container.sh (don't forget to change the volume) using the flags -g, -n, and -p
example:
```sh
sh ggvad_container.sh -g 0 -n my_container -p '8888:8888'
```

4. Activate cuda environment:
```sh
source activate ggvad
```

## Data pre-processing

1. Get the GENEA Challenge 2023 dataset and put it into `./dataset/`
(Our system is monadic so you'll only need the main-agent's data)

2. Download the [WavLM Base +](https://github.com/microsoft/unilm/tree/master/wavlm) and put it into the folder `/wavlm/`

3. Inside the folder `/workspace/ggvad`, run

```sh
python -m data_loaders.gesture.scripts.genea_prep
```

This will convert the bvh files to npy representations, downsample wav files to 16k and save them as npy arrays, and convert these arrays to wavlm representations. The VAD data must be processed separetely due to python libraries incompatibility. 

4. (Optional) Process VAD data

We provide the speech activity information (from speechbrain's VAD) data, but if you wish to process them yourself you should redo the steps of "Preparing environment" as before, but for the speechbrain environment: Build the image using the Dockerfile inside speechbrain (`docker build -t speechbrain .`), run the container (`docker run ... --name CONTAINER_NAME speechbrain:latest /bin/bash`) and run:

```sh
python -m data_loaders.gesture.scripts.genea_prep_vad
```

## Train model

To train the model described in the paper use the following command inside the repo:

```sh
python -m train.train_mdm --save_dir save/my_model_run --dataset genea2023+ --step 10  --use_text --use_vad True --use_wavlm True
```

## Gesture Generation

Generate motion using the trained model by running the following command. If you wish to generate gestures with the pretrained model of the Genea Challenge, use `--model_path ./save/default_vad_wavlm/model000290000.pt` 

```sh
python -m sample.generate --model_path ./save/my_model_run/model000XXXXXX.pt 
```

## Render

To render the official Genea 2023 visualizations follow the instructions provided [here](https://github.com/TeoNikolov/genea_visualizer/)

## Cite

If you with to cite this repo or the paper

```text
@inproceedings{tonoli2023gesture,
  title={Gesture Generation with Diffusion Models Aided by Speech Activity Information},
  author={Tonoli, Rodolfo L and Marques, Leonardo B de MM and Ueda, Lucas H and Costa, Paula Dornhofer Paro},
  booktitle={Companion Publication of the 25th International Conference on Multimodal Interaction},
  pages={193--199},
  year={2023}
}
```