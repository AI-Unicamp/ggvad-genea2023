while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container speechbrain_container_$number on gpu $gpu and port $port";

nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=$gpu --runtime=nvidia --userns=host --shm-size 64G -v /work/rodolfo.tonoli/ggvad-genea2023:/workspace/ggvad/ -p $port --name ggvad_container$number speechbrain_vad:latest /bin/bash