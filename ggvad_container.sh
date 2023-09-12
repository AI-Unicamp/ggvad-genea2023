while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container ggvad_container_$number on gpu $gpu and port $port";

docker run --rm -it --gpus device=$gpu --userns=host --shm-size 64G -v /work/rodolfo.tonoli/ggvad-genea2023:/workspace/ggvad/ -p $port --name ggvad_container$number ggvad:latest /bin/bash