#!/usr/bin/bash

usage() {
    echo "Usage: $0 [-i INTERMEDIATE] [-a ANNOTS_PATH] [-e EMBEDS_PATH] [-s SAVE_PATH]" 1>&2
    echo "-h prints this help message"
    echo "-i intrmediate files folder"
    echo "-a path where to find annotations"
    echo "-e path where to find embeddings"
    echo "-s folder where to save models"
    exit 1
}

# Default options
data="intermediate_files/use_dataset/"
annots="intermediate_files/prep_dataset/annotations/"
embeds="embeddings_tf/"
savepath="intermediate_files/use_dataset/train_fold/"
src="src/use_dataset"
threads=8

# Options parser
while getopts "h:i:a:e:s:" opt
do
    case "${opt}" in
        i) data=${OPTARG};;
        a) annots=${OPTARG};;
        e) embeds=${OPTARG};;
        s) savepath=${OPTARG};;
        h | *) usage;;
    esac
done

if [ -z "${data}" ] || [ -z "${annots}" ] || [ -z "${embeds}" ] || [ -z "${savepath}" ]; then
    usage
else
    chmod +x src/use_dataset/*
    mkdir -p $savepath
fi

prep=1
bpo=0
mfo=0
cco=0

if [ $prep -eq 1 ]
then
    for ont in 'cco' 'mfo' 'bpo'
    do
        python3 src/use_dataset/unpicle_labels.py -l intermediate_files/prep_dataset/annotations/labels.$ont -o intermediate_files/use_dataset/annotations/
    done
fi

if [ $bpo -eq 1 ]
then
    $src/train.py -a $annots -e $embeds -s $savepath -g bpo -f 4 -p 50 -b 32 -z 10
fi

if [ $mfo -eq 1 ]
then
    $src/train.py -a $annots -e $embeds -s $savepath -g mfo -f 4 -p 50 -b 32 -z 10
fi

if [ $cco -eq 1 ]
then
    $src/train.py -a $annots -e $embeds -s $savepath -g cco -f 4 -p 50 -b 32 -z 10
fi
