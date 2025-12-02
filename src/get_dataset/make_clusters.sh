#!/usr/bin/bash

cluster_go() {
	file=$(basename "$1")
        outdir="intermediate_files/get_dataset/clusters/"
        indir="intermediate_files/get_dataset/to_cluster/"
        cdhit="/data/exec/cdhit/cd-hit"
	echo $file
	mkdir -p "${outdir}${file%.*}"
	cd-hit -i "${indir}${file}" -o "${outdir}${file%.*}/db_90" -c 0.9 -n 5 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T 16 -M 0 > "${outdir}${file%.*}/db_90.log"
	cd-hit -i "${outdir}${file%.*}/db_90" -o "${outdir}${file%.*}/${file%.*}_60.fasta" -c 0.6 -n 4 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T 16 -M 0 > "${outdir}/${file%.*}/db_60.log"
}

usage() {
	echo "Usage: $0 [-g FASTA_GO] [-p PROC] [-t THREADS]" 1>&2
	echo "-h prints this help message"
	echo "-g the path where the go fasta files are (default: intermediate_files/get_dataset/to_cluster/)"
	echo "-p number of processes to use (default: 8)"
	exit 1
}

#Default options
path_to_files="intermediate_files/get_dataset/to_cluster/"
max_processes=8

# Options parser
while getopts "h:g:p:" opt
do
	case "${opt}" in
		g) path_to_files=${OPTARG};;
		p) max_processes=${OPTARG};;
		h | *) usage;;
	esac
done

# Initialize the semaphore
semaphore=""

# Function to release a slot in the semaphore
release_semaphore () {
	semaphore="${semaphore%x}"
}

# Trap the SIGCHLD signal to wait for child processes to finish
trap release_semaphore SIGCHLD

# Iterate through all files in the directory
for file in "$path_to_files"/*
do
	# Wait for a free slot in the semaphore
	while [ "$(jobs -r | wc -l)" -ge "$max_processes" ]
	do
		sleep 0.25
	done

	# Acquire a slot in the semaphore
	semaphore="$semaphore x"

	# Apply your function to the file in the background
	cluster_go "$file" &
done

# Wait for all background jobs to finish
wait
