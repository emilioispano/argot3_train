#!/usr/bin/bash

usage() {
    echo "Usage: $0 [-u UNIPROT] [-g GOA] [-t THREADS] [-r ROOTDIR]" 1>&2
    echo "-h prints this help message"
    echo "-u uniprot file"
    echo "-g goa file"
    echo "-t threads number"
    echo "-r root working directory"
    echo "-a skips the goa managing"
    echo "-b skips the go sorting"
    echo "-c skips the cluster preparation"
    echo "-d skips the cluster creation"
    exit 1
}

cluster_go() {
    file=$(basename "$1")
    fname=${file%.*}
    outdir=intermediate_files/get_dataset/clusters
    indir=intermediate_files/get_dataset/to_cluster
    cdhit=/data/exec/cdhit/cd-hit
    echo $file

    mkdir -p $outdir/$fname
    cd-hit -i $indir/$file -o $outdir/$fname/db_90 -c 0.9 -n 5 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T 16 -M 0 > $outdir/$fname/db_90.log
    cd-hit -i $outdir/$fname/db_90 -o $outdir/$fname/${fname}_60.fasta -c 0.6 -n 4 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T 16 -M 0 > $outdir/$fname/db_60.log
}

# Set default options
manage_goa=0
sort_go=0
prepare_clust=0
make_clust=1


# Options parser
while getopts "h:u:g:t:r:abcd" opt
do
    case "${opt}" in
        u) upr_file=${OPTARG};;
        g) goa_file=${OPTARG};;
        t) threads=${OPTARG};;
        r) root_dir=${OPTARG};;
        a) manage_goa=0;;
        b) sort_go=0;;
        c) prepare_clust=0;;
        d) make_clust=0;;
        h | *) usage;;
    esac
done

if [ -z $upr_file ] || [ -z $goa_file ] || [ -z $threads ] || [ -z $root_dir ]; then
    usage
else
    data=$root_dir/intermediate_files/get_dataset
    src=$root_dir/src/get_dataset

    mkdir -p $data

    if [ $manage_goa -eq 1 ]; then
        echo "Managing the GOA from ${goa_file} into go-centric..."
        python3 $src/manage_goa.py -g $goa_file -i $data
    fi

    if [ $sort_go -eq 1 ]; then
        echo "Sorting the go-centric GOA according to annotation quality..."
        python3 $src/sort_go_centric.py -g $data/go_centric_goa.txt -o $data/go_centric_sorted.txt
    fi

    if [ $prepare_clust -eq 1 ]; then
        echo "Preparing to cluster..."
        python3 $src/prepare_clusters.py -i $data/go_centric_sorted.txt -l 50000 -u $upr_file -o $data/to_cluster/ > $data/prepare_clusters.log
        grep good $data/prepare_clusters.log |awk '{print $1}' |sort |uniq > $data/without_good.txt
        big=$(grep -v good $data/prepare_clusters.log |awk '{print $1}' |sort |uniq | wc -l)
        echo "There are ${big} GOs with more than 50000 proteins, suspended"

        echo "Arranging the remaning GOs to be clustered in ${data}/to_cluster/..."
        for go in $(ls $data/to_cluster/)
        do
            c=$(grep '>' $data/to_cluster/$go |wc -l)
            echo "${go%.fasta}" $c >> $data/count_to_cluster.txt
        done
        sort -k2nr $data/count_to_cluster.txt > $data/count_to_cluster_sorted.txt
        mv $data/count_to_cluster_sorted.txt $data/count_to_cluster.txt
        mkdir -p $data/very_big/to_cluster/
        for go_big in $(head -n $big $data/count_to_cluster.txt |awk '{print $1}')
        do
            mv $data/to_cluster/$go_big.fasta $data/very_big/to_cluster/
        done
    fi

    if [ $make_clust -eq 1 ]; then
        echo "Performing the clusterisation in ${data}/clusters/..."
        path_to_files=$data/to_cluster/
        max_processes=8

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
            cluster_go $file &
        done

        # Wait for all background jobs to finish
        wait
        # bash $src/make_clusters.sh

        echo "Counting the representatives for each GO..."
        for go in $(ls $data/clusters)
        do
            c=$(grep '>' $data/clusters/$go/${go}_60.fasta |wc -l)
            echo $go $c >> $data/count_clusters.txt
        done
        sort -k2nr $data/count_clusters.txt > $data/count_clusters_sorted.txt
        mv $data/count_clusters_sorted.txt $data/count_clusters.txt
        mkdir -p $data/final_clusters/
        for go in $(ls $data/clusters)
        do
            cp $data/clusters/$go/${go}_60.fasta $data/final_clusters/$go.fasta
        done

        echo "For GOs with more than 100 proteins, selecting the first 100..."
        mkdir -p $data/very_big/clusters/
        for go in $(awk '$2 > 100 {print $1}' $data/count_clusters.txt)
        do
            mv $data/final_clusters/$go.fasta $data/very_big/clusters/
            awk '/^>/ {n++; if(n>100) exit;} {print}' $data/very_big/clusters/$go.fasta > $data/final_clusters/$go.fasta
        done
    fi
fi
