#!/usr/bin/bash

usage() {
    echo "Usage: $0 [-c TO_CLUSTER] [-b OBO] [-t THREADS] [-r ROOTDIR]" 1>&2
    echo "-h prints this help message"
    echo "-c intermediate files to cluster"
    echo "-b obo file"
    echo "-t threads number"
    echo "-r root working directory"
    echo "-k skips the cluster reading"
    echo "-x skips the embeddings calculation"
    echo "-y skips the go-related statistics"
    echo "-z skips the annotations saving"
    exit 1
}

# Default options
cdhit_clust="intermediate_files/get_dataset/clusters/"
data="intermediate_files/prep_dataset/"
cluster_path="intermediate_files/get_dataset/final_clusters/"
go_centric="intermediate_files/get_dataset/go_centric_goa.txt"
go_owl="/data/GOA/2024-02-09/go.owl"
threads=32

read_clustres=1
save_embed=1
gos_stats=1
save_annots=1

# Options parser
while getopts "h:c:b:t:r:kxyz" opt
do
    case "${opt}" in
        c) cluster_path=${OPTARG};;
        b) go_owl=${OPTARG};;
        t) threads=${OPTARG};;
        r) root_dir=${OPTARG};;
        k) read_clustres=0;;
        x) save_embed=0;;
        y) gos_stats=0;;
        z) save_annots=0;;
        h) usage;;
    esac
done

if [ -z "${cluster_path}" ] || [ -z "${threads}" ] || [ -z "${root_dir}" ]; then
    usage
else
    echo 'root dir: ' $root_dir
    echo 'data: ' $data
    echo 'cluster path: ' $cluster_path
    data="${root_dir}${data}"
    src="${root_dir}src/prep_dataset"
    mkdir -p $data

    cluster_path="${root_dir}${cluster_path}"

    if [ ${read_clustres} -eq 1 ]
    then
        echo "Creating the files prots_per_go.txt, gos_per_prot.txt in ${data}..."
        for file in $(ls ${cluster_path})
        do
            echo "${file%.fasta}" >> ${data}prots_per_go.txt
            grep '>' ${cluster_path}${file} >> ${data}prots_per_go.txt
        done
        python3 $src/rearrange.py -i ${data}prots_per_go.txt -o ${data}proteins_list.txt
        python3 $src/get_groundtruth_gaf.py -i ${data}proteins_list.txt -g /data/GOA/2024-02-09/goa_uniprot_all.gaf -o ${data}gos_per_prot.txt
        python3 $src/get_fastas.py -p ${data}proteins_list.txt -o ${data}uids.fasta -t 32
    fi

    if [ ${save_embed} -eq 1 ]
    then
        echo "Creating the protein embeddings in ${data}embeddings/..."
        mkdir ${data}embeddings/
        mkdir ${root_dir}embeddings_torch/
        mkdir ${root_dir}embeddings_tf/

        python3 $src/extract.py esm2_t33_650M_UR50D ${data}uids.fasta ${data}embeddings/ --repr_layers 33 --include per_tok --truncation_seq_length 20000
        echo "Re-calculating for those on which gpu failed..."
        do
            echo "${file%.*}" >> ${data}uids_ok.txt
        done
        sort ${data}uids_ok.txt |uniq > ${data}uids_ok_sorted.txt
        mv ${data}uids_ok_sorted.txt ${data}uids_ok.txt
        awk '{print $1}' ${data}gos_per_prot.txt |sort |uniq > ${data}uids.txt
        comm -13 ${data}uids_ok.txt ${data}uids.txt > ${data}uids_fail.txt
        python3 $src/get_fastas.py -p ${data}/uids_fail.txt -o ${data}uids_fail.fasta -t 32
        rm ${data}uids_fail.fasta.clean
        python3 $src/extract.py esm2_t33_650M_UR50D ${data}uids_fail.fasta ${data}embeddings/ --repr_layers 33 --include per_tok --truncation_seq_length 20000 --nogpu

        python3 $src/convert_to_tf.py -e ${data}embeddings/ -o ${root_dir}embeddings_tf/

        for file in $(ls ${data}embeddings/)
        do
            mv ${data}embeddings/$file ${root_dir}embeddings_torch/$file
        done
        rm -rf ${data}embeddings
    fi

    if [ ${gos_stats} -eq 1 ]
    then
        python3 $src/go_stats.py -p ${data}gos_per_prot.txt -g $go_centric
    fi

    if [ ${save_annots} -eq 1 ]
    then
        python3 $src/save_go_onts.py -i ${data}gos_per_prot.txt -g $go_owl -o ${data}annotations/
    fi
fi
