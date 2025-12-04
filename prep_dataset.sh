#!/usr/bin/bash

usage() {
    echo "Usage: $0 [-c TO_CLUSTER] [-b OBO] [-t THREADS] [-r ROOTDIR]" 1>&2
    echo "-h prints this help message"
    echo "-c intermediate files with clusters"
    echo "-u UNIPROT file"
    echo "-g go-centric GOA"
    echo "-f original GAF file (may be gzipped)"
    echo "-w owl file"
    echo "-t threads number"
    echo "-r root working directory"
    echo "-k skips the cluster reading"
    echo "-x skips the embeddings calculation"
    echo "-y skips the go-related statistics"
    echo "-z skips the annotations saving"
    exit 1
}

# Default options
cluster_path=intermediate_files/get_dataset/final_clusters
go_centric=intermediate_files/get_dataset/go_centric_goa.txt
go_owl=/data/GOA/2025-03-08/go.owl
gaf_file=/data/GOA/2025-03-08/goa_uniprot_all.gaf.gz
upr_file=/data/MONGO/2025-03/file_for_mongoDB/uniprot_all.fasta.gz
threads=1

read_clustres=1
save_embed=1
gos_stats=1
save_annots=1

# Options parser
while getopts "h:c:u:g:w:f:t:r:kxyz" opt
do
    case "${opt}" in
        c) cluster_path=${OPTARG};;
        u) upr_file=${OPTARG};;
        w) go_owl=${OPTARG};;
        f) gaf_file=${OPTARG};;
        g) go_centric=${OPTARG};;
        t) threads=${OPTARG};;
        r) root_dir=${OPTARG};;
        k) read_clustres=0;;
        x) save_embed=0;;
        y) gos_stats=0;;
        z) save_annots=0;;
        h | *) usage;;
    esac
done

if [ -z $cluster_path ] || [ -z $go_owl ] || [ -z $go_centric ] || [ -z $gaf_file ] || [ -z $upr_file ]; then
    usage
else
    echo 'root dir: ' $root_dir
    data=$root_dir/intermediate_files/prep_dataset
    mkdir -p $data
    echo 'data: ' $data
    echo 'cluster path: ' $cluster_path
    echo 'owl file: ' $go_owl
    echo 'GAF file: ' $gaf_file
    echo 'go-centric GOA: ' $go_centric

    src=$root_dir/src/prep_dataset

    if [ $read_clustres -eq 1 ]
    then
        #echo "Creating the files prots_per_go.txt, gos_per_prot.txt in ${data}..."
        #for file in $(ls $cluster_path)
        #do
        #    echo ${file%.fasta} >> $data/prots_per_go.txt
        #    grep '>' $cluster_path/$file >> $data/prots_per_go.txt
        #done
        #python3 $src/rearrange.py -i $data/prots_per_go.txt -o $data/proteins_list.txt
        #python3 $src/get_groundtruth_gaf.py -i $data/proteins_list.txt -g $gaf_file -o $data/gos_per_prot.txt
        python3 $src/get_fastas_uniprot.py -i $data/proteins_list.txt -u $upr_file -o $data/uids.fasta
    fi

    if [ ${save_embed} -eq 1 ]
    then
        echo "Creating the protein embeddings in ${data}/embeddings/..."
        mkdir $data/embeddings/
        mkdir $root_dir/embeddings_torch/
        mkdir $root_dir/embeddings_tf/

        python3 $src/extract.py esm2_t33_650M_UR50D $data/uids.fasta $data/embeddings/ --repr_layers 33 --include per_tok --truncation_seq_length 20000
        echo "Re-calculating for those on which gpu failed..."
        for file in $(ls $data/embeddings)
        do
            echo "${file%.*}" >> $data/uids_ok.txt
        done
        sort $data/uids_ok.txt |uniq > $data/uids_ok_sorted.txt
        mv $data/uids_ok_sorted.txt $data/uids_ok.txt
        awk '{print $1}' $data/gos_per_prot.txt |sort |uniq > $data/uids.txt
        comm -13 $data/uids_ok.txt $data/uids.txt > $data/uids_fail.txt
        python3 $src/get_fastas_uniprot.py -i $data/uids_fail.txt -o $data/uids_fail.fasta -u $data/uids.fasta
        rm $data/uids_fail.fasta.clean
        python3 $src/extract.py esm2_t33_650M_UR50D $data/uids_fail.fasta $data/embeddings/ --repr_layers 33 --include per_tok --truncation_seq_length 20000 --nogpu

        python3 $src/convert_to_tf.py -e $data/embeddings/ -o $root_dir/embeddings_tf/

        for file in $(ls $data/embeddings)
        do
            mv $data/embeddings/$file $root_dir/embeddings_torch/$file
        done
        rm -rf $data/embeddings
    fi

    if [ $gos_stats -eq 1 ]
    then
        python3 $src/go_stats.py -p $data/gos_per_prot.txt -g $go_centric
    fi

    if [ $save_annots -eq 1 ]
    then
        python3 $src/save_go_onts.py -i $data/gos_per_prot.txt -g $go_owl -o $data/annotations/
    fi
fi
