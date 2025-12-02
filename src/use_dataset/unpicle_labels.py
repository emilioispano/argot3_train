#!/usr/bin/python3

import tensorflow as tf
import pickle
import argparse
import os
from tqdm import tqdm
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--labels', required=True)
    parser.add_argument('-o', '--output', required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_args()
    input_file = args['labels']
    output_dir = args['output']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ontology = input_file.split('.')[-1]
    ont_dir = os.path.join(output_dir, ontology)
    if not os.path.exists(ont_dir):
        os.mkdir(ont_dir)

    with open(input_file, 'rb') as fp:
        labels_dict = pickle.load(fp)
        with tqdm(labels_dict.items(), total=len(labels_dict), desc='Processing labels...') as pbar:
            for prot, label in pbar:
                prot_file = os.path.join(ont_dir, f'{prot}.{ontology}')
                label = tf.reshape(tf.convert_to_tensor(label), (len(label),))
                tf.io.write_file(prot_file, tf.io.serialize_tensor(label))
