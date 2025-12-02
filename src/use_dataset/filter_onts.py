#!/usr/bin/python3

import sys
import os
import argparse
from tqdm import tqdm
import pickle


def get_args():
    # Define an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add optional command-line argument
    parser.add_argument('-a', '--annotations', required=True)
    parser.add_argument('-s', '--savepath', required=True)

    # Parse the command-line arguments and return them as a dictionary
    return vars(parser.parse_args())


if __name__ == '__main__':
    # Parsing arguments
    args = get_args()
    annots_path = args['annotations']
    out_path = args['savepath']

    print('Loading labels...')
    anfiles = os.listdir(annots_path)
    for ont in ['cco', 'mfo', 'bpo']:
        ont_path = os.path.join(out_path, ont)
        os.mkdir(ont_path)
        labels = {}
        ont_files = [x for x in anfiles if ont in x]
        len_ont = len(ont_files)
        print(f'Managing ontology: {ont} of length {len_ont}')
        with tqdm(ont_files, total=len(ont_files)) as pbar:
            for file in pbar:
                with open(os.path.join(annots_path, file), 'rb') as fp:
                    annots = pickle.load(fp)
                    if 1 in annots:
                        prot = file.split('.')[0]
                        labels[prot] = annots

        print('Ri-pickleing...')
        with tqdm(labels.items(), total=len(labels)) as pbar:
            for prot, label in pbar:
                with open(os.path.join(ont_path, f'{prot}.{ont}'), 'wb') as out:
                    pickle.dump(label, out)
