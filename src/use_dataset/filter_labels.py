#!/usr/bin/python3

import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--labels', required=True)
    parser.add_argument('-e', '--embeds', required=True)
    parser.add_argument('-o', '--ont', required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_args()
    labs_dir = args['labels']
    embeds_dir = args['embeds']
    ont = args['ont']

    labs_dir = os.path.join(labs_dir, ont)
    prots = {x.split('.')[0] for x in os.listdir(embeds_dir)}
    labs = {y.split('.')[0] for y in os.listdir(labs_dir)}

    labs_only = labs - prots

    print(f'Labels: {len(labs)}')
    print(f'Embeddings: {len(prots)}')
    print(f'Found {len(labs_only)} prots without embeddings')

    for prot in labs_only:
        os.remove(os.path.join(labs_dir, f'{prot}.{ont}'))
