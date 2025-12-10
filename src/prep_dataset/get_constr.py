#!/usr/bin/python3

import argparse
from taxonLibrary3 import Taxon
import os
from collections import defaultdict
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--const', required=True)
    parser.add_argument('-t', '--taxa', required=True)
    parser.add_argument('-o', '--outfile', required=True)

    return vars(parser.parse_args())


def load_constr(dir):
    files = os.listdir(dir)
    constraints = {}
    for file in files:
        taxon = file.split('_')[0]
        constraints[taxon] = defaultdict(set)
        with open(os.path.join(dir, file), 'r') as fp:
            for line in fp:
                go, desc, ont = line.strip().split('\t')
                constraints[taxon][ont].add(go)

    return constraints


def get_constr(taxa, constr, tax):
    if taxa in constr:
        return taxa

    father = tax.get_father(taxa)
    #print(f'{taxa} -> {father}')
    if not father:
        print(f'Something went wrong with tax: {taxa}...')
        return None

    result = get_constr(father, constr, tax)
    return result


if __name__ == '__main__':
    args = get_args()
    constr_folder = args['const']
    taxa_folder = args['taxa']
    out_file = args['outfile']

    print('Load taxonomy...')
    tax = Taxon(os.path.join(taxa_folder, 'nodes.dmp'), os.path.join(taxa_folder, 'merged.dmp'), os.path.join(taxa_folder, 'names.dmp'))

    print('Loading the rest...')
    constraints = set(load_constr(constr_folder).keys())

    all_taxa = set(tax.get_id_names_map().keys())
    all_constraints = {}
    with tqdm(all_taxa, total=len(all_taxa), desc='Browsing Tax Tree...') as pbar, open(out_file, 'w') as fp:
        for taxon in pbar:
            con = get_constr(taxon, constraints, tax)
            all_constraints[taxon] = con
            fp.write(f'{taxon}\t{con}\n')
