#!/usr/bin/python3

import sys
import os
import argparse
import gzip
from taxonLibrary3 import Taxon

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--goa', required=True)
    parser.add_argument('-t', '--taxon', required=True)
    parser.add_argument('-o', '--out', required=True)

    return vars(parser.parse_args())


def parse_taxonomy(tax_fold):
    nodes = os.path.join(tax_fold, 'nodes.dmp')
    merged = os.path.join(tax_fold, 'merged.dmp')
    names = os.path.join(tax_fold, 'names.dmp')

    tax = Taxon(nodes, merged, names)

    return tax.get_all_descendants('2'), tax


if __name__ == '__main__':
    args = get_args()
    out_dir = args['out']
    taxonomy_folder = args['taxon']
    goa_file = args['goa']

    print('Parsing taxonomy...')
    bacteria, taxon = parse_taxonomy(taxonomy_folder)

    res, nres = set(), set()
    fp = gzip.open(goa_file, 'rt') if goa_file.endswith('.gz') else open(goa_file, 'r')
    for i, line in enumerate(fp):
        if i % 1e6 == 0:
            print(f'line (in millions): {i / 1e6}', end='\r', file=sys.stderr)

        if line.startswith('!'):
            continue

        data = line.split('\t')

        if data[0] != 'UniProtKB' or  data[3] == 'NOT' or data[6] == 'ND' or data[11] != 'protein':
            continue

        tax = data[12]
        if '|' in tax:
            tax = tax.split('|')[0]
        taxon = tax.split(':')[-1]
        if taxon not in bacteria:
            continue

        if data[4] in ['GO:0017001', 'GO:0030655', 'GO:0046677']:
            res.add(data[1])
        else:
            nres.add(data[1])
    fp.close()

    nres = nres - res
    with open(os.path.join(out_dir, 'resistant.txt'), 'w') as r, open(os.path.join(out_dir, 'nonresistant.txt'), 'w') as n:
        for prot in res:
            r.write(f'{prot}\t1\n')
        for prot in nres:
            n.write(f'{prot}\t0\n')
