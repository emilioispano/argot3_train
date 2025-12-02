#!/usr/bin/python3

import argparse
from collections import defaultdict
import gzip


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-g', '--gaf', required=True)
    parser.add_argument('-o', '--output', required=True)

    return vars(parser.parse_args())


def get_truth(prots, gaf):
    fp = gzip.open(gaf, 'rt') if gaf.endswith('.gz') else open(gaf, 'r')
    grt = defaultdict(set)
    for i, line in enumerate(fp):
        if i % 1e6 == 0:
            print(f'Processing line: {int(i/1e6)}')
        if line.startswith('!'):
            continue
        data = line.strip().split('\t')
        if data[6] == 'ND' or data[3] == 'NOT':
            continue
        if data[1] in prots:
            grt[data[1]].add(data[4].replace(':', '_'))
    fp.close()

    return grt


if __name__ == '__main__':
    args = get_args()
    prots_list = args['input']
    gaf_file = args['gaf']
    output_file = args['output']

    with open(prots_list, 'r') as fp:
        proteins = {line.strip() for line in fp.readlines()}

    grt = get_truth(proteins, gaf_file)

    with open(output_file, 'w') as fp:
        for prot, gos in grt.items():
            for go in gos:
                fp.write(f'{prot}\t{go}\n')
