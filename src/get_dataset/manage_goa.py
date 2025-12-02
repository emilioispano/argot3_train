#!/usr/bin/python3

import sys
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import gzip


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--goa', required=True)
    parser.add_argument('-i', '--inter', required=False, default='intermediate/get_dataset/')

    return vars(parser.parse_args())


def parse_goa_line(line, goa_dict):
    if line.startswith('!'):
        return None
    data = line.split('\t')
    if data[0] != 'UniProtKB' or  data[3] == 'NOT' or data[4] in ['GO:0005575', 'GO:0008150', 'GO:0003674'] or data[6] == 'ND' or data[11] != 'protein':
        return None
    if data[6] != 'IEA':
        goa_dict[data[4]].append([data[1], 0, data[12]])
    elif data[7].startswith('UniProtKB-KW'):
        goa_dict[data[4]].append([data[1], 1, data[12]])
    else:
        goa_dict[data[4]].append([data[1], 2, data[12]])


def read_goa(goa_file):
    goa_dict = defaultdict(list)
    with gzip.open(goa_file, 'rt') as fp:
        for i, line in enumerate(fp):
            parse_goa_line(line, goa_dict)
            if i % 1e6 == 0:
                print(f'line (in millions): {i / 1e6} / 1378', end='\r', file=sys.stderr)
    return goa_dict


def write_goa(outpath, goa_dict):
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    out_goa = os.path.join(outpath, 'go_centric_goa.txt')
    print(f'writing to {out_goa}')
    with open(out_goa, 'w') as out:
        with tqdm(goa_dict.items(), total=len(goa_dict)) as pbar:
            for go, proteins in pbar:
                out.write(f'>{go}\n')
                for prot in proteins:
                    out.write(f'{prot[0]}\t{prot[1]}\t{prot[2]}\n')


if __name__ == '__main__':
    args = get_args()
    inter_path = args['inter']
    goa_file = args['goa']

    go_centric_goa = read_goa(goa_file)

    write_goa(inter_path, go_centric_goa)
