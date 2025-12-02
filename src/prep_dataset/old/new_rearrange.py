#!/usr/bin/python3

import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Rearrange file data')

    parser.add_argument('-i', '--input', type=str, help='Input file name')
    parser.add_argument('-o', '--output', type=str, help='Output file name')

    return vars(parser.parse_args())


def read_in(infile):
    data = {}

    with open(infile, 'r') as f:
        print(f'rearranging file {infile}...', file=sys.stderr)
        for line in f:
            if line.startswith('GO'):
                go = line.strip()
            else:
                prot= line.strip().split('|')[1]
                if prot not in data:
                    data[prot] = set()
                data[prot].add(go)

    return data


def write_out(data, outfile):
    with open(outfile, 'w') as f:
        print(f'Writing data to {outfile}...', file=sys.stderr)
        for prot, gos in data.items():
            for go in gos:
                f.write(f'{prot}\t{go}\n')


if __name__ == '__main__':
    args = get_args()
    in_file = args['input']
    out_file = args['output']

    gos_per_prot = read_in(in_file)
    write_out(gos_per_prot, out_file)
