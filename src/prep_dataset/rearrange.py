#!/usr/bin/python3

import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Rearrange file data')

    parser.add_argument('-i', '--input', type=str, help='Input file name')
    parser.add_argument('-o', '--output', type=str, help='Output file name')

    return vars(parser.parse_args())


def read_in(infile):
    prots = set()

    with open(infile, 'r') as f:
        print(f'rearranging file {infile}...', file=sys.stderr)
        for line in f:
            if line.startswith('GO'):
                c = 0
            else:
                if c >= 100:
                    continue
                prot = line.strip().split('|')[1]
                prots.add(prot)
                c += 1
    return prots


def write_out(data, outfile):
    with open(outfile, 'w') as f:
        print(f'Writing data to {outfile}...', file=sys.stderr)
        for prot in data:
            f.write(f'{prot}\n')


if __name__ == '__main__':
    args = get_args()
    in_file = args['input']
    out_file = args['output']

    prots = read_in(in_file)
    write_out(prots, out_file)
