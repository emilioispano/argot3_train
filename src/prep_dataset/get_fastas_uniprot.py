#!/usr/bin/python3

import argparse
import gzip


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-u', '--upr', required=True)
    parser.add_argument('-o', '--output', required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_args()
    prot_file = args['input']
    upr_file = args['upr']
    out_file = args['output']

    with open(prot_file, 'r') as fp:
        prots = {x.strip() for x in fp.readlines()}

    fp = gzip.open(upr_file, 'rt') if upr_file.endswith('.gz') else open(upr_file, 'r')
    with open(out_file, 'w') as out:
        write = False
        for line in fp:
            if line.startswith('>'):
                prot = line.split('|')[1]
                write = True if prot in prots else False
            if write:
                out.write(line)
    fp.close()
