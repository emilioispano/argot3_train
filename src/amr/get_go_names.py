#!/usr/bin/python3

import argparse
from owlLibrary3 import GoOwl


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--infile', required=True)
    parser.add_argument('-w', '--owl', required=True)
    parser.add_argument('-o', '--outfile', required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_args()
    owl_file = args['owl']
    input_file = args['infile']
    output_file = args['outfile']

    owl = GoOwl(owl_file)
    with open(input_file, 'r') as fp, open(output_file, 'w') as out:
        for line in fp:
            go = line.strip()
            go_name = owl.go_single_details(go)['name']
            out.write(f'{go}\t{go_name}\n')
