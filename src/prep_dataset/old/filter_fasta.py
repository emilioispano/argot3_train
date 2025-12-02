#!/usr/bin/python3

import sys
from Bio import SeqIO
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='input fasta to be filtered', required=True)
    parser.add_argument('-o', '--output', help='output filtered fasta file', required=True)

    return vars(parser.parse_args())


def filter_fasta(input_file, output_file):
    unique_entries = set()
    with open(output_file, 'w') as output_fasta:
        for record in SeqIO.parse(input_file, "fasta"):
            if record.id not in unique_entries:
                # Write the current entry to the output file
                SeqIO.write(record, output_fasta, "fasta")
                # Update the set of unique entries
                unique_entries.add(record.id)

    print(f"Filtered FASTA file has been filtered to {output_file}")


if __name__ == '__main__':
    args = get_args()
    infile = args['input']
    outfile = args['output']

    print(f'Parsing {infile} removing repetitions...', file=sys.stderr)
    filter_fasta(infile, outfile)
