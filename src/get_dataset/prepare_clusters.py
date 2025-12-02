#!/usr/bin/python3

import os
import argparse
from Bio import SeqIO
import sys
from collections import defaultdict
import gzip


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--infile', required=True)
    parser.add_argument('-l', '--limit', required=False, default=False, type=int)
    parser.add_argument('-u', '--upr', required=True)
    parser.add_argument('-o', '--outpath', required=False, default='intermediate_files/get_dataset/to_cluster/', type=str)

    return vars(parser.parse_args())


def get_gos(infile, limit):
    go_terms = defaultdict(set)
    for line in open(infile, 'r'):
        if line.startswith('>'):
            skip = False
            data = line.split('|')
            go_term = data[0].lstrip('>').replace(':', '_')
            count = int(data[2].split('=')[-1])
            print(f'{go_term} {len(go_terms)} / 30289', end='\r', file=sys.stderr)
            continue
        if skip:
            continue
        if limit and len(go_terms[go_term]) > limit:
            print(f'reached limit of {limit} annotations for {go_term}')
            skip = True
        elif count == 0:
            print(f'{go_term} has 0 good annotations')
            skip = True
        elif len(go_terms[go_term]) >= count:
            skip = True
        else:
            go_terms[go_term].add(line.split('\t')[0])

    return go_terms


def reversed(gos_prots, lim):
    prots_gos = defaultdict(set)

    for i, (go, prots) in enumerate(gos_prots.items()):
        print(f'{go} {i} / {len(gos_prots)}', end='\r', file=sys.stderr)
        if len(prots) > lim:
            print(f'{go} has {len(prots)} annotations')
        for prot in prots:
            prots_gos[prot].add(go)

    return prots_gos


def write_files(uniprot, out_path, prot_terms):
    prots = set(prot_terms.keys())
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    gos_records = defaultdict(list)
    with gzip.open(uniprot, 'rt') as fp:
        c = 1
        for record in SeqIO.parse(fp, "fasta"):
            if c % 100000 == 0:
                print(f'record: {c / 100000} / 1587', end='\r', file=sys.stderr)
            c += 1
            record_id = record.id.split('|')[1]
            if record_id in prots:
                for go in prot_terms[record_id]:
                    # Collect the record for the corresponding ID
                    gos_records[go].append(record.format("fasta"))

    # Write the collected records to separate files
    for i, (go, records) in enumerate(gos_records.items()):
        print(f'Writing {go} {i} / 30289')
        with open(out_path + go + '.fasta', 'a') as file:
            file.write("".join(records))


if __name__ == '__main__':
    print('Parsing Arguments...', file=sys.stderr)
    args = get_args()
    limit = args['limit']
    infile = args['infile']
    upr = args['upr']
    outpath = args['outpath']

    print(f'Getting good annotations with {limit} limit...', file=sys.stderr)
    go_terms = get_gos(infile, limit)

    print('Creating protein-centric dictionary...', file=sys.stderr)
    prot_terms = reversed(go_terms, 50000)

    print(f'Reading {upr} to write in {outpath}...', file=sys.stderr)
    write_files(upr, outpath, prot_terms)
