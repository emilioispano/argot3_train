#!/usr/bin/python3

import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--clusters', required=True)
    parser.add_argument('-g', '--goa', required=True)
    parser.add_argument('-o', '--output', required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_args()
    clust_dir = args['clusters']
    goa_sorted = args['goa']
    out_file = args['output']

    clusters = os.listdir(clust_dir)
    gos_clust = {}
    for cluster in clusters:
        go = cluster.split('.')[0]
        go = go.replace(':', '_')
        gos_clust[go] = set()
        with open(os.path.join(clust_dir, cluster), 'r') as fp:
            for line in cluster:
                if line.startswith('>'):
                    prot = line.split('|')[1]
                    gos_clust[go].add(prot)

    gos_sort = {}
    with open(goa_sorted, 'r') as fp:
        for line in fp:
            if line.startswith('>'):
                go = line.lstrip('>').split('|')[0]
                go = go.replace(':', '_')
                gos_sort[go] = {}
            else:
                prot, eco, _ = line.split('\t')
                if prot not in gos_sort[go]:
                    gos_sort[go][prot] = int(eco)
                else:
                    gos_sort[go][prot] = min([int(eco), gos_sort[go][prot]])

    with open(out_file, 'w') as fp:
        for go, prots in gos_clust.items():
            for prot in prots:
                if gos_sort[go][prot] == 0:
                    out.write(f'{go}\t{prot}\t0\n')
            for prot in prots:
                if gos_sort[go][prot] == 1:
                    out.write(f'{go}\t{prot}\t1\n')
            for prot in prots:
                if gos_sort[go][prot] == 2:
                    out.write(f'{go}\t{prot}\t2\n')
