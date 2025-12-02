#!/usr/bin/python3

from owlLibrary3 import GoOwl
import argparse
import os
import networkx as nx
from collections import defaultdict
import pickle
import sys
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-g', '--owl', required=True)
    parser.add_argument('-o', '--output', required=True)

    return vars(parser.parse_args())


def get_annots(infile):
    with open(infile, 'r') as fp:
        annots = defaultdict(set)
        for line in fp:
            prot, go = line.strip().split('\t')
            annots[prot].add(go.replace(':', '_'))

    return annots


def sort_ontology(root, owl):
    nx_ont = nx.DiGraph()
    gos_ont = owl.get_descendants_id(root, by_ontology=True, valid_edges=True)

    for go in gos_ont:
        nx_ont.add_node(go)

    for go in gos_ont:
        for go_son in owl.get_children_id(go, by_ontology=True, valid_edges=True):
            nx_ont.add_edge(go, go_son)

    return list(nx.topological_sort(nx_ont))


def propagate_gos(annots, owl):
    propagated_annots = {}
    with tqdm(annots.items(), total=len(annots)) as pbar:
        for prot, gos in pbar:
            propagated_gos = gos.copy()
            for go in gos:
                propagated_gos.update(owl.get_ancestors_id(go, by_ontology=True, valid_edges=True))
            propagated_annots[prot] = propagated_gos

    return propagated_annots


def link_prots(bpo, mfo, cco, annots):
    bpo_annots, mfo_annots, cco_annots = {}, {}, {}

    with tqdm(annots.items(), total=len(annots)) as pbar:
        for prot, gos in pbar:
            bpo_annots[prot] = [1 if go in gos else 0 for go in bpo]
            mfo_annots[prot] = [1 if go in gos else 0 for go in mfo]
            cco_annots[prot] = [1 if go in gos else 0 for go in cco]

    return bpo_annots, mfo_annots, cco_annots


def write_out(outpath, annots, ont):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    print(f'All together in labels.{ont}')
    with open(os.path.join(outpath, f'labels.{ont}'), 'wb') as fp:
        pickle.dump(annots, fp)

    print(f'One by one in PROTNAME.{ont}')
    for prot, gos in annots.items():
        with open(os.path.join(outpath, f'{prot}.{ont}'), 'wb') as fp:
            pickle.dump(gos, fp)


if __name__ == '__main__':
    args = get_args()
    prots_gos = args['input']
    owl_file = args['owl']
    annots_path = args['output']

    print('Getting annotations...')
    annots = get_annots(prots_gos)

    print('Reading owl file...')
    owl = GoOwl(owl_file)

    print('Sorting gos per ontology...')
    bpo = sort_ontology('GO_0008150', owl)
    print(f'Biological process: found {len(bpo)} GO terms')
    mfo = sort_ontology('GO_0003674', owl)
    print(f'Molecular function: found {len(mfo)} GO terms')
    cco = sort_ontology('GO_0005575', owl)
    print(f'Cellular component: found {len(cco)} GO terms')

    print('Propagating gos...')
    propagated_annots = propagate_gos(annots, owl)

    print('Writing orders...')
    with open('BPO_order.txt', 'w') as fp:
        fp.write('\n'.join(bpo))
    with open('MFO_order.txt', 'w') as fp:
        fp.write('\n'.join(mfo))
    with open('CCO_order.txt', 'w') as fp:
        fp.write('\n'.join(cco))

    print('Linking proteins to gos...')
    bpo, mfo, cco = link_prots(bpo, mfo, cco, propagated_annots)

    print('Writing bpo...')
    write_out(annots_path, bpo, 'bpo')
    print('Writing mfo...')
    write_out(annots_path, mfo, 'mfo')
    print('Writing cco...')
    write_out(annots_path, cco, 'cco')
