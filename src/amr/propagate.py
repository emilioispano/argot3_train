#!/usr/bin/python3

import argparse
from owlLibrary2 import GoOwl


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-g', '--owl', required=True)
    parser.add_argument('-p', '--pred', action='store_true')

    return vars(parser.parse_args())


def parse_prediction(infile, outfile, owl, obs):
    preds = {}
    with open(infile, 'r') as fp:
        for line in fp:
            data = line.strip().split('\t')
            if len(data) == 3:
                prot, go, score = data
            else:
                continue
            score = float(score)
            if prot not in preds:
                preds[prot] = {}
            preds[prot][go] = score

    prop = {}
    for prot in preds.keys():
        prop[prot] = {}
        for go in preds[prot].keys():
            prop[prot][go] = preds[prot][go]
            ancestors = set(owl.go_ancestors_by_ontology_using_valid_edges(go.replace(':', '_')).keys())
            for anc in ancestors:
                if anc in preds[prot]:
                    prop[prot][anc] = max([preds[prot][anc], preds[prot][go]])
                else:
                    prop[prot][anc] = preds[prot][go]

    with open(outfile, 'w') as out:
        for prot in prop.keys():
            for go in prop[prot].keys():
                out.write(f'{prot}\t{go}\t{prop[prot][go]}\n')


def parse_groundtruth(infile, outfile, owl, obs):
    grt = {}
    with open(infile, 'r') as fp:
        for line in fp:
            prot, go = line.strip().split('\t')
            if prot not in grt:
                grt[prot] = set()
            grt[prot].add(go)

    prop = {}
    for prot, gos in grt.items():
        propagation = gos.copy()
        for go in gos:
            ancestors = set(owl.go_ancestors_by_ontology_using_valid_edges(go.replace(':', '_')).keys())
            for anc in ancestors:
                propagation.add(anc)
        prop[prot] = propagation - obs

    with open(outfile, 'w') as out:
        for prot, gos in prop.items():
            for go in gos:
                out.write(f'{prot}\t{go}\n')


if __name__ == '__main__':
    args = get_args()
    annots_file = args['input']
    out_file = args['output']
    owl_file = args['owl']
    flag = args['pred']

    print('Parsing owl...')
    owl = GoOwl(owl_file, namespace='http://purl.obolibrary.org/obo/')
    owl.obsolete_deprecated_new()
    obsolete, deprecated = owl.get_obsolete_deprecated_set()
    bad_gos = obsolete | deprecated

    if flag:
        parse_prediction(annots_file, out_file, owl, bad_gos)
    else:
        parse_groundtruth(annots_file, out_file, owl, bad_gos)
