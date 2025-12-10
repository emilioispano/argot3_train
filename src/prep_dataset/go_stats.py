#!/usr/bin/python3

import sys
import argparse
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prots', help='gos_per_prots file', required=True)
    parser.add_argument('-g', '--goa', help='goa file', required=True)
    parser.add_argument('-l', '--log', help='log file', required=False, default='intermediate_files/prep_dataset/go_stats.log')

    return vars(parser.parse_args())


def get_core(prots):
    with open(prots, 'r') as fp:
        core = defaultdict(set)
        for line in fp:
            prot, go = line.strip().split('\t')
            core[prot].add(go)
    return core


def get_annots(goa, prots):
    alles = defaultdict(set)
    good = defaultdict(set)

    with open(goa, 'r') as fp:
        for i, line in enumerate(fp):
            if i % 1e6 == 0:
                print(f'line {i / 1e6} over 1004 c.ca', end='\r', file=sys.stderr)
            if line.startswith('>'):
                go = line.split('|')[0].strip('>')
            else:
                prot, code, _ = line.split('\t')
                if prot in prots:
                    alles[prot].add(go.replace(':', '_'))
                    if int(code) < 2:
                        good[prot].add(go.replace(':', '_'))

    return alles, good


def compare_annots(core, good, alles, outfile):
    core_freq, good_freq, all_freq = {}, {}, {}

    for prot, gos in core.items():
        for go in gos:
            if go not in core_freq:
                core_freq[go] = set()
            core_freq[go].add(prot)
    for prot, gos in good.items():
        for go in gos:
            if go not in good_freq:
                good_freq[go] = set()
            good_freq[go].add(prot)
    for prot, gos in alles.items():
        for go in gos:
            if go not in all_freq:
                all_freq[go] = set()
            all_freq[go].add(prot)

    good_only = set(good_freq.keys()) - set(core_freq.keys())
    alles_only = set(all_freq.keys()) - set(good_freq.keys())

    with open(outfile, 'w') as out:
        for go in core_freq.keys():
            skip = False
            try:
                lc = len(core_freq[go])
            except KeyError:
                print(f'youd better sleep...')
                skip = True
            try:
                lg = len(good_freq[go])
            except KeyError:
                print(f'somehow, go {go} is in core but not in good')
                skip = True
            try:
                la = len(all_freq[go])
            except KeyError:
                print(f'somehow, go {go} is in core but not in all')
                skip = True
            if not skip:
                out.write(f'+\tgo {go}\t{len(core_freq[go])}\t{len(good_freq[go])}\t{len(all_freq[go])}\n')
        for go in good_only:
            out.write(f'#\tgo {go} among good annotations but never selected\n')
        for go in alles_only:
            out.write(f'#\tgo {go} among IEA annotations but never selected\n')

        for prot in core.keys():
            try:
                good_cov = (len(core[prot]) / len(good[prot])) * 100
                all_cov = (len(core[prot]) / len(alles[prot])) * 100
                cross_cov = (len(good[prot]) / len(alles[prot])) * 100
                out.write(f'@\tprot {prot} with core {len(core[prot])}: {good_cov:.2f}% of good,\t{all_cov:.2f}% of all\n')
                out.write(f'@\tprot {prot} with good {len(good[prot])}: {cross_cov:.2f}% of all\n')
            except ZeroDivisionError:
                out.write(f'@\tprot {prot} has problems\n')
                print(f'{prot} has problems')


if __name__ == '__main__':
    args = get_args()

    core_annots = get_core(args['prots'])
    all_annots, good_annots = get_annots(args['goa'], set(core_annots.keys()))

    compare_annots(core_annots, good_annots, all_annots, args['log'])
