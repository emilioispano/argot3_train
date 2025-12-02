#!/usr/bin/python3

import argparse
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pred', required=True)
    parser.add_argument('-g', '--grt', required=True)
    parser.add_argument('-f', '--fig', required=False, default='')
    parser.add_argument('-o', '--outfile', required=False, default='')

    return vars(parser.parse_args())


def load_file(infile, scores=False):
    results = defaultdict(set)
    with open(infile, 'r') as fp:
        for line in fp:
            data = line.strip().split('\t')
            try:
                prot, go = data[0], data[1]
            except IndexError:
                print(infile)
                print(line)
                continue
            if scores:
                score = float(data[2])
                to_add = (go, score)
            else:
                to_add = go
            results[prot].add(to_add)
    return results


if __name__ == '__main__':
    args = get_args()
    pred_file = args['pred']
    grt_file = args['grt']
    output_fig = args['fig']
    output_file = args['outfile']

    preds = load_file(pred_file, scores=True)
    grt = load_file(grt_file)

    prs, rcs, fmaxs = [0 for _ in range(101)], [0 for _ in range(101)], [0 for _ in range(101)]
    for t in range(101):
        th = t/100
        tp, fp, fn = 0, 0, 0
        for prot, gos in preds.items():
            for (go, score) in gos:
                if score >= th:
                    if go in grt[prot]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if go in grt[prot]:
                        fn += 1
        try:
            pr = tp / (tp + fp)
            rc = tp / (tp + fn)
            fmax = (2 * pr * rc) / (pr + rc)
        except ZeroDivisionError:
            pr, rc, fmax = 0, 0, 0
        print(f'Th: {th}. Fmax: {fmax:.3f} - Pr: {pr:.3f} - Rc: {rc:.3f}')
        prs[t] = pr
        rcs[t] = rc
        fmaxs[t] = fmax
