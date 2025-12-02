#!/usr/bin/python3

from models import *
import math
import os
import sys
from tqdm import tqdm
import numpy as np
import argparse
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ONT_LENS = {'bpo': 27185,
            'cco': 4057,
            'mfo': 11197}


def get_args():
    # Define an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add optional command-line argument
    parser.add_argument('-e', '--embeddings', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-g', '--ontology', required=False, default='')
    parser.add_argument('-c', '--cco', required=False, default='order/CCO_order.txt')
    parser.add_argument('-m', '--mfo', required=False, default='order/MFO_order.txt')
    parser.add_argument('-p', '--bpo', required=False, default='order/BPO_order.txt')
    parser.add_argument('-1', '--cc_model', required=False, default='models/cco_model.h5')
    parser.add_argument('-2', '--mf_model', required=False, default='models/mfo_model.h5')
    parser.add_argument('-3', '--bp_model', required=False, default='models/bpo_model.h5')

    # Parse the command-line arguments and return them as a dictionary
    return vars(parser.parse_args())


def retrieve_model(ont, out_channels, print_summary=False):
    if ont == 'cco':
        model = get_model_cco(out_channels)
    elif ont == 'mfo':
        model = get_model_mfo(out_channels)
    elif ont == 'bpo':
        model = get_model_bpo(out_channels)

    def myprint(s):
        with open('modelsummary.txt','a') as f:
            print(s, file=f)

    if print_summary:
        model.summary(print_fn=myprint)

    return model


def get_order(ontology):
    with open(ontology, 'r') as fp:
        return [x.strip() for x in fp.readlines()]


if __name__ == '__main__':
    # Parsing arguments
    args = get_args()
    embed_path = args['embeddings']
    ont = args['ontology']
    bpo_order = args['bpo']
    mfo_order = args['mfo']
    cco_order = args['cco']
    bpo_w = args['bp_model']
    mfo_w = args['mf_model']
    cco_w = args['cc_model']
    pred_dir = args['output']

    ont_weigths = {'cco': cco_w,
                   'mfo': mfo_w,
                   'bpo': bpo_w}

    ont_orders = {'cco': get_order(cco_order),
                  'mfo': get_order(mfo_order),
                  'bpo': get_order(bpo_order)}

    prots = [x.split('.')[0] for x in os.listdir(embed_path)]
    preds = {}
    onts = [ont] if ont else ['cco', 'mfo', 'bpo']
    for ont in onts:
        preds[ont] = {}
        model = retrieve_model(ont, ONT_LENS[ont])
        model.load_weights(ont_weigths[ont])

        X = {}
        for prot in prots:
            prot_file = os.path.join(embed_path, f'{prot}.txt')
            X[prot] = tf.io.parse_tensor(tf.io.read_file(prot_file), out_type=tf.float32)

        with tqdm(X.items(), total=len(X), desc=f'Predicting {ont}...') as pbar:
            for prot, x in pbar:
                preds[ont][prot] = model.predict(x, verbose=0)

        with open(os.path.join(pred_dir, f'{ont}_raw.txt'), 'w') as fp:
            for prot, pred in preds[ont].items():
                for go, score in zip(ont_orders[ont], pred.flatten()):
                    if float(score) > 0.01:
                        fp.write(f'{prot}\t{go}\t{score:.2f}\n')
