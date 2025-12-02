#!/usr/bin/python3

from collections import deque
import concurrent.futures
from models import *
import math
from itertools import chain
import random
import time
import os
import sys
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import tensorflow as tf
from tensorflow.keras import layers, metrics, callbacks, utils

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_args():
    # Define an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add optional command-line argument
    parser.add_argument('-g', '--grt', required=True)
    parser.add_argument('-e', '--embeddings', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-s', '--savepath', required=True)

    # Parse the command-line arguments and return them as a dictionary
    return vars(parser.parse_args())


def retrieve_model(out_channels, print_summary=False):
    model = get_model_amr(out_channels)

    def myprint(s):
        with open('modelsummary.txt','a') as f:
            print(s, file=f)

    if print_summary:
        model.summary(print_fn=myprint)

    return model


if __name__ == '__main__':
    # Parsing arguments
    args = get_args()
    embed_path = args['embeddings']
    grt_file = args['grt']
    model_weights = args['model']
    save_path = args['savepath']

    print('Loading groundtruth...')
    with open(grt_file, 'r') as fp:
        grt = {}
        for line in fp:
            prot, label = line.strip().split('\t')
            grt[prot] = int(label)

    model = retrieve_model(100, print_summary=True)
    model.load_weights(model_weights)
    preds = {}

    with tqdm(grt.keys(), total=len(grt)) as pbar:
        for prot in pbar:
            x = tf.io.parse_tensor(tf.io.read_file(os.path.join(embed_path, f'{prot}.txt')), out_type=tf.float32)
            #x = tf.reshape(x, [1, x.shape[0], x.shape[1]])
            y_pred = model.predict(x, verbose=0)
            preds[prot] = float(y_pred[0][0])

    with open(os.path.join(save_path, 'predictions.txt'), 'w') as fp:
        for prot, y_pred in preds.items():
            fp.write(f'{prot}\t{y_pred:.2f}\n')
