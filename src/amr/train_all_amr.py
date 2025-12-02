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
    parser.add_argument('-l', '--labels', required=True)
    parser.add_argument('-e', '--embeddings', required=True)
    parser.add_argument('-s', '--savepath', required=True)
    parser.add_argument('-p', '--epochs', required=False, default=100, type=int)
    parser.add_argument('-b', '--batch', required=False, default=64, type=int)
    parser.add_argument('-z', '--buffer', required=False, default=8, type=int)
    parser.add_argument('-r', '--resume', required=False, default=0, type=int)

    # Parse the command-line arguments and return them as a dictionary
    return vars(parser.parse_args())


def load_folds(n_folds):
    folds = []
    for i in range(n_folds):
        with open(f'intermediate_files/amr/folds/fold_{i}.txt', 'r') as fp:
            folds.append([x.strip() for x in fp.readlines()])

    return folds


def load_prots(file):
    with open(file, 'r') as fp:
        return [x.strip() for x in fp.readlines()]


def write_prots(prots, file):
    with open(file, 'w') as fp:
        for prot in prots:
            fp.write(f'{prot}\n')


def compute_f(TP, FP, FN):
    precision, recall, f_max = [0 for _ in range(101)], [0 for _ in range(101)], [0 for _ in range(101)]
    for th in range(101):
        if int(TP[th] + FP[th]) == 0:
            precision[th] = 0
            recall[th] = 0
            f_max[th] = 0
        else:
            precision[th] = TP[th] / (TP[th] + FP[th])
            recall[th] = TP[th] / (TP[th] + FN[th])
            f_max[th] = (2 * precision[th] * recall[th]) / (precision[th] + recall[th])

    return precision, recall, f_max


def get_folds(all_prots, test_split):
    fold_size = math.ceil(len(all_prots) * test_split)
    folds = []
    fold = []
    for i, prot in enumerate(all_prots):
        fold.append(prot)
        if len(fold) == fold_size or i == len(all_prots) - 1:
            folds.append(fold)
            fold = []

    return folds


def padding(X):
    max_x = max([tf.shape(tensor)[0] for tensor in X])
    padded_tensors = [tf.pad(tensor, [[0, max_x - tensor.shape[0]], [0, 0]]) for tensor in X]
    return tf.stack(padded_tensors, axis=0)


def load_prot(prot, embed_path, labels):
    prot_file = os.path.join(embed_path, f'{prot}.txt')
    x = tf.io.parse_tensor(tf.io.read_file(prot_file), out_type=tf.float32)
    y = tf.constant([labels[prot]], dtype=tf.float32)  # Ensure y has shape (1,)
    return x, y


def load_batch_buffer(batch, embed_path, labels, buffer):
    X, Y = [], []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_prot, prot, embed_path, labels) for prot in batch]
        results = [future.result() for future in futures]
        for x, y in results:
            if x is not None:
                X.append(x)
                Y.append(y)

    X = padding(X)
    Y = tf.stack(Y, axis=0)
    Y = tf.reshape(Y, [Y.shape[0], 1])
    buffer.append((X, Y))


def yield_batches(prots, batch_size):
    for start in range(0, len(prots), batch_size):
        end = min(start + batch_size, len(prots))
        yield prots[start:end]


def batches_generator_buffer(prots, labels, embed_path, batch_size, buffer_size=10, shuffle=True, loop=False):
    repeat = True
    while repeat:
        if shuffle:
            random.shuffle(prots)

        buffer = deque(maxlen=buffer_size)

        for batch in yield_batches(prots, batch_size):
            if buffer:
                yield buffer.popleft()
                load_batch_buffer(batch, embed_path, labels, buffer)
            load_batch_buffer(batch, embed_path, labels, buffer)
            yield buffer.popleft()

        repeat = loop


def update_metrics(Y_pred, Y_true, TP, FP, FN):
    # Convert lists to NumPy arrays if needed
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    thresholds = np.arange(0, 1.01, 0.01)
    for th_idx, th in enumerate(thresholds):
        TP[th_idx] += np.sum((Y_pred >= th) & (Y_true == 1))
        FP[th_idx] += np.sum((Y_pred >= th) & (Y_true == 0))
        FN[th_idx] += np.sum((Y_pred < th) & (Y_true == 1))

    return TP, FP, FN


def retrieve_model(out_channels, print_summary=False):
    model = get_model_amr(out_channels)

    def myprint(s):
        with open('modelsummary.txt','a') as f:
            print(s, file=f)

    if print_summary:
        model.summary(print_fn=myprint)

    return model


def model_checkpoint(model_summary, TP, FP, FN, epoch):
    pr, rc, fmax = compute_f(TP, FP, FN)
    with open(model_summary, 'a') as out:
        max_f = max(fmax)
        max_th = fmax.index(max_f)
        max_pr = pr[max_th]
        max_rc = rc[max_th]
        out.write(f'Epoch: {epoch + 1}. F_max: {max_f:.4f}, precision: {max_pr:.4f}, recall: {max_rc:.4f} (th: {max_th})\n')


if __name__ == '__main__':
    # Parsing arguments
    args = get_args()
    embed_path = args['embeddings']
    labels_file = args['labels']
    save_path = args['savepath']
    num_epochs = args['epochs']
    batch_size = args['batch']
    buffer_size = args['buffer']
    epoch_resume = args['resume']

    print('Loading labels...')
    with open(labels_file, 'r') as fp:
        labels = {}
        for line in fp:
            prot, label = line.strip().split('\t')
            labels[prot] = int(label)

    print('Loading data...')
    resume = True if epoch_resume else False
    if resume:
        print('Resume mode')
        train_prots = load_prots('intermediate_files/amr/split/train_prots.txt')
        test_prots = load_prots('intermediate_files/amr/split/test_prots.txt')
    else:
        folds = load_folds(5)
        test_prots = folds[-1]
        train_prots = []
        for fold in folds[:-1]:
            train_prots += fold
        write_prots(train_prots, 'intermediate_files/amr/split/train_prots.txt')
        write_prots(test_prots, 'intermediate_files/amr/split/test_prots.txt')

    num_train_batches = int(len(train_prots) / batch_size)
    num_test_batches = int(len(test_prots) / batch_size)

    for epoch in range(num_epochs):
        if resume and epoch < epoch_resume:
            print(f'Epoch {epoch + 1}/{num_epochs} already trained, skipping...')
            continue

        model = retrieve_model(100, print_summary=True)
        if resume:
            model.load_weights(os.path.join(save_path, f'amr_all_{epoch - 1}.h5'))
            resume = False
        else:
            if epoch > 0:
                model.load_weights(os.path.join(save_path, f'amr_all_{epoch - 1}.h5'))

        print(f'Epoch {epoch + 1} / {num_epochs}, training.')
        print(f'Steps: {num_train_batches}')

        train_generator = batches_generator_buffer(train_prots, labels, embed_path, batch_size, buffer_size, loop=True)
        model.fit(train_generator, epochs=3, steps_per_epoch=num_train_batches)

        model.save_weights(os.path.join(save_path, f'amr_all_{epoch}.h5'))

        test_generator = batches_generator_buffer(test_prots, labels, embed_path, batch_size, buffer_size)
        Y_true, Y_pred = [], []

        with tqdm(range(num_test_batches), total=num_test_batches, desc='Testing...') as pbar:
            for X_test, Y_test in test_generator:
                y_pred = model.predict_on_batch(X_test)

                y_pred = y_pred.flatten().tolist()
                Y_test = Y_test.numpy().flatten().tolist()

                Y_true.extend(Y_test)
                Y_pred.extend(y_pred)
                pbar.update(1)

        TP, FP, FN = update_metrics(Y_pred, Y_true, [0 for _ in range(101)], [0 for _ in range(101)], [0 for _ in range(101)])

        model_checkpoint(os.path.join(save_path, f'amr_all_recap.txt'), TP, FP, FN, epoch)

        tf.keras.backend.clear_session()
