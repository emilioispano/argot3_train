#!/usr/bin/python3

import os
import sys
import argparse
import pickle
import random
import numpy as np
from itertools import chain
from tqdm import tqdm
import tensorflow as tf
from models_attn_med import *

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
print("Number of devices:", strategy.num_replicas_in_sync)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations', required=True)
    parser.add_argument('-e', '--embeddings', required=True)
    parser.add_argument('-g', '--ontology', required=True)
    parser.add_argument('-s', '--savepath', required=True)
    parser.add_argument('-c', '--checkpoint', required=False, default=5, type=int)
    parser.add_argument('-p', '--epochs', required=False, default=100, type=int)
    parser.add_argument('-b', '--batch', required=False, default=64, type=int)
    parser.add_argument('-r', '--resume', required=False, default=0, type=int)
    return vars(parser.parse_args())

def load_prots(file):
    with open(file, 'r') as fp:
        prots = [line.strip() for line in fp]
    return prots

def write_prots(prots, file):
    with open(file, 'w') as fp:
        for prot in prots:
            fp.write(f'{prot}\n')

def load_folds(n_folds, ont):
    folds = []
    for i in range(n_folds + 1):
        with open(f'intermediate_files/use_dataset/folds/{ont}_fold_{i}.txt', 'r') as fp:
            folds.append([x.strip() for x in fp.readlines()])
    return folds

def load_labels(annotations):
    prots = os.listdir(annotations)
    labels_dict = {}

    def load_label(prot):
        prot_file = os.path.join(annotations, prot)
        prot_name = prot.split('.')[0]
        with open(prot_file, 'rb') as fp:
            label = pickle.load(fp)
        return prot_name, np.array(label, dtype=np.float32)

    with tqdm(total=len(prots), desc='Loading Labels', file=sys.stderr) as pbar:
        for prot in prots:
            prot_name, label = load_label(prot)
            labels_dict[prot_name] = label
            pbar.update(1)
    return labels_dict

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

def update_metrics(Y_pred, Y_true, TP, FP, FN):
    thresholds = np.arange(0, 1.01, 0.01)
    for th_idx, th in enumerate(thresholds):
        TP[th_idx] += np.sum((Y_pred >= th) & (Y_true == 1))
        FP[th_idx] += np.sum((Y_pred >= th) & (Y_true == 0))
        FN[th_idx] += np.sum((Y_pred < th) & (Y_true == 1))
    return TP, FP, FN

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

def model_checkpoint(model_summary, TP, FP, FN, epoch, ont):
    pr, rc, fmax = compute_f(TP, FP, FN)
    with open(model_summary, 'a') as out:
        max_f = max(fmax)
        max_th = fmax.index(max_f)
        max_pr = pr[max_th]
        max_rc = rc[max_th]
        out.write(f'Final train, epoch: {epoch}. F_max: {max_f:.4f}, precision: {max_pr:.4f}, recall: {max_rc:.4f} (th: {max_th})\n')

def single_protein_loader(prot, embed_path, labels, out_channels):
    # Function to be wrapped with tf.py_function
    prot_str = prot.numpy().decode()
    prot_file = os.path.join(embed_path, f'{prot_str}.txt')
    try:
        x = tf.io.parse_tensor(tf.io.read_file(prot_file), out_type=tf.float32)
        y = labels[prot_str]
        if x.shape[0] <= 10000:
            return x.numpy(), y
        else:
            return np.zeros([1000, 1280], dtype=np.float32), np.zeros([out_channels], dtype=np.float32)
    except Exception as e:
        # If any issue, return zeros (padding will fix batch shape)
        return np.zeros([1000, 1280], dtype=np.float32), np.zeros([out_channels], dtype=np.float32)

def make_tf_dataset(prots, embed_path, labels, batch_size, out_channels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(prots)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(prots))

    # ---- These become closure variables:
    embed_path_closure = embed_path
    labels_closure = labels
    out_channels_closure = out_channels

    def load_and_process(prot):
        prot_str = prot.numpy().decode()
        prot_file = os.path.join(embed_path_closure, f'{prot_str}.txt')
        try:
            x = tf.io.parse_tensor(tf.io.read_file(prot_file), out_type=tf.float32)
            y = labels_closure[prot_str]
            if x.shape[0] <= 10000:
                return x.numpy(), y
            else:
                return np.zeros([1000, 1280], dtype=np.float32), np.zeros([out_channels_closure], dtype=np.float32)
        except Exception as e:
            return np.zeros([1000, 1280], dtype=np.float32), np.zeros([out_channels_closure], dtype=np.float32)

    def tf_load_and_process(prot):
        x, y = tf.py_function(func=load_and_process, inp=[prot], Tout=[tf.float32, tf.float32])
        x.set_shape([None, 1280])
        y.set_shape([out_channels])
        return x, y

    ds = ds.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 1280], [out_channels]))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    return ds


def run_testing(model, test_dataset, num_test_batches, save_path, ont, epoch):
    Y_true, Y_pred = [], []
    for X_test, Y_test in test_dataset.take(num_test_batches):
        y_pred = model.predict_on_batch(X_test)
        Y_true.append(Y_test)
        Y_pred.append(y_pred)
    Y_true = tf.concat(Y_true, axis=0)
    Y_pred = tf.concat(Y_pred, axis=0)
    TP, FP, FN = update_metrics(Y_pred.numpy(), Y_true.numpy(), [0 for _ in range(101)], [0 for _ in range(101)], [0 for _ in range(101)])
    model_checkpoint(os.path.join(save_path, f'finall_{ont}_recap.txt'), TP, FP, FN, epoch, ont)

if __name__ == '__main__':
    args = get_args()
    embed_path = args['embeddings']
    annots_path = args['annotations']
    ont = args['ontology']
    save_path = args['savepath']
    num_epochs = args['epochs']
    batch_size = args['batch']
    epoch_resume = args['resume']
    check_epochs = args['checkpoint']

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    split_path = os.path.join(save_path, 'split')
    if not os.path.isdir(split_path):
        os.mkdir(split_path)

    print('Loading labels...')
    labels_dir = os.path.join(annots_path, ont)
    labels = load_labels(labels_dir)
    all_prots = list(labels.keys())
    out_channels = labels[all_prots[0]].shape[0]

    print('Training net...')
    folds = load_folds(4, ont)
    print(f'Training for ontology of length {out_channels}...')

    if epoch_resume > 0:
        print('Resume mode')
        test_prots = load_prots(os.path.join(split_path, f'{ont}_testt.txt'))
        train_prots = load_prots(os.path.join(split_path, f'{ont}_trainn.txt'))
    else:
        test_prots = folds[-1]
        train_prots = list(chain.from_iterable(folds[:-1]))
        write_prots(test_prots, os.path.join(split_path, f'{ont}_testt.txt'))
        write_prots(train_prots, os.path.join(split_path, f'{ont}_trainn.txt'))

    num_train_batches = int(len(train_prots) / batch_size)
    num_test_batches = int(len(test_prots) / batch_size)

    epoch = epoch_resume
    while epoch < num_epochs:
        with strategy.scope():
            model = retrieve_model(ont, out_channels, print_summary=True)
        if epoch > 0:
            model.load_weights(os.path.join(save_path, f'finall_{ont}_{epoch - 1}.h5'))
        epochs = check_epochs - (epoch % check_epochs)
        print(f'Final train. Epochs {epoch + 1} - {epoch + epochs} / {num_epochs}.')
        print(f'Steps: {num_train_batches}')

        train_dataset = make_tf_dataset(train_prots, embed_path, labels, batch_size, out_channels, shuffle=True)

        model.fit(train_dataset, epochs=epochs, steps_per_epoch=num_train_batches)

        model.save_weights(os.path.join(save_path, f'finall_{ont}_{epoch + epochs - 1}.h5'))
        epoch += epochs

        #if epoch % check_epochs == 0:
        #    test_dataset = make_tf_dataset(test_prots, embed_path, labels, batch_size, out_channels, shuffle=False)
        #    run_testing(model, test_dataset, num_test_batches, save_path, ont, epoch)
