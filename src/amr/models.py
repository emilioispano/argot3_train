#!/usr/bin/python3

import tensorflow as tf
from tensorflow.keras import layers, metrics, regularizers
from tensorflow import keras
import os


def get_saved_model(model_path):
    model = keras.models.load_model(model_path)

    model_path, model_folder = os.path.split(model_path)
    if not model_folder:
        model_path, model_folder = os.path.split(model_path)

    ont, fold, epoch = model_folder.split('_')

    return model, int(fold), int(epoch)


def custom_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-4  # Small constant to avoid log(0)
    w = 20.0  # Make sure w is a float

    # Cast y_true to float32
    y_true = tf.cast(y_true, tf.float32)

    # Clip values to avoid log(0) or log(1)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    # Calculate binary cross-entropy loss
    loss = - (y_true * w * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # Sum the loss across all elements
    # return tf.reduce_mean(loss)
    return tf.reduce_sum(loss)


def residual_block(x, filters, kernel_size, strides):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def recursive_block(x, recursize):
    shortcut = x
    x = layers.Bidirectional(layers.LSTM(recursize, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Bidirectional(layers.LSTM(recursize, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def get_model_amr(out_channels):
    input = layers.Input(shape=(None, 1280))
    x = layers.Masking(mask_value=0.0)(input)

    x = layers.Conv1D(16, kernel_size=3, strides=1, padding='same')(x)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = residual_block(x, 16, kernel_size=3, strides=1)

    x = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(x)
    x = recursive_block(x, 8)
    x = recursive_block(x, 8)
    x = recursive_block(x, 8)

    x = layers.GlobalAveragePooling1D()(x)

    #x = layers.Dense(out_channels, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.BinaryAccuracy()])

    return model


def next_get_model_amr(out_channels):
    input = layers.Input(shape=(None, 1280))

    x = layers.Masking(mask_value=0.0)(input)

    x = layers.Conv1D(16, kernel_size=3, strides=1, padding='same')(x)
    x = residual_block(x, 16, kernel_size=3, strides=1)

    x = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(x)
    x = recursive_block(x, 8)

    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(x)
    x = recursive_block(x, 8)

    x = layers.Conv1D(16, kernel_size=7, strides=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(16, kernel_size=7, strides=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(x)
    x = recursive_block(x, 8)
    x = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(x)
    x = recursive_block(x, 8)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(out_channels, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.BinaryAccuracy()])

    return model


def get_model_cco(out_channels):
    input = layers.Input(shape=(None, 1280))

    x = layers.Masking(mask_value=0.0)(input)

    x = layers.Conv1D(16, kernel_size=3, strides=1, padding='same')(x)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 16)
    x = recursive_block(x, 16)

    x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 32, kernel_size=3, strides=1)
    x = residual_block(x, 32, kernel_size=3, strides=1)
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 128, kernel_size=3, strides=1)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 128)
    x = recursive_block(x, 128)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(out_channels, activation='relu')(x)
    output = layers.Dense(out_channels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    #model.compile(loss=new_binary_crossentropy, optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
    model.compile(loss=custom_binary_crossentropy, optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.CategoricalAccuracy()])

    return model


def get_model_mfo(out_channels):
    input = layers.Input(shape=(None, 1280))

    x = layers.Masking(mask_value=0.0)(input)

    x = layers.Conv1D(16, kernel_size=3, strides=1, padding='same')(x)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 16)
    x = recursive_block(x, 16)

    x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 32, kernel_size=3, strides=1)
    x = residual_block(x, 32, kernel_size=3, strides=1)
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 128, kernel_size=3, strides=1)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 128)
    x = recursive_block(x, 128)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 256, kernel_size=3, strides=1)
    x = layers.Conv1D(256, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 256)
    x = recursive_block(x, 256)
    x = layers.Bidirectional(layers.LSTM(256))(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(out_channels, activation='relu')(x)
    output = layers.Dense(out_channels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    #model.compile(loss=new_binary_crossentropy, optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
    model.compile(loss=custom_binary_crossentropy, optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.CategoricalAccuracy()])

    return model


def get_model_bpo(out_channels):
    input = layers.Input(shape=(None, 1280))

    x = layers.Masking(mask_value=0.0)(input)

    x = layers.Conv1D(16, kernel_size=3, strides=1, padding='same')(x)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = residual_block(x, 16, kernel_size=3, strides=1)
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 16)
    x = recursive_block(x, 16)

    x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 32, kernel_size=3, strides=1)
    x = residual_block(x, 32, kernel_size=3, strides=1)
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 128, kernel_size=3, strides=1)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 128)
    x = recursive_block(x, 128)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 256, kernel_size=3, strides=1)
    x = layers.Conv1D(256, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, kernel_size=5, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 256)
    x = recursive_block(x, 256)
    x = layers.Bidirectional(layers.LSTM(256))(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(out_channels, activation='relu')(x)
    output = layers.Dense(out_channels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    #model.compile(loss=new_binary_crossentropy, optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
    model.compile(loss=custom_binary_crossentropy, optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.CategoricalAccuracy()])

    return model


def alternative(out_channels):
    input = layers.Input(shape=(None, 1280))

    x = layers.Masking(mask_value=0.0)(input)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)
    x = recursive_block(x, 32)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, kernel_size=3, strides=1)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)
    x = recursive_block(x, 64)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(out_channels, activation='relu')(x)
    output = layers.Dense(out_channels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(loss=custom_binary_crossentropy, optimizer='adam', metrics=[metrics.CategoricalAccuracy()])

    return model
