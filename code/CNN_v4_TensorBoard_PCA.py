from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import csv
import cv2
import numpy as np
import os
import skimage
import sys
import time
import datetime
import tensorflow as tf
from keras.callbacks import TensorBoard
from os import makedirs
from os.path import exists, join

import split_data
#import build_model

import IPython

# input arguments
# X, y np arrays
# testsize, randnum
# nodels, ksize, input_size, num_class

if __name__ == '__main__':

    start = time.time()
    testsize = 0.10
    randnum = 42
    nodels = [64, 32]
    ksize = 3
    num_class = 2
    input_size = 100
    epochs = 50

    X_train = np.load('/Users/Karen_Loscocco/Documents/Galaxy-Morphologies-Classification.github.io/pca_reconstructed_100_100_2.npy')
    y_train = np.load('/Users/Karen_Loscocco/Documents/Galaxy-Morphologies-Classification.github.io/y_train_full_size.npy')

    X_train_reshape = X_train.reshape(X_train.shape[0],100,100,1)

    y_sortedidx = np.argsort(y_train[:,0])
    y_train = y_train[y_sortedidx, 1:3]

    filter_idx = (y_train[:,0] >= 0.8) | (y_train[:,1] >= 0.8)
    y_train = y_train[filter_idx]

    y_train_hard = to_categorical(np.argmax(y_train, axis = 1))

    X_train, X_test, y_train, y_test = train_test_split(X_train_reshape, y_train_hard, test_size = testsize, random_state = randnum)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = testsize, random_state = randnum)

    #X_train, X_validate, y_train, y_validate = split_data.split_data(X_train_reshape, y_train_hard, testsize, randnum)

    np.save('X_test_pca', X_test)
    np.save('y_test_pca', y_test)

    print('{} train samples'.format(X_train.shape[0]))
    print('{} test samples'.format(X_validate.shape[0]))

    batch_size = 128 #int(X_train.shape[0]/2) #use less memory, train faster with the mini batches

    ######TENSOR BOARD######
    # save class labels to disk to color data points in TensorBoard accordingly
    d_t = datetime.datetime.now().strftime('%m%d_%H%M')
    log_dir = './logs_{}'.format(d_t)
    mname = 'model_{}.h5'.format(d_t)

    if not exists(log_dir):
      makedirs(log_dir)

    with open(join(log_dir, 'metadata.tsv'), 'w') as f:
      np.savetxt(f, y_validate)

    # with open(log_dir + '/metadata.tsv', 'w') as f:
    #   np.savetxt(f, y_validate)

    print('\nTo View TensorBoard, Run the following: tensorboard --logdir="{}"\n'.format(log_dir))

    tb = TensorBoard(log_dir = log_dir,
                     histogram_freq = 1,
                     write_graph = True,
                     write_images = True,
                     embeddings_freq=1,
                     embeddings_layer_names=['features'],
                     embeddings_metadata='metadata.tsv',
                     embeddings_data=X_validate,
                     batch_size = batch_size)

    ########################

    #build_model1
    #####model = build_model.build_model(nodels, ksize, input_size, num_class)
    # model = build_model2.build_model(input_size, num_class)
    # model = build_model_ammar.build_model(input_size, num_class)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(input_size, input_size, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='features'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax'))


    model.compile(loss = 'categorical_crossentropy',
                  optimizer = keras.optimizers.Adadelta(), #'Adam',
                  metrics=['mae', 'acc'])
    print('Finished compiling')

    model.fit(X_train, y_train,
              batch_size = batch_size,
              callbacks = [tb],
              epochs = epochs,
              verbose = 1,
              validation_data = (X_validate, y_validate)
              )

    print('Finished fitting')

    model.save(mname)
    print('Model saved')

    #model.summary()
    score = model.evaluate(X_validate, y_validate, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('%s: %.2f%%' % (model.metrics_names[1], score[1]*100))

    print('Entire algorithm completed after ' + str(time.time() - start) + ' sec')
