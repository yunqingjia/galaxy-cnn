from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook as tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import sys
import time


start = time.time()

X_train = np.load('C:\\Users\\yjia1\\projects\\X_train_100_100.npy')
y_train = np.load('C:\\Users\\yjia1\\projects\\y_train_full_size.npy')

print('X_train shape: {}'.format(X_train.shape[0]))

# Data Preprocessing
y_sortedidx = np.argsort(y_train[:, 0])
y_train = y_train[y_sortedidx, 1:3]

filter_idx = (y_train[:,0] >= 0.8) | (y_train[:,1] >= 0.8)
y_train = y_train[filter_idx]
X_train = X_train[filter_idx]

size = X_train.shape[1]

y_train_hard = to_categorical(np.argmax(y_train, axis = 1))

num_samples = len(y_train_hard)

print('Training on {} data points'.format(num_samples))

X_train_flat = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))

print('X_train_flat shape: {}'.format(X_train_flat.shape[0]))

#X_mini_train = X_train_flat[:50]

x = StandardScaler().fit_transform(X_train_flat)

print('x shape: {}'.format(x.shape[0]))

pca = PCA(0.95)

lower_dimensional_data = pca.fit_transform(x)

print('lower_dimensional_data shape: {}'.format(lower_dimensional_data.shape[0]))

print('Finished fitting with {} components. Total time: {} Seconds'.format(pca.n_components_, time.time()-start))

approximation = pca.inverse_transform(lower_dimensional_data)

print('approximation shape: {}'.format(approximation.shape[0]))

np.save('pca_low_dim_100_100_2', lower_dimensional_data)
np.save('pca_reconstructed_100_100_2', approximation)

# for i in range(50):

#     plt.figure(figsize=(8,4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(X_train[i].reshape(size,size))
#                   #cmap = plt.cm.gray, interpolation='nearest',
#                   #clim=(0, 255))
#     plt.xlabel('{} components'.format(size*size), fontsize = 14)
#     plt.title('Original Image', fontsize = 20);

#     plt.subplot(1, 2, 2);
#     plt.imshow(approximation[i].reshape(size,size))
#                   #cmap = plt.cm.gray, interpolation='nearest',
#                   #clim=(0, 255));
#     plt.xlabel('{} components'.format(pca.n_components_), fontsize = 14)
#     plt.title('95% of Explained Variance', fontsize = 20);

#     plt.savefig('./pca_pics_full_size/galaxy_fs_pca_{}.jpeg'.format(i))

#     plt.close()



