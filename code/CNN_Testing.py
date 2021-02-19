import numpy as np
import tensorflow as tf
import time
import sys
from keras.utils import to_categorical
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    start = time.time()

    model = tf.keras.models.load_model('C:\\Users\\yjia1\\projects\\model_0717_2120.h5')

    X_test = np.load('C:\\Users\\yjia1\\projects\\X_test_pca.npy')
    y_test = np.load('C:\\Users\\yjia1\\projects\\y_test_pca.npy')

    y_predict = model.predict(X_test)

    y_predict[y_predict > 0.5] = 1
    y_predict[y_predict <= 0.5] = 0

    y_predict_flat = np.argmax(y_predict, axis = 1)
    y_test_flat = np.argmax(y_test, axis = 1)

    print('y_pred')
    print(y_predict_flat[:10])
    print(y_test_flat[:10])
    #print('y_test')
    #print(y_test[:10])


    #score = model.evaluate(X_test, y_test, verbose = 0)

    # predictions = lr.predict(X_test)
    # predictions[predictions > 0.5] = 1
    # predictions[predictions <= 0.5] = 0

    c1idx, c2idx = (y_test == 0), (y_test == 1)

    c1t = np.sum(y_predict[c1idx] == y_test[c1idx])
    c1f = np.sum(c1idx) - c1t
    c2t = np.sum(y_predict[c2idx] == y_test[c2idx])
    c2f = np.sum(c2idx) - c2t

    score = (c1t + c2t) / (c1t + c2t + c1f + c2f)
    print('score: {}'.format(score))

    cm = metrics.confusion_matrix(y_test_flat, y_predict_flat)

    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0:.4f}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.show()

    print('Testing completed after ' + str(time.time() - start) + ' sec')
