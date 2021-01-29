from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import IPython
import numpy as np
import seaborn as sns
import sys
import time

if __name__ == '__main__':

    if len(sys.argv) > 1:
        regtype = sys.argv[1]
    else:
        regtype = 'reg'

    start = time.time()

    testsize = 0.10
    randnum = 42

    print('getting data')
    X_train = np.load('/Users/Karen_Loscocco/Documents/Galaxy-Morphologies-Classification.github.io/pca_reconstructed_100_100_2.npy')
    y_train = np.load('/Users/Karen_Loscocco/Documents/Galaxy-Morphologies-Classification.github.io/y_train_full_size.npy')

    #y_train = np.load('/Users/Karen_Loscocco/Desktop/galaxy-zoo-the-galaxy-challenge/Data_NPY/y_train.npy')

    y_sortedidx = np.argsort(y_train[:,0])
    y_train = y_train[y_sortedidx, 1:3]

    filter_idx = (y_train[:,0] >= 0.8) | (y_train[:,1] >= 0.8)
    y_train = y_train[filter_idx]

    y_train_hard = np.argmax(y_train, axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_hard, test_size = testsize, random_state = randnum)

    print('{} train samples'.format(X_train.shape[0]))
    print('{} test samples'.format(X_test.shape[0]))

    print('init model')

    if regtype == 'ridge':
        print('doing ridge')
        lr = Ridge(alpha=100)

        print('fit model')
        lr.fit(X_train,y_train)

        print('results')
        #score = lr.score(X_test,y_test)
        #print('score: {}'.format(score))

        predictions = lr.predict(X_test)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        #c1 = np.sum(y_test)
        #c0 = len(y_test) - c1

        c1idx, c2idx = (y_test == 0), (y_test == 1)

        c1t = np.sum(predictions[c1idx] == y_test[c1idx])
        c1f = np.sum(c1idx) - c1t
        c2t = np.sum(predictions[c2idx] == y_test[c2idx])
        c2f = np.sum(c2idx) - c2t

        score = (c1t + c2t) / (c1t + c2t + c1f + c2f)
        print('score: {}'.format(score))


    else:
        lr = LogisticRegression()#,normalize=True)

        print('fit model')
        lr.fit(X_train,y_train)

        print('results')
        score = lr.score(X_test,y_test)
        print('score: {}'.format(score))

        predictions = lr.predict(X_test)

    #labels = ['Class1','Class2']

    IPython.embed()
    cm = metrics.confusion_matrix(y_test, predictions)#, #abel_encoder={0: 'Class1', 1: 'Class2'})

    #print(predictions[:20])

    #classes=classes, label_encoder={0: "unoccupied", 1: "occupied"}

    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0:.4f}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.show()





