
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    start = time.time()

    testsize = 0.10
    randnum = 42

    print('getting data')
    X_train = np.load('/Users/Karen_Loscocco/Documents/Galaxy-Morphologies-Classification.github.io/pca_reconstructed_100_100_2.npy')
    y_train = np.load('/Users/Karen_Loscocco/Documents/Galaxy-Morphologies-Classification.github.io/y_train_full_size.npy')


    y_sortedidx = np.argsort(y_train[:,0])
    y_train = y_train[y_sortedidx, 1:3]

    filter_idx = (y_train[:,0] >= 0.8) | (y_train[:,1] >= 0.8)
    y_train = y_train[filter_idx]

    y_train_hard = np.argmax(y_train, axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_hard, test_size = testsize, random_state = randnum)

    print('{} train samples'.format(X_train.shape[0]))
    print('{} test samples'.format(X_test.shape[0]))

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

    clf.fit(X_train, y_train)

    print(clf.feature_importances_)

    ypredict = clf.predict(X_test)

    score = clf.score(X_test,y_test)

    cm = metrics.confusion_matrix(y_test, ypredict)

    #print(predictions[:20])

    #classes=classes, label_encoder={0: "unoccupied", 1: "occupied"}

    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0:.4f}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.show()


