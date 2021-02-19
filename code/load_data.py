import cv2
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

class load_data():

    def __init__(self, directory):
        self.directory = directory

    def get_X(self, size, trim):
        start = time.time()

        dirx = self.directory + '\\images_training_rev1\\images_training_rev1'

        # determining the crop idx
        idx1 = int((size - trim) / 2)
        idx2 = int((size + trim) / 2)

        # getting the size of the data set so we can pre-allocate a numpy array
        N = len([file for file in os.listdir(dirx) if '.jpg' in file])
        print(N)
        X = np.empty((N, trim, trim))

        # loop through all the files and add them to the data array
        count = 0
        fpaths = os.listdir(dirx)
        fpaths.sort()
        exceptions = []

        i = 0
        for fpath in fpaths:
            try:
                X[i, :] = cv2.imread(dirx + fpath, 0)[idx1:idx2, idx1:idx2].transpose(1, 0)
                count += 1
                print(str(i) + ' ' + fpath)
                i+=1
                # print("hello?")
            except:
                # print('===========')
                exceptions.append(fpath)
                pass
                i+=1

        print('Finished loading X after ' + str(time.time() - start) + 'sec')

        return X

    def get_y(self):
        start = time.time()

        diry = self.directory + '\\training_solutions_rev1.csv'
        
        # extract y_train information
        y = np.genfromtxt(diry, delimiter = ',')
        y = y[1:, :]

        print('Finished loading input after ' + str(time.time() - start) + 'sec')

        return y

    def split_data(X, y, testsize, randnum):
            start = time.time()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testsize, random_state = randnum)
            X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[1],1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[1],1)
            print('Finished splitting after ' + str(time.time() - start) + ' sec')
            
            return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    # define input arguments
    directory = 'C:\\Users\\yjia1\\projects\\galaxy-zoo-the-galaxy-challenge'
    size = 424
    trim = 100
    testsize = 0.33
    randnum = 42

    ld = load_data(directory)

    X = ld.get_X(size,trim)

    y = ld.get_y()

    X_train, X_test, y_train, y_test = ld.split_data(X, y, testsize, randnum)

    np.save('X_train', X_train)
    np.save('X_test', X_test)

    np.save('y_train', y_train)
    np.save('y_test', y_test)