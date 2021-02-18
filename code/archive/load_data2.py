import cv2
import numpy as np
import os
import time
import split_data

def get_X_train(directory,size,trim):
    # set time stamp
    start = time.time()

    directory = directory + '/images_training_rev1/'
    print(directory)
    # determining the crop idx
    idx1 = int((size - trim) / 2)
    idx2 = int((size + trim) / 2)

    # getting the size of the data set so we can pre-allocate a numpy array
    N = len([file for file in os.listdir(directory) if '.jpg' in file])
    X = np.empty((N, trim, trim))

    # loop through all the files and add them to the data array
    count = 0
    fpaths = os.listdir(directory)
    fpaths.sort()
    exceptions = []

    i = 0
    for fpath in fpaths:
        try:
            X[i, :] = cv2.imread(directory + fpath, 0)[idx1:idx2, idx1:idx2].transpose(1, 0)
            count += 1
            print(str(i) + ' ' + fpath)
            i+=1
        except:
            print('===========')
            exceptions.append(fpath)
            pass
            i+=1

    print('Finished loading X after ' + str(time.time() - start) + 'sec')

    return X

def get_y_train(directory):
    # set time stamp
    start = time.time()

    # extract y_train information
    y_fn = directory + '/training_solutions_rev1.csv'
    y = np.genfromtxt(y_fn, delimiter = ',')
    y = y[1:, :]

    print('Finished loading input after ' + str(time.time() - start) + 'sec')

    return y

if __name__ == '__main__':

    # define input arguments
    directory = '/Users/Karen_Loscocco/Desktop/galaxy-zoo-the-galaxy-challenge'
    size = 424
    trim = 100
    testsize = 0.33
    randnum = 42

    X = get_X_train(directory,size,trim)

    y = get_y_train(directory)

    X_train, X_test, y_train, y_test = split_data.split_data(X, y, testsize, randnum)

    np.save('X_train', X_train)
    np.save('X_test', X_test)

    np.save('y_train', y_train)
    np.save('y_test', y_test)