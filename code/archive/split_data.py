import time
from sklearn.model_selection import train_test_split

def split_data(X, y, testsize, randnum):
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testsize, random_state = randnum)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[1],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[1],1)
    print('Finished splitting after ' + str(time.time() - start) + ' sec')
    return X_train, X_test, y_train, y_test

