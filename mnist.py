# mnist_loader.py - Load MNIST dataset for neural network training

import pickle
import gzip
import numpy as np


# Load raw MNIST data from gzip file
def load_data():
    with gzip.open("./data/mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data


# Load and reshape MNIST data for neural network usage
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data


# Convert digit to one-hot encoded vector
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    from nn import NN
    import numpy as np

    training_data, _, test_data = load_data_wrapper()

    net = NN(np.array([784, 30, 10]))
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
