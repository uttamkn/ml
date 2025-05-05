"""Neural Network"""

import numpy as np


class NN:
    def __init__(self, sizes: np.typing.NDArray):
        self.num_of_layers = len(sizes)  # number of layers
        self.sizes = sizes  # number of neurons in each layer (size of each layer)

        # array of m x n matrices of length num_of_layers - 1 (because the first one is considered as the input layer)
        # m - number of neurons in the current layer
        # n - number of neurons in the previous layer
        # if there m neurons in the nth layer then, the weights can be represented by a (m) x (number of neurons in n-1 layer) matrix
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # array of size x 1 matrices of length num_of_layers - 1 (because the first one is considered as the input layer)
        # each neuron will have its own bias
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def feedforward(self, a: np.typing.NDArray):
        """
        Input: a vector of features

        Output: output array of the nn
        """
        # will iterate num_of_layers-1 times
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b  # logit (this is an array) z = wx+b
            a = sigmoid(z)  # activation (this is an array) a = sigma(z)

        return a

    # stochastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        for e in range(epochs):
            print(f"Epoch: {e}")
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # this will update the weights and biases of the model based on the mini_batch (eta - learning rate)
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(f"Epoch {e}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {e} complete")

    def update_mini_batch(self, mini_batch, eta):
        # [delta C = nabla C * delta v]
        # nabla C (gradient) gives us the relationship between the change in parameters v (weights and biases in this case) and the change in C (cost)
        # gradient tells how the change in v affects C
        # so we use this to find how much of w and b should be changed to minimize cost
        # that can be done by calculating - eta * (nabla C) [eta is the learning rate]
        # and the same thing is calculated for all the input output pairs in this minibatch and the average of that is subracted with the weights and biases

        # initializing to 0
        nabla_C_w = [np.zeros(w.shape) for w in self.weights]
        nabla_C_b = [np.zeros(b.shape) for b in self.biases]

        # find the sum of gradients of all the input output pairs in the minibatch
        for x, y in mini_batch:
            delta_nabla_C_w, delta_nabla_C_b = self.backprop(x, y)
            nabla_C_w = [nw + dnw for nw, dnw in zip(nabla_C_w, delta_nabla_C_w)]
            nabla_C_b = [nb + dnb for nb, dnb in zip(nabla_C_b, delta_nabla_C_b)]

        # update weights and biases by subracting them with average_gradient*learning rate
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_C_w)
        ]

        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_C_b)
        ]

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_w, nabla_b)`` representing the gradient for the cost function C_x.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # ======= feedforward ========
        activation = x
        # list to store all the activations, layer by layer (x is the input layer)
        activations = [x]
        zs = []  # list to store all the z vectors (logits), layer by layer
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # ======== backward pass ========
        # find the loss: now the network has made a prediction (activations[-1]) and you compare it with the correct answer y
        # delta = loss * how sensitive the activation is to changes (sigmoid_prime)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # formulas for gradients for C/w and C/b (can be derived easily)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        # ignore first layer since its the input layer and the last layer since its already calculated
        for i in range(2, self.num_of_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        return (nabla_w, nabla_b)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x /
        partial a for the output activations."""
        return output_activations - y


# utils
def sigmoid(z: np.typing.NDArray):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
