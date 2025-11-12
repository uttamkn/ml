"""Neural Network"""

from __future__ import annotations

from typing import List, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator


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

        for x, y in mini_batch:
            # this will give us the gradient of Cost function w.r.t w and b (same shape as self.weights and self.biases)
            # basically tells us how much to change for all the weights and biases to move towards the minimum
            delta_nabla_C_w, delta_nabla_C_b = self.backprop(x, y)

            # find the sum of gradients of all the input output pairs in the minibatch
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
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

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
        # [delta = partial C / partial z]
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # formulas for gradients for C/w and C/b (can be derived easily)
        # c/w = c/z . z/w = delta . activation(prev)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # c/b = c/z . z/b = delta . 1
        nabla_b[-1] = delta

        # ignore first layer since its the input layer and the last layer since its already calculated
        for i in range(2, self.num_of_layers):
            z = zs[-i]
            # here the formula for delta changes because the cost does not directly depend on the activations
            # `delta` is the error of the next layer (already computed).
            # weights[next] tells you how much current layer activations contributed to each neuron in the next layer.
            # So, when you multiply weights . old_delta, you're “bringing back” the error from the next layer to this one
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sigmoid_prime(z)

            # c/w
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())

            # c/b
            nabla_b[-i] = delta

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


# ============================================
# 🔹 Activation Functions
# ============================================


class Activation:
    """Base class for activation functions."""

    def fn(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def prime(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError


class Sigmoid(Activation):
    def fn(self, z):
        return 1 / (1 + np.exp(-z))

    def prime(self, z):
        s = self.fn(z)
        return s * (1 - s)


class ReLU(Activation):
    def fn(self, z):
        return np.maximum(0, z)

    def prime(self, z):
        return (z > 0).astype(float)


class Tanh(Activation):
    def fn(self, z):
        return np.tanh(z)

    def prime(self, z):
        t = np.tanh(z)
        return 1 - t**2


class Softmax(Activation):
    def fn(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def prime(self, z):
        # derivative handled inside cost when using cross-entropy
        raise NotImplementedError(
            "Softmax derivative is usually combined with cross-entropy."
        )


# ============================================
# 🔹 Cost (Loss) Functions
# ============================================


class CostFunction:
    """Base class for cost functions."""

    def fn(self, output_activations: NDArray, y: NDArray) -> float:
        raise NotImplementedError

    def delta(
        self,
        output_activations: NDArray,
        y: NDArray,
        z: NDArray,
        activation: Activation,
    ) -> NDArray:
        """Return δ = ∂C/∂z for the output layer."""
        raise NotImplementedError


class MeanSquaredError(CostFunction):
    def fn(self, output_activations: NDArray, y: NDArray) -> float:
        return float(0.5 * np.linalg.norm(output_activations - y) ** 2)

    def delta(self, output_activations, y, z, activation):
        return (output_activations - y) * activation.prime(z)


class CrossEntropy(CostFunction):
    def fn(self, output_activations, y):
        # Add epsilon to avoid log(0)
        eps = 1e-10
        return -np.sum(y * np.log(output_activations + eps))

    def delta(self, output_activations, y, z, activation):
        # For sigmoid or softmax outputs: simplified gradient
        return output_activations - y


# ============================================
# 🔹 Network Config Schema
# ============================================


class NetworkConfig(BaseModel):
    layer_sizes: List[int] = Field(..., description="Number of neurons per layer.")
    activation: Literal["sigmoid", "relu", "tanh", "softmax"] = Field(
        "sigmoid", description="Activation function name."
    )
    cost_function: Literal["mse", "cross_entropy"] = Field(
        "cross_entropy", description="Cost function name."
    )
    learning_rate: float = Field(0.1, gt=0)
    epochs: int = Field(30, ge=1)
    mini_batch_size: int = Field(10, ge=1)
    verbose: bool = True

    @field_validator("layer_sizes")
    def check_layers(cls, v):
        if len(v) < 2:
            raise ValueError("Network must have at least input and output layers.")
        return v


# ============================================
# 🔹 Neural Network Class
# ============================================


class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.num_layers = len(config.layer_sizes)
        self.sizes = config.layer_sizes

        # choose activation and cost
        self.activation_fn = self._make_activation(config.activation)
        self.cost_fn = self._make_cost(config.cost_function)

        # random init
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def _make_activation(self, name: str) -> Activation:
        return {
            "sigmoid": Sigmoid(),
            "relu": ReLU(),
            "tanh": Tanh(),
            "softmax": Softmax(),
        }[name]

    def _make_cost(self, name: str) -> CostFunction:
        return {
            "mse": MeanSquaredError(),
            "cross_entropy": CrossEntropy(),
        }[name]

    # ----------------------------------------
    # Feedforward
    # ----------------------------------------
    def feedforward(self, a: NDArray) -> NDArray:
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.activation_fn.fn(np.dot(w, a) + b)

        # handle last layer separately if softmax
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        if isinstance(self.activation_fn, Softmax):
            a = Softmax().fn(z)
        else:
            a = self.activation_fn.fn(z)
        return a

    # ----------------------------------------
    # Training (SGD)
    # ----------------------------------------
    def SGD(self, training_data, test_data=None):
        n = len(training_data)
        for epoch in range(self.config.epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + self.config.mini_batch_size]
                for k in range(0, n, self.config.mini_batch_size)
            ]
            for batch in mini_batches:
                self.update_mini_batch(batch)

            if self.config.verbose:
                if test_data:
                    print(
                        f"Epoch {epoch + 1}: {self.evaluate(test_data)} / {len(test_data)}"
                    )
                else:
                    print(f"Epoch {epoch + 1} complete")

    # ----------------------------------------
    # Update via Backpropagation
    # ----------------------------------------
    def update_mini_batch(self, mini_batch):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        eta = self.config.learning_rate
        m = len(mini_batch)
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    # ----------------------------------------
    # Backpropagation
    # ----------------------------------------
    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_fn.fn(z)
            activations.append(activation)

        # output layer
        delta = self.cost_fn.delta(activations[-1], y, zs[-1], self.activation_fn)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # backprop through hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_fn.prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_w, nabla_b

    # ----------------------------------------
    # Evaluation
    # ----------------------------------------
    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(pred == truth) for pred, truth in results)
