from __future__ import annotations

from typing import List, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator
from nn.activation import Activation, ReLU, Sigmoid, Softmax, Tanh
from nn.cost import CostFunction, CrossEntropy, MeanSquaredError


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


class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.num_layers = len(config.layer_sizes)
        self.sizes = config.layer_sizes

        self.activation_fn = self._make_activation(config.activation)
        self.cost_fn = self._make_cost(config.cost_function)

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
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = self.activation_fn.prime(z)
            delta = np.dot(self.weights[-layer + 1].T, delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].T)

        return nabla_w, nabla_b

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(pred == truth) for pred, truth in results)
