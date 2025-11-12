import numpy as np
from numpy.typing import NDArray


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
